from typing import Optional, Tuple, Union
import torch
import torch.nn.functional as F
from torch import Tensor
from torch.nn import Parameter
from torch.nn import Parameter as Param
from torch_scatter import scatter
from torch_sparse import SparseTensor, masked_select_nnz, matmul

from torch_geometric.nn.conv import MessagePassing
from torch_geometric.typing import Adj, OptTensor
from torch_geometric.nn.inits import glorot, zeros

try:
    from pyg_lib.ops import segment_matmul  # noqa
    _WITH_PYG_LIB = True
except ImportError:
    _WITH_PYG_LIB = False

    def segment_matmul(inputs: Tensor, ptr: Tensor, other: Tensor) -> Tensor:
        raise NotImplementedError


@torch.jit._overload
def masked_edge_index(edge_index, edge_mask):
    # type: (Tensor, Tensor) -> Tensor
    pass


@torch.jit._overload
def masked_edge_index(edge_index, edge_mask):
    # type: (SparseTensor, Tensor) -> SparseTensor
    pass


def masked_edge_index(edge_index, edge_mask):
    if isinstance(edge_index, Tensor):
        return edge_index[:, edge_mask]
    else:
        return masked_select_nnz(edge_index, edge_mask, layout='coo')


### Gated RGCN Convolution
class GatedRGCNConv(MessagePassing):
    def __init__(
            self,
            in_channels: Union[int, Tuple[int, int]],
            out_channels: int,
            num_relations: int,
            num_bases: Optional[int] = None,
            num_blocks: Optional[int] = None,
            aggr: str = 'mean',
            root_weight: bool = True,
            is_sorted: bool = False,
            bias: bool = True,
            **kwargs,
    ):
        kwargs.setdefault('aggr', aggr)
        super().__init__(node_dim=0, **kwargs)
        self._WITH_PYG_LIB = torch.cuda.is_available() and _WITH_PYG_LIB
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_bases = num_bases
        self.num_blocks = num_blocks
        self.num_relations = num_relations

        if isinstance(in_channels, int):
            in_channels = (in_channels, in_channels)
        self.in_channels_l = in_channels[0]

        if num_bases is not None:
            self.weight = Parameter(
                torch.Tensor(num_bases, in_channels[0], out_channels))
            self.comp = Parameter(torch.Tensor(num_relations, num_bases))

        elif num_blocks is not None:
            assert (in_channels[0] % num_blocks == 0
                    and out_channels % num_blocks == 0)
            self.weight = Parameter(
                torch.Tensor(num_relations, num_blocks,
                             in_channels[0] // num_blocks,
                             out_channels // num_blocks))
            self.register_parameter('comp', None)

        else:
            self.weight = Parameter(
                torch.Tensor(num_relations, in_channels[0], out_channels))
            self.register_parameter('comp', None)

        if root_weight:
            self.root = Param(torch.Tensor(in_channels[1], out_channels))
        else:
            self.register_parameter('root', None)

        if bias:
            self.bias = Param(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)

        # define GRU unit
        self.rnn = torch.nn.GRUCell(input_size=out_channels, hidden_size=out_channels, bias=bias)

        self.reset_parameters()

    def reset_parameters(self):
        glorot(self.weight)
        glorot(self.comp)
        glorot(self.root)
        zeros(self.bias)
        self.rnn.reset_parameters()

    def forward(self, x, edge_index, edge_type=None):
        # Convert input features to a pair of node features or node indices.
        if isinstance(x, tuple):
            x_l = x[0]
        else:
            x_l = x
        if x_l is None:
            x_l = torch.arange(self.in_channels_l, device=self.weight.device)

        x_r: Tensor = x_l
        if isinstance(x, tuple):
            x_r = x[1]

        size = (x_l.size(0), x_r.size(0))

        if isinstance(edge_index, SparseTensor):
            edge_type = edge_index.storage.value()
        assert edge_type is not None

        # propagate_type: (x: Tensor, edge_type_ptr: OptTensor)
        out = torch.zeros(x_r.size(0), self.out_channels, device=x_r.device)

        weight = self.weight
        if self.num_bases is not None:  # Basis-decomposition =================
            weight = (self.comp @ weight.view(self.num_bases, -1)).view(
                self.num_relations, self.in_channels_l, self.out_channels)

        if self.num_blocks is not None:  # Block-diagonal-decomposition =====

            if x_l.dtype == torch.long and self.num_blocks is not None:
                raise ValueError('Block-diagonal decomposition not supported '
                                 'for non-continuous input features.')

            for i in range(self.num_relations):
                tmp = masked_edge_index(edge_index, edge_type == i)
                h = self.propagate(tmp, x=x_l, edge_type_ptr=None, size=size)
                h = h.view(-1, weight.size(1), weight.size(2))
                h = torch.einsum('abc,bcd->abd', h, weight[i])
                out += h.contiguous().view(-1, self.out_channels)
        else:  # No regularization/Basis-decomposition ========================
            if self._WITH_PYG_LIB and isinstance(edge_index, Tensor):
                if not self.is_sorted:
                    if (edge_type[1:] < edge_type[:-1]).any():
                        edge_type, perm = edge_type.sort()
                        edge_index = edge_index[:, perm]
                edge_type_ptr = torch.ops.torch_sparse.ind2ptr(
                    edge_type, self.num_relations)
                out = self.propagate(edge_index, x=x_l,
                                     edge_type_ptr=edge_type_ptr, size=size)
            else:
                for i in range(self.num_relations):
                    tmp = masked_edge_index(edge_index, edge_type == i)
                    if x_l.dtype == torch.long:
                        out += self.propagate(tmp, x=weight[i, x_l],
                                              edge_type_ptr=None, size=size)
                    else:
                        h = self.propagate(tmp, x=x_l, edge_type_ptr=None,
                                           size=size)
                        out = out + (h @ weight[i])

        root = self.root
        if root is not None:
            # use GRU unit
            out = self.rnn(out, x_r @ root)
            # out += root[x_r] if x_r.dtype == torch.long else x_r @ root

        if self.bias is not None:
            out += self.bias

        return out

    def message(self, x_j: Tensor, edge_type_ptr: OptTensor) -> Tensor:
        if edge_type_ptr is not None:
            return segment_matmul(x_j, edge_type_ptr, self.weight)

        return x_j

    def message_and_aggregate(self, adj_t: SparseTensor, x: Tensor) -> Tensor:
        adj_t = adj_t.set_value(None)
        return matmul(adj_t, x, reduce=self.aggr)

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}({self.in_channels}, '
                f'{self.out_channels}, num_relations={self.num_relations})')