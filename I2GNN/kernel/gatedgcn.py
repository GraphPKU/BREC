import torch
from torch import Tensor
from torch.nn import Parameter as Param
from torch_sparse import SparseTensor, matmul

from torch_geometric.nn.conv import MessagePassing
from torch_geometric.typing import Adj, OptTensor

from torch_geometric.nn.inits import uniform


class HeteroGatedGraphConv(MessagePassing):
    def __init__(self, out_channels: int, num_layers: int, aggr: str = 'add',
                 bias: bool = True, **kwargs):
        super().__init__(aggr=aggr, **kwargs)

        self.out_channels = out_channels
        self.num_layers = num_layers

        self.weight_0 = Param(Tensor(num_layers, out_channels, out_channels))
        self.weight_1 = Param(Tensor(num_layers, out_channels, out_channels))
        self.rnn = torch.nn.GRUCell(out_channels, out_channels, bias=bias)

        self.reset_parameters()

    def reset_parameters(self):
        uniform(self.out_channels, self.weight_0)
        uniform(self.out_channels, self.weight_1)
        self.rnn.reset_parameters()


    def forward(self, x: Tensor, edge_index: Adj, z: Tensor,
                edge_weight: OptTensor = None) -> Tensor:
        """"""

        if x.size(-1) > self.out_channels:
            raise ValueError('The number of input channels is not allowed to '
                             'be larger than the number of output channels')

        if x.size(-1) < self.out_channels:
            zero = x.new_zeros(x.size(0), self.out_channels - x.size(-1))
            x = torch.cat([x, zero], dim=1)

        # create mask for heterogenous message passing
        mask_1 = z
        mask_0 = 1 - z

        for i in range(self.num_layers):
            m_0 = mask_0 * torch.matmul(x, self.weight_0[i])
            m_1 = mask_1 * torch.matmul(x, self.weight_1[i])
            # propagate_type: (x: Tensor, edge_weight: OptTensor)
            m = self.propagate(edge_index, x=m_0+m_1, edge_weight=edge_weight,
                               size=None)
            x = self.rnn(m, x)

        return x


    def message(self, x_j: Tensor, edge_weight: OptTensor):
        return x_j if edge_weight is None else edge_weight.view(-1, 1) * x_j

    def message_and_aggregate(self, adj_t: SparseTensor, x: Tensor) -> Tensor:
        return matmul(adj_t, x, reduce=self.aggr)

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}({self.out_channels}, '
                f'num_layers={self.num_layers})')