import torch
from torch_scatter import scatter
from torch_geometric.utils import to_undirected


class IdentityMapping(torch.autograd.Function):
    @staticmethod
    def forward(ctx, mask):
        """
        Given node mask, return identity, direct grad on the mask

        :param ctx:
        :param mask:
        :return:
        """
        assert mask.dtype == torch.float  # must be differentiable
        return mask

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output


class CustomedIdentityMapping(torch.autograd.Function):
    @staticmethod
    def forward(ctx, mask):
        """
        Given node mask, return identity, direct grad on the mask

        :param ctx:
        :param mask:
        :return:
        """
        assert mask.dtype == torch.float  # must be differentiable
        return torch.ones_like(mask, device=mask.device, dtype=mask.dtype)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output


class Nodemask2Edgemask(torch.autograd.Function):

    @staticmethod
    def forward(ctx, mask, *args):
        """
        Given node masks, return edge masks as edge weights for message passing.

        :param ctx:
        :param mask:
        :param args:
        :return:
        """
        assert mask.dtype == torch.float  # must be differentiable
        edge_index, n_nodes = args
        ctx.save_for_backward(mask, edge_index[1], n_nodes)
        return nodemask2edgemask(mask, edge_index)

    @staticmethod
    def backward(ctx, grad_output):
        _, edge_index_col, n_nodes = ctx.saved_tensors
        final_grad = scatter(grad_output, edge_index_col, dim=-1, reduce='mean', dim_size=n_nodes)
        return final_grad, None, None


def nodemask2edgemask(mask: torch.Tensor, edge_index: torch.Tensor, placeholder=None) -> torch.Tensor:
    """
    util function without grad

    :param mask:
    :param edge_index:
    :param placeholder:
    :return:
    """
    single = mask.ndim == 1
    return mask[edge_index[0]] * mask[edge_index[1]] if single else mask[:, edge_index[0]] * mask[:, edge_index[1]]


class DirectedEdge2UndirectedEdge(torch.autograd.Function):
    @staticmethod
    def forward(ctx, mask, *args):
        assert mask.dtype == torch.float  # must be differentiable
        edge_index, = args
        direct_mask = edge_index[0] < edge_index[1]
        direct_edge_index = edge_index[:, direct_mask]
        _, undirected_mask = to_undirected(direct_edge_index, mask)
        ctx.save_for_backward(direct_mask)
        return undirected_mask.T

    @staticmethod
    def backward(ctx, grad_output):
        direct_mask, = ctx.saved_tensors
        grad_output = grad_output.T
        final_grad = grad_output[direct_mask] + grad_output[torch.logical_not(direct_mask)]
        return final_grad, None, None


def directedge2undirectedge(mask, edge_index):
    direct_mask = edge_index[0] < edge_index[1]
    direct_edge_index = edge_index[:, direct_mask]
    _, undirected_mask = to_undirected(direct_edge_index, mask)
    return undirected_mask.T
