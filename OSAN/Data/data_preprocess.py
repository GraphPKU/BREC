from typing import List, Tuple
import torch
import numpy as np
import numba
from torch_geometric.data import Data, InMemoryDataset
from torch_geometric.utils import to_undirected, coalesce

from data.data_utils import edge_index2dense_adj
from subgraph.khop_subgraph import parallel_khop_neighbor


class GraphToUndirected:
    """
    Wrapper of to_undirected:
    https://pytorch-geometric.readthedocs.io/en/latest/modules/utils.html?highlight=undirected#torch_geometric.utils.to_undirected
    """

    def __call__(self, graph: Data):
        if graph.edge_attr is not None:
            edge_index, edge_attr = to_undirected(graph.edge_index, graph.edge_attr, graph.num_nodes)
        else:
            edge_index = to_undirected(graph.edge_index, graph.edge_attr, graph.num_nodes)
            edge_attr = None
        return Data(x=graph.x,
                    edge_index=edge_index,
                    edge_attr=edge_attr,
                    y=graph.y,
                    num_nodes=graph.num_nodes)


class GraphCoalesce:

    def __call__(self, graph: Data):
        if graph.edge_attr is None:
            edge_index = coalesce(graph.edge_index, None, num_nodes=graph.num_nodes)
            edge_attr = None
        else:
            edge_index, edge_attr = coalesce(graph.edge_index, graph.edge_attr, graph.num_nodes)
        return Data(x=graph.x,
                    edge_index=edge_index,
                    edge_attr=edge_attr,
                    y=graph.y,
                    num_nodes=graph.num_nodes)


class AugmentwithLineGraph:
    def __call__(self, g: Data):
        # TODO: consider no edge attr situation
        direct_mask = g.edge_index[0] < g.edge_index[1]
        edge_index = g.edge_index[:, direct_mask]
        edge_attr = g.edge_attr[direct_mask]
        new_node_attr, new_edge_attr, new_edge_index = graph2linegraph(g.x.numpy(),
                                                                       edge_attr.numpy(),
                                                                       edge_index.numpy().T)
        new_edge_attr = torch.from_numpy(np.vstack(new_edge_attr))
        new_node_attr = torch.from_numpy(new_node_attr)
        new_edge_index = torch.from_numpy(new_edge_index)

        new_edge_index, new_edge_attr = to_undirected(new_edge_index, new_edge_attr, new_node_attr.shape[0])

        g.lin_x = new_node_attr
        g.lin_edge_index = new_edge_index
        g.lin_edge_attr = new_edge_attr
        g.lin_num_nodes = new_node_attr.shape[0]

        return g


@numba.njit(cache=True)
def graph2linegraph(node_attr: np.ndarray, edge_attr: np.ndarray, edge_index: np.ndarray) \
        -> Tuple[np.ndarray, List[np.ndarray], np.ndarray]:
    """

    node_attr:
    edge_attr: expected directed, E x 2
    edge_index:
    """
    # TODO: discuss what the new node/edge feature could be
    new_node_attr = node_attr[edge_index[:, 0]] + node_attr[edge_index[:, 1]]
    # only for defining type
    new_edge_attr = [edge_attr[0]]
    new_edge_index = [[-1, -1]]

    # no self-loop
    for i, (n1, n2) in enumerate(edge_index[:-1]):
        for j in range(i + 1, edge_index.shape[0]):
            if n1 in edge_index[j] or n2 in edge_index[j]:
                new_edge_index.append([i, j])
                new_edge_attr.append(edge_attr[i] + edge_attr[j])

    # remove the first items
    return new_node_attr, new_edge_attr[1:], np.array(new_edge_index[1:]).T


def khop_data_process(dataset: InMemoryDataset, khop: int):
    """
    Add attribute of khop subgraph indices, so that during sampling they don't need to be re-calculcated.

    :param dataset:
    :param khop:
    :return:
    """
    max_node = 0
    for g in dataset:
        max_node = max(max_node, g.num_nodes)

    for i, g in zip(dataset._indices, dataset):
        mask = parallel_khop_neighbor(g.edge_index.cpu().numpy(), g.num_nodes, max_node, khop)
        dataset._data_list[i].khop_idx = torch.from_numpy(mask).to(torch.float32)
    return dataset


class AugmentwithKhopList:
    """
    a class for the khop_data_process function
    """
    def __init__(self, max_num_node: int, khop: int):
        self.max_num_node = max_num_node
        self.khop = khop

    def __call__(self, g: Data):
        mask = parallel_khop_neighbor(g.edge_index.numpy(), g.num_nodes, self.max_num_node, self.khop)
        g.khop_idx = torch.from_numpy(mask).to(torch.float32)
        return g


class AugmentwithAdj:

    def __init__(self, max_num_node: int):
        self.max_num_node = max_num_node

    def __call__(self, g: Data):
        adj = edge_index2dense_adj(g.edge_index,
                                   num_nodes=g.num_nodes,
                                   max_nodes=self.max_num_node,
                                   default_dtype=torch.float)
        g.adj = adj
        return g
