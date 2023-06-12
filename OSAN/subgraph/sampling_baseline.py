from typing import List, Tuple
import random

import numpy as np
import torch
from torch import Tensor, LongTensor
from torch_geometric.data import Data
from torch_geometric.utils import is_undirected, to_undirected

from subgraph.construct import nodesubset_to_subgraph
from subgraph.mst_subgraph import numba_kruskal
from subgraph.greedy_expanding_tree import numba_greedy_expand_tree
from subgraph.khop_subgraph import numba_k_hop_subgraph


def node_rand_sampling(graph: Data,
                       n_subgraphs: int,
                       node_per_subgraph: int = -1,
                       relabel: bool = False,
                       add_full_graph: bool = False) -> List[Data]:
    """
    Sample subgraphs.

    :param graph:
    :param n_subgraphs:
    :param node_per_subgraph:
    :param relabel:
    :param add_full_graph:
    :return:
        A list of graphs and their index masks
    """
    n_nodes = graph.num_nodes

    if node_per_subgraph < 0:  # drop nodes
        node_per_subgraph += n_nodes
        node_per_subgraph = max(1, node_per_subgraph)

    # TODO: may need to clone the graph if some properties are important

    graphs = [Data(x=graph.x,
                   edge_index=graph.edge_index,
                   edge_attr=graph.edge_attr,
                   num_nodes=graph.num_nodes,
                   y=graph.y)] if add_full_graph else []
    for i in range(n_subgraphs):
        indices = torch.randperm(n_nodes)[:node_per_subgraph]
        sort_indices = torch.sort(indices).values
        graphs.append(nodesubset_to_subgraph(graph, sort_indices, relabel=relabel))

    return graphs


def edge_rand_sampling(graph: Data,
                       n_subgraphs: int,
                       edge_per_subgraph: int = -1,
                       add_full_graph: bool = False) -> List[Data]:
    """

    :param graph:
    :param n_subgraphs:
    :param edge_per_subgraph:
    :param add_full_graph:
    :return:
    """
    edge_index, edge_attr, undirected = edge_sample_preproc(graph)
    n_edge = edge_index.shape[1]
    if edge_per_subgraph < 0:  # drop edges
        edge_per_subgraph += n_edge
        edge_per_subgraph = max(edge_per_subgraph, 0)

    if n_edge == 0 or edge_per_subgraph >= n_edge:
        return [Data(x=graph.x,
                     edge_index=graph.edge_index,
                     edge_attr=graph.edge_attr,
                     num_nodes=graph.num_nodes,
                     y=graph.y)] * (n_subgraphs + int(add_full_graph))

    graphs = [Data(x=graph.x,
                   edge_index=graph.edge_index,
                   edge_attr=graph.edge_attr,
                   num_nodes=graph.num_nodes,
                   y=graph.y)] if add_full_graph else []
    for i in range(n_subgraphs):
        indices = torch.randperm(n_edge)[:edge_per_subgraph]
        sort_indices = torch.sort(indices).values

        subgraph_edge_index = edge_index[:, sort_indices]
        subgraph_edge_attr = edge_attr[sort_indices, :] if edge_attr is not None else None

        if undirected:
            if subgraph_edge_attr is not None:
                subgraph_edge_index, subgraph_edge_attr = to_undirected(subgraph_edge_index, subgraph_edge_attr,
                                                                        num_nodes=graph.num_nodes)
            else:
                subgraph_edge_index = to_undirected(subgraph_edge_index, subgraph_edge_attr,
                                                    num_nodes=graph.num_nodes)

        graphs.append(Data(
            x=graph.x,
            edge_index=subgraph_edge_index,
            edge_attr=subgraph_edge_attr,
            num_nodes=graph.num_nodes,
            y=graph.y,
        ))

    return graphs


def edge_sample_preproc(data: Data) -> Tuple[LongTensor, Tensor, bool]:
    """
    If undirected, return the non-duplicate directed edges for sampling

    :param data:
    :return:
    """
    if data.edge_attr is not None and data.edge_attr.ndim == 1:
        edge_attr = data.edge_attr.unsqueeze(-1)
    else:
        edge_attr = data.edge_attr

    undirected = is_undirected(data.edge_index, edge_attr, data.num_nodes)

    if undirected:
        keep_edge = data.edge_index[0] <= data.edge_index[1]
        edge_index = data.edge_index[:, keep_edge]
        edge_attr = edge_attr[keep_edge, :] if edge_attr is not None else edge_attr
    else:
        edge_index = data.edge_index

    return edge_index, edge_attr, undirected


def khop_subgraph_sampling(data: Data,
                           n_subgraphs: int,
                           khop: int = 3,
                           relabel: bool = False,
                           add_full_graph: bool = False) -> List[Data]:
    """
    Sample the k-hop-neighbors subgraphs randomly

    :param data:
    :param n_subgraphs:
    :param khop:
    :param relabel:
    :param add_full_graph:
    :return:
    """
    sample_indices = random.choices(range(data.num_nodes), k=n_subgraphs)
    graphs = [Data(x=data.x,
                   edge_index=data.edge_index,
                   edge_attr=data.edge_attr,
                   num_nodes=data.num_nodes,
                   y=data.y)] if add_full_graph else []

    np_edge_index = data.edge_index.cpu().numpy()
    for idx in sample_indices:
        node_idx, edge_index, edge_mask = numba_k_hop_subgraph(np_edge_index,
                                                               idx,
                                                               khop,
                                                               data.num_nodes,
                                                               relabel)

        num_nodes = data.num_nodes
        if relabel:
            num_nodes = len(node_idx)

        graphs.append(Data(x=data.x if not relabel else data.x[node_idx],
                           edge_index=torch.from_numpy(edge_index).to(data.edge_index.device),
                           edge_attr=data.edge_attr[edge_mask] if data.edge_attr is not None else None,
                           num_nodes=num_nodes,
                           y=data.y))
    return graphs


def khop_subgraph_sampling_fromcache(data: Data,
                                     n_subgraphs: int,
                                     relabel: bool = False,
                                     add_full_graph: bool = False) -> List[Data]:
    """
    :param data:
    :param n_subgraphs:
    :param relabel:
    :param add_full_graph:
    :return:
    """
    khop_idx = data.khop_idx.numpy()
    sample_indices = random.choices(range(data.num_nodes), k=n_subgraphs)
    graphs = [Data(x=data.x,
                   edge_index=data.edge_index,
                   edge_attr=data.edge_attr,
                   num_nodes=data.num_nodes,
                   y=data.y)] if add_full_graph else []

    for idx in sample_indices:
        node_idx = np.where(khop_idx[idx])[0]
        graphs.append(nodesubset_to_subgraph(data, torch.from_numpy(node_idx), relabel))
    return graphs


def khop_subgraph_sampling_dual(data: Data,
                                n_subgraphs: int,
                                khop: int = 3,
                                relabel: bool = False,
                                add_full_graph: bool = False) -> List[Data]:
    """
    delete khop subgraph of a seed node

    :param data:
    :param n_subgraphs:
    :param khop:
    :param relabel:
    :param add_full_graph:
    :return:
    """
    sample_indices = random.choices(range(data.num_nodes), k=n_subgraphs)
    graphs = [Data(x=data.x,
                   edge_index=data.edge_index,
                   edge_attr=data.edge_attr,
                   num_nodes=data.num_nodes,
                   y=data.y)] if add_full_graph else []

    np_edge_index = data.edge_index.cpu().numpy()
    for idx in sample_indices:
        node_idx, _, _ = numba_k_hop_subgraph(np_edge_index,
                                              idx,
                                              khop,
                                              data.num_nodes,
                                              relabel=False)

        if len(node_idx) == data.num_nodes:  # never delete all nodes
            node_idx = np.random.randint(0, data.num_nodes, 1)  # pick a random one
        else:
            node_idx = np.setdiff1d(np.arange(data.num_nodes), node_idx, assume_unique=True)

        graphs.append(nodesubset_to_subgraph(data, torch.from_numpy(node_idx), relabel))
    return graphs


def khop_subgraph_sampling_dual_fromcache(data: Data,
                                          n_subgraphs: int,
                                          relabel: bool = False,
                                          add_full_graph: bool = False) -> List[Data]:
    """
    for data object already WITH khop neighbor information for EACH node

    @param data:
    @param n_subgraphs:
    @param relabel:
    @param add_full_graph:
    @return:
    """
    khop_idx = data.khop_idx.numpy()
    sample_indices = random.choices(range(data.num_nodes), k=n_subgraphs)
    graphs = [Data(x=data.x,
                   edge_index=data.edge_index,
                   edge_attr=data.edge_attr,
                   num_nodes=data.num_nodes,
                   y=data.y)] if add_full_graph else []

    for idx in sample_indices:
        node_idx = np.where(khop_idx[idx])[0]

        if len(node_idx) == data.num_nodes:  # never delete all nodes
            node_idx = np.random.randint(0, data.num_nodes, 1)  # pick a random one
        else:
            node_idx = np.setdiff1d(np.arange(data.num_nodes), node_idx, assume_unique=True)

        graphs.append(nodesubset_to_subgraph(data, torch.from_numpy(node_idx), relabel))
    return graphs


def max_spanning_tree_subgraph(data: Data, n_subgraphs: int, add_full_graph: bool = False) -> List[Data]:
    """

    :param data:
    :param n_subgraphs:
    :param add_full_graph:
    :return:
    """
    graphs = [Data(x=data.x,
                   edge_index=data.edge_index,
                   edge_attr=data.edge_attr,
                   num_nodes=data.num_nodes,
                   y=data.y)] if add_full_graph else []

    if not data.num_edges:
        graphs += [Data(x=data.x,
                   edge_index=data.edge_index,
                   edge_attr=data.edge_attr,
                   num_nodes=data.num_nodes,
                   y=data.y)] * n_subgraphs
        return graphs

    np_edge_index = data.edge_index.cpu().numpy().T
    for i in range(n_subgraphs):
        sort_index = np.random.permutation(np_edge_index.shape[0])
        edge_mask = numba_kruskal(sort_index, np_edge_index, data.num_nodes)
        edge_mask = torch.from_numpy(edge_mask)
        graphs.append(Data(x=data.x,
                           edge_index=data.edge_index[:, edge_mask],
                           edge_attr=data.edge_attr[edge_mask] if data.edge_attr is not None else None,
                           num_nodes=data.num_nodes,
                           y=data.y))
    return graphs


def greedy_expand_sampling(graph: Data,
                           n_subgraphs: int,
                           node_per_subgraph: int = -1,
                           relabel: bool = False,
                           add_full_graph: bool = False) -> List[Data]:
    """

    :param graph:
    :param n_subgraphs:
    :param node_per_subgraph:
    :param relabel:
    :param add_full_graph:
    :return:
    """
    if node_per_subgraph >= graph.num_nodes:
        return [Data(x=graph.x,
                     edge_index=graph.edge_index,
                     edge_attr=graph.edge_attr,
                     num_nodes=graph.num_nodes,
                     y=graph.y)] * n_subgraphs

    edge_index = graph.edge_index.cpu().numpy()
    graphs = [Data(x=graph.x,
                   edge_index=graph.edge_index,
                   edge_attr=graph.edge_attr,
                   num_nodes=graph.num_nodes,
                   y=graph.y)] if add_full_graph else []
    for _ in range(n_subgraphs):
        node_mask = numba_greedy_expand_tree(edge_index, node_per_subgraph, None, graph.num_nodes, repeat=1)
        graphs.append(nodesubset_to_subgraph(graph, torch.from_numpy(node_mask), relabel=relabel))

    return graphs
