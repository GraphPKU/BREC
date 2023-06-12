from typing import Optional, Tuple

import torch
from torch import Tensor
from torch_geometric.data import Data
import numpy as np
import numba

from data.data_utils import edgeindex2neighbordict


@numba.njit(cache=True, locals={'edge_mask': numba.bool_[::1], 'np_node_idx': numba.int64[::1]})
def numba_k_hop_subgraph(edge_index: np.ndarray, seed_node: int, khop: int, num_nodes: int, relabel: bool) \
        -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    k_hop_subgraph of PyG is too slow

    :param edge_index:
    :param seed_node:
    :param khop:
    :param num_nodes:
    :param relabel:
    :return:
    """
    node_idx = {seed_node}
    visited_nodes = {seed_node}

    neighbor_dict = edgeindex2neighbordict(edge_index, num_nodes)

    cur_neighbors = {seed_node}
    for hop in range(khop):
        last_neighbors, cur_neighbors = cur_neighbors, set()

        for node in last_neighbors:
            for neighbor in neighbor_dict[node]:
                if neighbor not in visited_nodes:
                    cur_neighbors.add(neighbor)
                    visited_nodes.add(neighbor)
        node_idx.update(cur_neighbors)

    edge_mask = np.zeros(edge_index.shape[1], dtype=np.bool_)
    for i in range(edge_index.shape[1]):
        if edge_index[0][i] in node_idx and edge_index[1][i] in node_idx:
            edge_mask[i] = True

    edge_index = edge_index[:, edge_mask]
    node_idx = sorted(list(node_idx))

    if relabel:
        new_idx_dict = dict()
        for i, idx in enumerate(node_idx):
            new_idx_dict[idx] = i

        for i in range(2):
            for j in range(edge_index.shape[1]):
                edge_index[i, j] = new_idx_dict[edge_index[i, j]]

    np_node_idx = np.array(node_idx, dtype=np.int64)
    return np_node_idx, edge_index, edge_mask


@numba.njit(cache=True, parallel=False)
def best_khop_neighbor(edge_index: np.ndarray,
                       instance_weight: np.array,
                       num_nodes: int,
                       khop: int,
                       relabel: bool,
                       dual: bool):
    """
    If not dual, select the khop subgraph with the highest sum of weights.
    else select the subgraph with min sum of weights

    :param edge_index:
    :param instance_weight:
    :param num_nodes:
    :param khop:
    :param relabel:
    :param dual:
    :return:
    """
    nodes = np.zeros(0, dtype=np.int64)
    best_ref = -1.e10 if not dual else 1.e10

    for i in range(num_nodes):
        node_idx, _, _ = numba_k_hop_subgraph(edge_index, numba.int64(i), khop, num_nodes, relabel)
        if dual:
            ref = instance_weight[node_idx].mean()
            if ref < best_ref:
                best_ref = ref
                nodes = node_idx
        else:
            ref = instance_weight[node_idx].sum()
            if ref > best_ref:
                best_ref = ref
                nodes = node_idx

    return nodes


@numba.njit(cache=True, parallel=False)
def parallel_khop_neighbor(edge_index: np.ndarray,
                           num_nodes: int,
                           max_nodes: int,
                           khop: int):
    """

    :param edge_index:
    :param num_nodes:
    :param max_nodes:
    :param khop:
    :return:
    """
    masks = np.zeros((num_nodes, max_nodes), dtype=np.bool_)

    for i in range(num_nodes):
        node_idx, _, _ = numba_k_hop_subgraph(edge_index, numba.int64(i), khop, num_nodes, False)
        for n in node_idx:
            masks[i, n] = True
    return masks


def khop_subgraphs(graph: Data,
                   khop: int = 3,
                   instance_weight: Optional[Tensor] = None) -> Tensor:
    """
    Code for IMLE scheme, not sample on the fly
    For each seed node, get the k-hop neighbors first, then prune the graph as e.g. max spanning tree

    If not pruning, return the node masks, else edge masks

    :param graph:
    :param khop:
    :param instance_weight: if not pruning, this should be node weight, so that the seed node is picked according to
    the highest node weight. if pruning with MST algorithm, this should be edge weight, and node weight is the scatter
     of the incident edge weights
    :return: return node mask if not pruned, else edge mask
    """
    sampled_masks = torch.zeros_like(instance_weight, dtype=torch.float32, device=instance_weight.device)
    indices = torch.argmax(instance_weight, dim=0).cpu().tolist()

    edge_index = graph.edge_index.cpu().numpy()

    for i, idx in enumerate(indices):
        _node_idx, _edge_index, edge_mask = numba_k_hop_subgraph(edge_index, idx, khop, graph.num_nodes, False)
        sampled_masks[_node_idx, i] = 1.0

    return sampled_masks


def khop_global(graph: Data, weights: Optional[Tensor] = None) -> Tensor:
    """

    :param graph:
    :param weights: if not pruning, this should be node weight, so that the seed node is picked according to
    the highest node weight. if pruning with MST algorithm, this should be edge weight, and node weight is the scatter
     of the incident edge weights
    :return: return node mask if not pruned, else edge mask
    """
    khop_subgraph_idx = graph.khop_idx[:, :graph.num_nodes]  # C x N
    khop_subgraph_idx_dup = khop_subgraph_idx[..., None].repeat(1, 1, weights.shape[1])    # C x N x k

    mean_of_weights = (weights[None] * khop_subgraph_idx_dup).sum(1) / khop_subgraph_idx_dup.sum(1)
    idx = torch.argmax(mean_of_weights, dim=0)

    mask = khop_subgraph_idx[idx, :].T
    return mask


def khop_global_dual(graph: Data, weights: Optional[Tensor]) -> Tensor:
    """
    Instead of sampling khop neighbors of a seed node, we delete khop nodes of a seed node.

    :param graph:
    :param weights:
    :return:
    """
    khop_subgraph_idx = graph.khop_idx[:, :graph.num_nodes]
    khop_subgraph_idx_dup = khop_subgraph_idx[..., None].repeat(1, 1, weights.shape[1])
    max_idx = torch.argmax(weights, dim=0)

    mean_of_weights = (weights[None] * khop_subgraph_idx_dup).sum(1) / khop_subgraph_idx_dup.sum(1)
    idx = torch.argmin(mean_of_weights, dim=0)
    mask = 1 - khop_subgraph_idx[idx, :].T
    mask[max_idx, :] = 1.0  # never delete all the nodes

    return mask
