import numba
import numpy as np
import torch
from torch import Tensor
from torch_geometric.data import Data


@numba.njit(cache=True, locals={'parts': numba.int32[::1], 'edge_selected': numba.bool_[::1]})
def numba_kruskal(sort_index: np.ndarray, edge_index_list: np.ndarray, num_nodes: int) -> np.ndarray:
    parts = np.full(num_nodes, -1, dtype=np.int32)  # -1: unvisited
    edge_selected = np.zeros_like(sort_index, dtype=np.bool_)
    edge_selected[sort_index[0]] = True
    n1, n2 = edge_index_list[sort_index[0]]
    parts[n1] = 0
    parts[n2] = 0

    edge_selected_set = {(n1, n2,)}

    parts_hash = 1

    for idx in sort_index[1:]:
        n1, n2 = edge_index_list[idx]
        if (n2, n1) in edge_selected_set:
            edge_selected[idx] = True
            continue

        if parts[n1] == -1 and parts[n2] == -1:
            parts[n1] = parts_hash
            parts[n2] = parts_hash
            parts_hash += 1
            edge_selected[idx] = True
            edge_selected_set.add((n1, n2,))
        elif parts[n1] != -1 and parts[n2] == -1:
            parts[n2] = parts[n1]
            edge_selected[idx] = True
            edge_selected_set.add((n1, n2,))
        elif parts[n2] != -1 and parts[n1] == -1:
            parts[n1] = parts[n2]
            edge_selected[idx] = True
            edge_selected_set.add((n1, n2,))
        elif parts[n1] != -1 and parts[n2] != -1 and parts[n1] != parts[n2]:
            parts[parts == parts[n2]] = parts[n1]
            edge_selected[idx] = True
            edge_selected_set.add((n1, n2,))

    return edge_selected


def mst_subgraph_sampling(graph: Data, edge_weight: Tensor) -> Tensor:
    """

    :param graph:
    :param edge_weight: shape (E, n_subgraphs)
    :return:
    """
    device = graph.edge_index.device

    if not edge_weight.shape[0]:
        return torch.ones_like(edge_weight, device=device, dtype=torch.float)

    edge_masks = []
    n_subgraphs = edge_weight.shape[1]
    sort_idx = torch.argsort(edge_weight, dim=0, descending=True).cpu().numpy()

    np_edge_index = graph.edge_index.cpu().numpy().T
    for i in range(n_subgraphs):
        edge_mask = numba_kruskal(sort_idx[:, i], np_edge_index, graph.num_nodes)
        edge_mask = torch.from_numpy(edge_mask).to(device)
        edge_masks.append(edge_mask)

    return torch.vstack(edge_masks).to(torch.float32).T
