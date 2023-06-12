from heapq import heappush, heappop

import numba
import numpy as np
import torch


@numba.njit(cache=True, locals={'edge_mask': numba.float32[::1]})
def numba_edge_selection(edge_index: np.ndarray, edge_weights: np.ndarray, k: int) -> np.ndarray:
    idx_dict = {(-1, -1): -1}
    weight_dict = {(-1, -1): 0.}

    h = [(1e10, (-1, -1))]

    for idx, (n1, n2) in enumerate(edge_index):
        idx_dict[(n1, n2)] = idx
        if n1 < n2:
            weight_dict[(n1, n2)] = edge_weights[idx]
        else:
            weight_dict[(n2, n1)] += edge_weights[idx]
            heappush(h, (-weight_dict[(n2, n1)], (n2, n1)))

    edge_mask = np.zeros(len(edge_weights), dtype=np.float32)

    for i in range(k):
        if not len(h):
            break

        _, (n1, n2) = heappop(h)
        if n1 != -1:
            edge_mask[idx_dict[(n1, n2)]] = 1.
            edge_mask[idx_dict[(n2, n1)]] = 1.

    return edge_mask


def undirected_edge_sample(edge_index: torch.Tensor, edge_weights: torch.Tensor, k: int):
    """

    :param edge_index:
    :param edge_weights: (N, k)
    :param k:
    :return:
    """
    device = edge_index.device
    if k < 0:
        k += edge_weights.shape[0] // 2
        k = max(k, 0)

    if k * 2 >= edge_index.shape[1]:
        return torch.ones_like(edge_weights, dtype=edge_weights.dtype, device=device)

    edge_index = edge_index.cpu().numpy().T
    edge_weights = edge_weights.cpu().numpy()

    masks = []

    for i in range(edge_weights.shape[1]):
        masks.append(numba_edge_selection(edge_index, edge_weights[:, i], k))

    masks = np.vstack(masks).T
    return torch.from_numpy(masks).to(device)
