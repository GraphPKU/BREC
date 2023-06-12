import torch
import numba
import numpy as np


@numba.njit(cache=True)
def numba_sample_node(weight: np.ndarray, k: int):
    mask = np.zeros(weight.shape, dtype=np.bool_)
    close_set = {-1}
    for row in range(weight.shape[1]):
        idx = np.argsort(weight[:, row])[::-1]
        count = 0
        while True:
            for i in idx:
                if len(close_set) < weight.shape[0] + 1:  # not covered the whole graph
                    if i in close_set:
                        pass
                    else:
                        close_set.add(i)
                        mask[i, row] = True
                        count += 1
                else:
                    if mask[i, row]:
                        pass
                    else:
                        mask[i, row] = True
                        count += 1
                if count >= k:
                    break
            if count >= k:
                break

    return mask


def sample_heuristic(weight: torch.Tensor, k: int):
    if k < 0:
        k += weight.shape[0]
        k = max(k, 1)  # in case only 1 node

    if k >= weight.shape[0]:
        return torch.ones_like(weight, dtype=torch.float, device=weight.device)

    mask = numba_sample_node(weight.cpu().numpy(), k)
    return torch.from_numpy(mask).to(torch.float).to(weight.device)


if __name__ == '__main__':
    weight = torch.tensor([[0.0471, 4.0962, -0.2081],
                           [-0.6776, 3.1570, 1.3461],
                           [-0.5243, 0.4003, 0.3225],
                           [3.5561, 0.6059, 0.0305],
                           [-0.0270, 0.0559, 0.5547],
                           [0.5782, -0.2840, -0.6519],
                           [2.0829, -1.4955, -0.2017],
                           [1.4557, 0.0058, -0.0113],
                           [-0.2255, -0.9465, 0.1351],
                           [0.0229, -0.2911, -0.3258],
                           [2.8296, 0.0436, 0.7478],
                           [-0.6239, 2.1137, 0.8284],
                           [-0.9461, -1.1008, 0.0775],
                           [-0.4388, -0.5300, 0.3143]])
    k = 10
    mask = sample_heuristic(weight, k)
