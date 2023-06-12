import torch


def residual(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    if x.shape == y.shape:
        return (x + y) / 2 ** 0.5
    else:
        return y
