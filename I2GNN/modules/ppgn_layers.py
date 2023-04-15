import torch
import pdb


def diag_offdiag_maxpool(input):
    N = input.shape[-1]

    max_diag = torch.max(torch.diagonal(input, dim1=-2, dim2=-1), dim=2)[0]  # BxS

    # with torch.no_grad():
    max_val = torch.max(max_diag)
    min_val = torch.max(-1 * input)
    val = torch.abs(torch.add(max_val, min_val))

    min_mat = torch.mul(val, torch.eye(N, device=input.device)).view(1, 1, N, N)

    max_offdiag = torch.max(torch.max(input - min_mat, dim=3)[0], dim=2)[0]  # BxS

    return torch.cat((max_diag, max_offdiag), dim=1)  # output Bx2S


def diag_offdiag_meanpool(input, level='graph'):
    N = input.shape[-1]
    if level == 'graph':
        mean_diag = torch.mean(torch.diagonal(input, dim1=-2, dim2=-1), dim=2)  # BxS
        mean_offdiag = (torch.sum(input, dim=[-1, -2]) - mean_diag * N) / (N * N - N)
    else:
        mean_diag = torch.diagonal(input, dim1=-2, dim2=-1)
        mean_offdiag = (torch.sum(input, dim=-1) + torch.sum(input, dim=-2) - 2 * mean_diag) # / (2 * N - 2)
    return torch.cat((mean_diag, mean_offdiag), dim=1)  # output Bx2S
