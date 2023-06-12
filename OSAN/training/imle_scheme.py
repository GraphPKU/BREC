from typing import Tuple

import torch

from subgraph.khop_subgraph import khop_subgraphs, khop_global, khop_global_dual
from subgraph.greedy_expanding_tree import greedy_expand_tree
from subgraph.mst_subgraph import mst_subgraph_sampling
from subgraph.or_optimal_subgraph import get_or_suboptim_subgraphs, get_or_optim_subgraphs
from subgraph.undirected_edge_selection import undirected_edge_sample
from subgraph.node_sample_heuristic import sample_heuristic


def get_split_idx(inc_tensor: torch.Tensor) -> Tuple:
    """
    Get splits from accumulative vector

    :param inc_tensor:
    :return:
    """
    return tuple((inc_tensor[1:] - inc_tensor[:-1]).detach().cpu().tolist())


class IMLEScheme:
    def __init__(self, imle_sample_policy, ptr, graphs, sample_k, return_list, perturb, sample_rand):
        self.imle_sample_policy = imle_sample_policy
        self.sample_k = sample_k
        self._ptr = ptr
        self._graphs = graphs
        self._return_list = return_list
        self._perturb = perturb
        self._sample_rand = sample_rand

    @property
    def ptr(self):
        return self._ptr

    @ptr.setter
    def ptr(self, value):
        self._ptr = value

    @ptr.deleter
    def ptr(self):
        del self._ptr

    @property
    def graphs(self):
        return self._graphs

    @graphs.setter
    def graphs(self, new_graphs):
        self._graphs = new_graphs

    @graphs.deleter
    def graphs(self):
        del self._graphs

    @property
    def perturb(self):
        return self._perturb

    @perturb.setter
    def perturb(self, value):
        self._perturb = value

    @property
    def return_list(self):
        return self._return_list

    @return_list.setter
    def return_list(self, value):
        self._return_list = value

    @property
    def sample_rand(self):
        return self._sample_rand

    @sample_rand.setter
    def sample_rand(self, value):
        self._sample_rand = value

    @torch.no_grad()
    def torch_sample_scheme(self, logits: torch.Tensor):
        aux_output = []

        local_logits = logits.detach() if not self.sample_rand else \
            torch.randn(logits.shape, device=logits.device, dtype=logits.dtype)
        local_logits = torch.split(local_logits, self.ptr, dim=0)

        sample_instance_idx = []
        for i, logit in enumerate(local_logits):
            if self.perturb:
                noise = torch.randn(logit.shape, device=logit.device) * logit.std(0, keepdims=True) * 0.1
                logit = logit + noise

            if self.imle_sample_policy == 'node':
                if self.sample_k < 0:
                    k = logit.shape[0] + self.sample_k
                    k = max(k, 1)  # in case only 1 node
                else:
                    k = min(self.sample_k, logit.shape[0])
                thresh = torch.topk(logit, k=k, dim=0, sorted=True).values[-1, :]  # kth largest
                mask = (logit >= thresh[None]).to(torch.float)
            elif self.imle_sample_policy == 'node_heuristic':
                mask = sample_heuristic(logit, self.sample_k)
            elif self.imle_sample_policy == 'node_ordered':
                if self.sample_k < 0:
                    k = logit.shape[0] + self.sample_k
                    k = max(k, 1)
                else:
                    k = self.sample_k

                mask = torch.zeros_like(logit, dtype=torch.float, device=logit.device)
                sorted_idx = torch.sort(logit, dim=0).indices

                r = sorted_idx[-k:, :].reshape(-1)
                c = torch.arange(logit.shape[1]).repeat(k)
                mask[r, c] = 1
                aux_output.append(sorted_idx[-k:, :])
            elif self.imle_sample_policy == 'edge':
                mask = undirected_edge_sample(self.graphs[i].edge_index, logit, self.sample_k)
            elif self.imle_sample_policy == 'edge_linegraph':
                if self.sample_k < 0:
                    k = logit.shape[0] + self.sample_k
                    k = max(k, 0)
                else:
                    k = min(self.sample_k, logit.shape[0])
                thresh = torch.topk(logit, k=k, dim=0, sorted=True).values[-1, :]  # kth largest
                mask = (logit >= thresh[None]).to(torch.float)
            elif self.imle_sample_policy == 'khop_subgraph':
                mask = khop_subgraphs(self.graphs[i], self.sample_k, instance_weight=logit)
            elif self.imle_sample_policy == 'khop_global':
                mask = khop_global(self.graphs[i], logit)
            elif self.imle_sample_policy == 'khop_global_dual':
                mask = khop_global_dual(self.graphs[i], logit)
            elif self.imle_sample_policy == 'mst':
                mask = mst_subgraph_sampling(self.graphs[i], logit)
            elif self.imle_sample_policy == 'greedy_exp':
                mask = greedy_expand_tree(self.graphs[i], logit, self.sample_k).T
            elif self.imle_sample_policy == 'or':
                if self.sample_k < 0:
                    k = logit.shape[0] + self.sample_k
                    k = max(k, 1)  # in case only 1 node
                else:
                    k = self.sample_k
                mask = get_or_suboptim_subgraphs(logit, k)
            elif self.imle_sample_policy == 'or_optim':
                mask = get_or_optim_subgraphs(self.graphs[i], logit, self.sample_k)
            else:
                raise NotImplementedError

            mask.requires_grad = False
            sample_instance_idx.append(mask)

        if not self.return_list:
            sample_instance_idx = torch.cat(sample_instance_idx, dim=0)
            sample_instance_idx.requires_grad = False

        return sample_instance_idx, aux_output if aux_output else None
