import torch
from torch_scatter import scatter

def generalized_scatter(src, index, dim_size, dim=0, aggregators=['add', 'max']):
    return torch.cat([scatter(src, index=index, dim=dim, dim_size=dim_size, reduce=agg) for agg in aggregators], -1)

class Aggregator(torch.nn.Module):
    def __init__(self, aggregators=['add', 'max'], input_dim=None, keep_same_dim=False):
        super().__init__()
        self.keep_same_dim = keep_same_dim
        self.aggregators = aggregators
        if keep_same_dim:
            self.linear = torch.nn.Linear(input_dim*len(aggregators), input_dim)
    def reset_parameters(self):
        if self.keep_same_dim:
            self.linear.reset_parameters()

    def forward(self, src, index, dim_size):
        out = generalized_scatter(src, index, dim_size, aggregators=self.aggregators)
        if self.keep_same_dim:
            out = self.linear(out)
        return out
