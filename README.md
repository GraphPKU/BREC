# Towards Better Evaluation of GNN Expressiveness with BREC Dataset

## About

This package is official implementation of the following paper: [Towards Better Evaluation of GNN Expressiveness with BREC Dataset](https://arxiv.org/abs/2304.07702). Evalution process can be easily implemented by this package. For more detailed and advanced usage, please refer to [BREC](https://github.com/GraphPKU/BREC)

**BREC**  is a new dataset for GNN expressiveness comparison.
It addresses the limitations of previous datasets, including difficulty, granularity, and scale, by incorporating
400 pairs of various graphs in four categories (Basic, Regular, Extension, CFI).
The graphs are organized pair-wise, where each pair is tested individually to return whether a GNN can distinguish them. We propose a new evaluation method, **RPC** (Reliable Paired Comparisons), with a contrastive training framework.

## Usages

### Install

Install [pytorch](https://pytorch.org/) and [pytorch_geometric](https://github.com/pyg-team/pytorch_geometric) with corresponding versions aligning with your device. Then `pip install brec`.

### Example

Here is a simple example:

```python
import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv

from brec.dataset import BRECDataset
from brec.evaluator import evaluate


class GCN(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = GCNConv(1, 16)
        self.conv2 = GCNConv(16, 16)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)

        return x

    def reset_parameters(self):
        self.conv1.reset_parameters()
        self.conv2.reset_parameters()


model = GCN()

dataset = BRECDataset()
evaluate(
    dataset, model, device=torch.device("cpu"), log_path="log.txt", training_config=None
)


```
