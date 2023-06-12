from typing import List, Optional, Union
import itertools

import torch
from torch.utils.data.dataloader import default_collate
from torch_geometric.loader.dataloader import Collater
from torch_geometric.data import Data, Dataset, HeteroData, Batch

from data.data_utils import SubgraphSetBatch, get_ptr


class SubgraphSetCollator:
    """
    Given subgraphs [[g1_1, g1_2, g1_3], [g2_1, g2_2, g2_3], ...]
    Collate them as a batch
    """

    def __init__(self,
                 follow_batch: Optional[List[str]] = None,
                 exclude_keys: Optional[List[str]] = None):
        assert follow_batch is None and exclude_keys is None, "Not supported"
        self.follow_batch = follow_batch
        self.exclude_keys = exclude_keys

    def __call__(self, batch_list: List[List[Data]]):
        list_subgraphs = list(itertools.chain.from_iterable(batch_list))

        num_graphs = len(batch_list)
        num_subgraphs = len(batch_list[0])
        batch = Batch.from_data_list(list_subgraphs, None, None)
        original_graph_mask = torch.repeat_interleave(torch.arange(num_graphs), num_subgraphs)
        ptr = get_ptr(original_graph_mask)

        return SubgraphSetBatch(x=batch.x,
                                edge_index=batch.edge_index,
                                edge_attr=batch.edge_attr,
                                edge_weight=None,
                                selected_node_masks=None,
                                y=batch.y[ptr[:-1]],
                                batch=batch.batch,
                                inter_graph_idx=original_graph_mask,
                                num_graphs=batch.num_graphs)


class MYDataLoader(torch.utils.data.DataLoader):
    """
    Customed data loader modified from data loader of PyG.
    https://pytorch-geometric.readthedocs.io/en/latest/_modules/torch_geometric/loader/dataloader.html#DataLoader
    Enable sampling subgraphs of the original graph, and merge the sampled graphs in a batch
    batch_size = batch_size * n_subgraphs

    :param dataset:
    :param batch_size:
    :param shuffle:
    :param n_subgraphs:
    :param subgraph_loader:
    :param follow_batch:
    :param exclude_keys:
    :param kwargs:
    """

    def __init__(
            self,
            dataset: Union[Dataset, List[Data], List[HeteroData]],
            batch_size: int = 1,
            shuffle: bool = False,
            subgraph_loader: bool = False,
            follow_batch: Optional[List[str]] = None,
            exclude_keys: Optional[List[str]] = None,
            **kwargs,
    ):
        # Save for PyTorch Lightning:
        assert follow_batch is None and exclude_keys is None
        self.follow_batch = follow_batch
        self.exclude_keys = exclude_keys

        collate_fn = SubgraphSetCollator if subgraph_loader else Collater

        super().__init__(
            dataset,
            batch_size,
            shuffle,
            collate_fn=collate_fn(follow_batch=follow_batch, exclude_keys=exclude_keys),
            **kwargs,
        )
