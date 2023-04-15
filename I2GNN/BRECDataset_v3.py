import networkx as nx
import numpy as np
import torch
import torch_geometric
from torch_geometric.data import InMemoryDataset
from torch_geometric.utils.convert import from_networkx
from torch_geometric.loader import DataLoader
import os
from tqdm import tqdm
import copy

torch_geometric.seed_everything(2022)


def graph6_to_pyg(x):
    return from_networkx(nx.from_graph6_bytes(x))


class BRECDataset(InMemoryDataset):
    def __init__(
        self,
        name="no_param",
        root="Data",
        transform=None,
        pre_transform=None,
        pre_filter=None,
    ):
        self.root = root
        self.name = name
        super().__init__(root, transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def processed_dir(self):
        name = "processed"
        return os.path.join(self.root, self.name, name)

    @property
    def raw_file_names(self):
        return ["brec_v3.npy"]

    @property
    def processed_file_names(self):
        return ["brec_v3.pt"]

    # def __getitem__(self, idx):
    #     print(idx)
    #     if isinstance(idx, (int, np.integer)):
    #         data_tuple = (self.get(self.indices()[idx * 2]), self.get(self.indices()[idx * 2 + 1]))
    #         data_tuple = (
    #             data_tuple
    #             if self.transform is None
    #             else (self.transform(data_tuple[0]), self.transform(data_tuple[1]))
    #         )
    #         return data_tuple
    #     else:
    #         indices = self.indices()
    #         if isinstance(idx, slice):
    #             indices = indices[idx]
    #         dataset = copy.copy(self)
    #         dataset._indices = indices
    #         return dataset

    def process(self):

        data_list = np.load(self.raw_paths[0], allow_pickle=True)
        data_list = [graph6_to_pyg(data) for data in data_list]

        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in tqdm(data_list)]

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])


def main():
    dataset = BRECDataset()


if __name__ == "__main__":
    main()
