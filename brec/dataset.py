import networkx as nx
import numpy as np
import torch
import torch_geometric
from torch_geometric.data import (
    InMemoryDataset,
    download_url,
    extract_zip,
)
from torch_geometric.utils.convert import from_networkx
import os
from tqdm import tqdm
import shutil

torch_geometric.seed_everything(2022)


def graph6_to_pyg(x):
    return from_networkx(nx.from_graph6_bytes(x))


def add_x(data):
    data.x = torch.ones([data.num_nodes, 1])
    return data


class BRECDataset(InMemoryDataset):
    url = "https://www.dropbox.com/scl/fi/lzbll513dcgo9x9d3ozso/brec_v3.zip?rlkey=pyd7yss25wasajyffu47ierg3&dl=1"

    def __init__(
        self,
        name="no_param",
        root="Data",
        transform=None,
        pre_transform=None,
        pre_filter=None,
    ):
        """_summary_

        Args:
            name (str, optional): The name of processed datasets if with additional settings for data processing. Defaults to "no_param".
            root (str, optional): Data file directory. Defaults to "Data".
            transform (_type_, optional): Data transform function. Defaults to None.
            pre_transform (_type_, optional): Data pre-transform function. Defaults to None.
            pre_filter (_type_, optional): Data pre-filter function. Defaults to None.
        """
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

    def download(self):
        shutil.rmtree(self.raw_dir)
        path = download_url(self.url, self.root)
        extract_zip(path, self.raw_dir)
        os.unlink(path)

    def process(self):
        data_list = np.load(self.raw_paths[0], allow_pickle=True)
        data_list = [graph6_to_pyg(data) for data in data_list]
        data_list = [add_x(data) for data in data_list]

        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in tqdm(data_list)]

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])


def main():
    dataset = BRECDataset()
    print(len(dataset))


if __name__ == "__main__":
    main()
