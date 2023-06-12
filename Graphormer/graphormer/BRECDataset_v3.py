import networkx as nx
import numpy as np
import torch
import torch_geometric
from torch_geometric.data import InMemoryDataset
from torch_geometric.utils.convert import from_networkx
import os
from tqdm import tqdm
import pyximport

pyximport.install(setup_args={"include_dirs": np.get_include()})

import algos


def convert_to_single_emb(x, offset=512):
    feature_num = x.size(1) if len(x.size()) > 1 else 1
    feature_offset = 1 + torch.arange(0, feature_num * offset, offset, dtype=torch.long)
    x = x + feature_offset
    return x


torch_geometric.seed_everything(2022)


def graph6_to_pyg(x):
    return from_networkx(nx.from_graph6_bytes(x))


def add_xy(data):
    data.x = torch.ones([data.num_nodes, 1], dtype=torch.long)
    data.y = torch.tensor(1)
    return data


def preprocess_item(item):
    edge_attr, edge_index, x = item.edge_attr, item.edge_index, item.x
    N = x.size(0)
    x = convert_to_single_emb(x)

    # node adj matrix [N, N] bool
    adj = torch.zeros([N, N], dtype=torch.bool)
    adj[edge_index[0, :], edge_index[1, :]] = True

    # edge feature here
    # if len(edge_attr.size()) == 1:
    #     edge_attr = edge_attr[:, None]
    # attn_edge_type = torch.zeros([N, N, edge_attr.size(-1)], dtype=torch.long)
    # attn_edge_type[edge_index[0, :], edge_index[1, :]] = (
    #     convert_to_single_emb(edge_attr) + 1
    # )
    attn_edge_type = torch.zeros([N, N, 1], dtype=torch.long)
    attn_edge_type[edge_index[0, :], edge_index[1, :]] = 1

    shortest_path_result, path = algos.floyd_warshall(adj.numpy())
    max_dist = np.amax(shortest_path_result)
    edge_input = algos.gen_edge_input(max_dist, path, attn_edge_type.numpy())
    spatial_pos = torch.from_numpy((shortest_path_result)).long()
    # print(spatial_pos)
    attn_bias = torch.zeros([N + 1, N + 1], dtype=torch.float)  # with graph token

    # combine
    item.x = x
    item.adj = adj
    item.attn_bias = attn_bias
    item.attn_edge_type = attn_edge_type
    item.spatial_pos = spatial_pos
    item.in_degree = adj.long().sum(dim=1).view(-1)
    item.out_degree = adj.long().sum(dim=0).view(-1)
    item.edge_input = torch.from_numpy(edge_input).long()

    return item


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
        # self.data = torch.load(self.processed_paths[0])

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

    def process(self):
        data_list = np.load(self.raw_paths[0], allow_pickle=True)
        data_list = [graph6_to_pyg(data) for data in data_list]
        data_list = [add_xy(data) for data in data_list]
        # data_list = [preprocess_item(data) for data in tqdm(data_list)]

        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in tqdm(data_list)]

        # torch.save(data_list, self.processed_paths[0])

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])
        # def __getitem__(self, idx):
        #     data = self.data[idx]
        #     print(data)
        #     print(idx)
        #     print(data.idx)
        #     data.idx = idx
        #     return data
    def __getitem__(self, idx):
        # print(idx)
        if isinstance(idx, int):
            # print(idx)
            item = self.get(self.indices()[idx])
            # item = self.data[idx]
            item.idx = idx

            # print('no')
            # print(type(item))
            # # print(item)
            # print(type(preprocess_item(item)))
            # print('yes')
            # return item
            return preprocess_item(item)
        else:
            # return self.data[idx]
            return self.index_select(idx)


def main():
    dataset = BRECDataset()
    print(len(dataset))


if __name__ == "__main__":
    main()
