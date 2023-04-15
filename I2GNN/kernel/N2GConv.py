import torch.nn
import torch.nn.functional as F
from torch.nn import Linear, Conv1d
from torch_geometric.nn import (
    global_sort_pool, global_add_pool, global_mean_pool, global_max_pool
)
import torch
from torch.nn import ModuleList, ReLU, Sequential
from torch_geometric.nn.dense.linear import Linear



class N2GConv(torch.nn.Module):
    def __init__(self, in_dim, out_dim, base_gnn, use_rd=False, subgraph2_pooling='mean', subgraph_pooling='mean',
                 concat_pooling=False, use_pooling_nn=False, gate=False):
        super(N2GConv, self).__init__()
        ### base gnn
        self.base_gnn = base_gnn

        ### distance embedding
        z_emb_dim = in_dim
        self.z_embedding = torch.nn.Embedding(100, z_emb_dim)
        self.use_rd = use_rd
        if use_rd:
            self.rd_projection = torch.nn.Linear(2, z_emb_dim)

        ### pooling function
        self.concat_pooling = concat_pooling
        if concat_pooling:
            self.node_embedding = create_pooling_func([subgraph2_pooling, subgraph_pooling], use_pooling_nn, out_dim)
            s2_dim = 2 if subgraph2_pooling == 'mean-center' else 3 if subgraph2_pooling == 'mean-center-side' else 1
            s2_dim = s2_dim + 1 if subgraph_pooling == 'mean-context' else s2_dim
            self.concat_pooling_nn = Sequential(Linear(out_dim * (1 + s2_dim), 128), ReLU(), Linear(128, out_dim))

        ### activation layer
        self.act = F.elu


    def forward(self, x, data):

        ### distance embedding
        z_emb = self.z_embedding(data.z)
        if z_emb.ndim == 3:
            z_emb = z_emb.sum(dim=1)
        if self.use_rd:
            rd = self.rd_projection(data.rd)
            z_emb = torch.cat([z_emb, rd], dim=-1)
        x = torch.cat([x, z_emb], dim=-1)


        ### graph convolution
        x = self.base_gnn(x, data.edge_index, data.edge_attr)

        ### concat node embedding across different subgraphs
        if self.concat_pooling:
            x = torch.cat(
                [x, self.node_embedding(x, data)[data.node_to_original_node]], dim=-1)
            x = self.concat_pooling_nn(x)

        return x


class create_pooling_func(torch.nn.Module):
    def __init__(self, pool_methods, use_pooling_nn=False, h_dim=None):
        super(create_pooling_func, self).__init__()
        ## pool_methods = [subgraph2_pooling_method, subgraph_pooling_method]
        assert len(pool_methods) == 2
        self.pool_methods = pool_methods
        ## subgraph2_pooling method can be:
        # mean pooling
        if pool_methods[0] == 'mean':
            self.subgraph2_pool = mean_pool
        # center pooling: return the center node embedding
        elif pool_methods[0] == 'center':
            self.subgraph2_pool = center_pool
        # mean-center-side pooling: return the direct sum of mean pooling, center node and side node embedding
        elif pool_methods[0] == 'mean-center-side':
            self.subgraph2_pool = mean_center_side_pool
        else:
            raise Exception('Unknown subgraph2 pooling method!')

        # apply pooling neural networks
        self.use_pooling_nn = use_pooling_nn
        s2_dim = 2*h_dim if pool_methods[0] == 'mean-center' else 3*h_dim if pool_methods[0] == 'mean-center-side' else h_dim
        s1_dim = s2_dim + h_dim if pool_methods[1] == 'mean-context' else s2_dim
        if use_pooling_nn:
            self.edge_pooling_nn = Sequential(Linear(s2_dim, s2_dim), ReLU(),
                                              Linear(s2_dim, s2_dim))
            self.node_pooling_nn = Sequential(Linear(s1_dim, s1_dim), ReLU(),
                                              Linear(s1_dim, s1_dim))
        else:
            self.edge_pooling_nn = torch.nn.Identity()
            self.node_pooling_nn = torch.nn.Identity()


    def forward(self, x, data):
        # use context pooling
        if self.pool_methods[1] == 'mean-context':
            x_node = torch.nn.Identity()(x)
            x_node = global_mean_pool(x_node, data.node_to_original_node)

        # subgraph2 pooling
        x = self.subgraph2_pool(x, data.node_to_subgraph2, data)
        x = self.edge_pooling_nn(x)

        # subgraph pooling
        x = global_mean_pool(x, data.subgraph2_to_subgraph)
        if self.pool_methods[1] == 'mean-context':
            x = torch.cat([x, x_node], dim=-1)
        x = self.node_pooling_nn(x)
        return x

def mean_pool(x, index, data):
    return global_mean_pool(x, index)

def center_pool(x, index, data):
    return x[data.center_idx[:, 0]]

def side_pool(x, index, data):
    return x[data.center_idx[:, 1]]

def mean_center_side_pool(x, index, data):
    return torch.cat([global_mean_pool(x, index), x[data.center_idx[:, 0]], x[data.center_idx[:, 1]]], dim=-1)


