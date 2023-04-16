import torch
import torch_sparse
from ogb.graphproppred.mol_encoder import BondEncoder, AtomEncoder
from torch import nn as nn
from torch.nn import functional as F
from torch_geometric import nn as nng
from torch_geometric.nn import global_add_pool, global_mean_pool, global_max_pool, GlobalAttention, Set2Set, global_sort_pool
from torch_geometric.data import Batch, Data
from torch_sparse import SparseTensor, coalesce
from torch_scatter import scatter_add
from copy import copy
import pdb


def center_pool(x, node_to_subgraph):
    node_to_subgraph = node_to_subgraph.cpu().numpy()
    # the first node of each subgraph is its center
    _, center_indices = np.unique(node_to_subgraph, return_index=True)
    x = x[center_indices]
    return x


class MLP(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.main = [
            nn.Linear(in_dim, 2 * in_dim),
            nn.BatchNorm1d(2 * in_dim),
            nn.ReLU()
        ]
        self.main.append(nn.Linear(2 * in_dim, out_dim))
        self.main = nn.Sequential(*self.main)

    def forward(self, x):
        return self.main(x)


#################
# Embeddings
#################
class OGBMolEmbedding(nn.Module):
    def __init__(self, dim, embed_edge=True, x_as_list=False):
        super().__init__()
        self.atom_embedding = AtomEncoder(emb_dim=dim)
        if embed_edge:
            self.edge_embedding = BondEncoder(emb_dim=dim)
        self.x_as_list = x_as_list

    def forward(self, data):
        data = new(data)
        data.x = self.atom_embedding(data.x)
        if self.x_as_list:
            data.x = [data.x]
        if hasattr(self, 'edge_embedding'):
            data.edge_attr = self.edge_embedding(data.edge_attr)
        return data


class NodeEmbedding(nn.Module):
    def __init__(self, in_dim, out_dim, x_as_list=False):
        super().__init__()
        self.embedding = nn.Embedding(num_embeddings=in_dim, embedding_dim=out_dim)
        self.x_as_list = x_as_list

    def forward(self, data):
        data = new(data)
        data.x = self.embedding(data.x)
        if self.x_as_list:
            data.x = [data.x]
        return data


class VNAgg(nn.Module):
    def __init__(self, dim, conv_type="gin"):
        super().__init__()
        self.conv_type = conv_type
        if "gin" in conv_type:
            self.mlp = nn.Sequential(
                MLP(dim, dim),
                nn.BatchNorm1d(dim),
                nn.ReLU()
            )
        elif "gcn" in conv_type:
            self.W0 = nn.Linear(dim, dim)
            self.W1 = nn.Linear(dim, dim)
            self.nl_bn = nn.Sequential(
                nn.BatchNorm1d(dim),
                nn.ReLU()
            )
        else:
            raise NotImplementedError('Unrecognised model conv : {}'.format(conv_type))

    def forward(self, virtual_node, embeddings, batch_vector):
        if batch_vector.size(0) > 0:  # ...or the operation will crash for empty graphs
            G = nng.global_add_pool(embeddings, batch_vector)
        else:
            G = torch.zeros_like(virtual_node)
        if "gin" in self.conv_type:
            virtual_node = virtual_node + G
            virtual_node = self.mlp(virtual_node)
        elif "gcn" in self.conv_type:
            virtual_node = self.W0(virtual_node) + self.W1(G)
            virtual_node = self.nl_bn(virtual_node)
        else:
            raise NotImplementedError('Unrecognised model conv : {}'.format(self.conv_type))
        return virtual_node


class ConvBlock(nn.Module):
    def __init__(self, dim, dropout=0.5, activation=F.relu, virtual_node=False, virtual_node_agg=True,
                 k=4, last_layer=False, conv_type='gin', edge_embedding=None):
        super().__init__()
        self.edge_embed = edge_embedding
        self.conv_type = conv_type
        if conv_type == 'gin+':
            self.conv = GINEPLUS(MLP(dim, dim), dim, k=k)
        elif conv_type == 'naivegin+':
            self.conv = NAIVEGINEPLUS(MLP(dim, dim), dim, k=k)
        elif conv_type == 'gin':
            self.conv = nng.GINEConv(MLP(dim, dim), train_eps=True)
        elif conv_type == 'gcn':
            self.conv = nng.GCNConv(dim, dim)
        self.norm = nn.BatchNorm1d(dim)
        self.act = activation or nn.Identity()
        self.last_layer = last_layer

        self.dropout_ratio = dropout

        self.virtual_node = virtual_node
        self.virtual_node_agg = virtual_node_agg
        if self.virtual_node and self.virtual_node_agg:
            self.vn_aggregator = VNAgg(dim, conv_type=conv_type)

    def forward(self, data):
        data = new(data)
        x, ei, ea, b = data.x, data.edge_index, data.edge_attr, data.batch
        mhei, d = data.multihop_edge_index, data.distance
        h = x
        if self.virtual_node:
            if self.conv_type == 'gin+':
                h[0] = h[0] + data.virtual_node[b]
            else:
                h = h + data.virtual_node[b]
        if self.conv_type == 'gin':
            H = self.conv(h, ei, edge_attr=self.edge_embed(ea))
        elif self.conv_type == 'gcn':
            H = self.conv(h, ei)
        else:
            H = self.conv(h, mhei, d, self.edge_embed(ea))
        if self.conv_type == 'gin+':
            h = H[0]
        else:
            h = H
        h = self.norm(h)
        if not self.last_layer:
            h = self.act(h)
        h = F.dropout(h, self.dropout_ratio, training=self.training)

        if self.virtual_node and self.virtual_node_agg:
            v = self.vn_aggregator(data.virtual_node, h, b)
            v = F.dropout(v, self.dropout_ratio, training=self.training)
            data.virtual_node = v
        if self.conv_type == 'gin+':
            H[0] = h
            h = H
        data.x = h
        return data


class GlobalPool(nn.Module):
    def __init__(self, fun, cat_size=False, cat_candidates=False, subgraph_to_graph='False'):
        super().__init__()
        self.cat_size = cat_size
        self.fun = getattr(nng, "global_{}_pool".format(fun.lower()))
        self.cat_candidates = cat_candidates
        self.subgraph_to_graph = subgraph_to_graph

    def forward(self, batch):
        if self.subgraph_to_graph:
            x, b = batch.x, batch.subgraph_to_graph
        else:
            x, b = batch.x, batch.batch
        pooled = self.fun(x, b, size=batch.num_graphs)
        if self.cat_size:
            sizes = nng.global_add_pool(torch.ones(x.size(0), 1).type_as(x), b, size=batch.num_graphs)
            pooled = torch.cat([pooled, sizes], dim=1)
        if self.cat_candidates:
            ei = batch.edge_index
            mask = batch.edge_attr == 3
            candidates = scatter_add(x[ei[0, mask]], b[ei[0, mask]], dim=0, dim_size=batch.num_graphs)
            pooled = torch.cat([pooled, candidates], dim=1)
        return pooled


class ClassifierNetwork(nn.Module):
    def __init__(self, hidden=100, out_dim=128, layers=3, dropout=0.5, virtual_node=False,
                 k=4, conv_type='gin', nested=False, subgraph_pooling='mean'):
        super().__init__()
        self.k = k
        self.conv_type = conv_type
        convs = [ConvBlock(hidden,
                           dropout=dropout,
                           virtual_node=virtual_node,
                           k=min(i + 1, k),
                           conv_type=conv_type,
                           edge_embedding=BondEncoder(emb_dim=hidden))
                 for i in range(layers - 1)]
        convs.append(ConvBlock(hidden,
                               dropout=dropout,
                               virtual_node=virtual_node,
                               virtual_node_agg=False,  # on last layer, use but do not update virtual node
                               last_layer=True,
                               k=min(layers, k),
                               conv_type=conv_type,
                               edge_embedding=BondEncoder(emb_dim=hidden)))
        self.main = nn.Sequential(
            OGBMolEmbedding(hidden, embed_edge=False, x_as_list=(conv_type == 'gin+')),
            *convs)
        self.aggregate = nn.Sequential(
            GlobalPool('mean', subgraph_to_graph=nested),
            nn.Linear(hidden, out_dim)
        )
        self.virtual_node = virtual_node
        if self.virtual_node:
            self.v0 = nn.Parameter(torch.zeros(1, hidden), requires_grad=True)

        ### Pooling function to generate sub-graph embeddings
        self.subgraph_pooling = subgraph_pooling
        if self.subgraph_pooling == "sum":
            self.subpool = global_add_pool
        elif self.subgraph_pooling == "mean":
            self.subpool = global_mean_pool
        elif self.subgraph_pooling == "max":
            self.subpool = global_max_pool
        elif self.subgraph_pooling == "attention":
            self.subpool = GlobalAttention(gate_nn = torch.nn.Sequential(torch.nn.Linear(emb_dim, 2*emb_dim), torch.nn.BatchNorm1d(2*emb_dim), torch.nn.ReLU(), torch.nn.Linear(2*emb_dim, 1)))
        elif self.subgraph_pooling == "center":
            self.subpool = center_pool
        else:
            self.subpool = None

    def forward(self, data):
        data = make_multihop_edges(data, self.k)
        if self.virtual_node:
            data.virtual_node = self.v0.expand(data.num_graphs, self.v0.shape[-1])
        g = self.main(data)
        if self.conv_type == 'gin+':
            g.x = g.x[0]

        if 'node_to_subgraph' in data and 'subgraph_to_graph' in data:
            g.x = self.subpool(g.x, g.node_to_subgraph)

        return self.aggregate(g)


def make_multihop_edges(data, k):
    """
    Adds edges corresponding to distances up to k to a data object.
    :param data: torch_geometric.data object, in coo format
    (ie an edge (i, j) with label v is stored with an arbitrary index u as:
     edge_index[0, u] = i, edge_index[1, u]=j, edge_attr[u]=v)
    :return: a new data object with new fields, multihop_edge_index and distance.
    distance[u] contains values from 1 to k corresponding to the distance between
    multihop_edge_index[0, u] and multihop_edge_index[1, u]
    """
    data = new(data)

    N = data.num_nodes
    E = data.num_edges
    if E == 0:
        data.multihop_edge_index = torch.empty_like(data.edge_index)
        data.distance = torch.empty_like(data.multihop_edge_index[0])
        return data

    # Get the distance 0
    multihop_edge_index = torch.arange(0, N, dtype=data.edge_index[0].dtype, device=data.x.device)
    distance = torch.zeros_like(multihop_edge_index)
    multihop_edge_index = multihop_edge_index.unsqueeze(0).repeat(2, 1)

    # Get the distance 1
    multihop_edge_index = torch.cat((multihop_edge_index, data.edge_index), dim=1)
    distance = torch.cat((distance, torch.ones_like(data.edge_index[0])), dim=0)

    A = SparseTensor(row=data.edge_index[0], col=data.edge_index[1], value=torch.ones_like(data.edge_index[0]).float(),
                     sparse_sizes=(N, N), is_sorted=False)
    Ad = A  # A to the power d

    # Get larger distances
    for d in range(2, k + 1):
        Ad = torch_sparse.matmul(Ad, A)
        row, col, v = Ad.coo()
        d_edge_index = torch.stack((row, col))
        d_edge_attr = torch.empty_like(row).fill_(d)
        multihop_edge_index = torch.cat((multihop_edge_index, d_edge_index), dim=1)
        distance = torch.cat((distance, d_edge_attr), dim=0)

    # remove dupicate, keep only shortest distance
    multihop_edge_index, distance = coalesce(multihop_edge_index, distance, N, N, op='min')

    data.multihop_edge_index = multihop_edge_index
    data.distance = distance

    return data


class NAIVEGINEPLUS(nng.MessagePassing):
    def __init__(self, fun, dim, k=4, **kwargs):
        super().__init__(aggr='add')
        self.k = k
        self.nn = fun
        self.eps = nn.Parameter(torch.zeros(k + 1, dim), requires_grad=True)

    def forward(self, x, multihop_edge_index, distance, edge_attr):
        assert x.size(-1) == edge_attr.size(-1)
        result = (1 + self.eps[0]) * x
        for i in range(self.k):
            if i == 0:
                out = self.propagate(multihop_edge_index[:, distance == i + 1], edge_attr=edge_attr, x=x)
            else:
                out = self.propagate(multihop_edge_index[:, distance == i + 1], edge_attr=None, x=x)
            result += (1 + self.eps[i + 1]) * out
        result = self.nn(result)
        return result

    def message(self, x_j, edge_attr):
        if edge_attr is not None:
            return F.relu(x_j + edge_attr)
        else:
            return F.relu(x_j)

    def __repr__(self):
        return '{}(nn={}, k={})'.format(self.__class__.__name__, self.nn, self.eps.size(0))


class GINEPLUS(nng.MessagePassing):
    def __init__(self, fun, dim, k=4, **kwargs):
        super().__init__(aggr='add')
        self.k = k
        self.nn = fun
        self.eps = nn.Parameter(torch.zeros(k + 1, dim), requires_grad=True)

    def forward(self, XX, multihop_edge_index, distance, edge_attr):
        """Warning, XX is now a list of previous xs, with x[0] being the last layer"""
        assert len(XX) >= self.k
        assert XX[-1].size(-1) == edge_attr.size(-1)
        result = (1 + self.eps[0]) * XX[0]
        for i, x in enumerate(XX):
            if i >= self.k:
                break
            if i == 0:
                out = self.propagate(multihop_edge_index[:, distance == i + 1], edge_attr=edge_attr, x=x)
            else:
                out = self.propagate(multihop_edge_index[:, distance == i + 1], edge_attr=None, x=x)
            result += (1 + self.eps[i + 1]) * out
        result = self.nn(result)
        return [result] + XX

    def message(self, x_j, edge_attr):
        if edge_attr is not None:
            return F.relu(x_j + edge_attr)
        else:
            return F.relu(x_j)


def new(data):
    """returns a new torch_geometric.data.Data containing the same tensors.
    Faster than data.clone() (tensors are not cloned) and preserves the old data object as long as
    tensors are not modified in-place. Intended to modify data object, without modyfing the tensors.
    ex:
    d1 = data(x=torch.eye(3))
    d2 = new(d1)
    d2.x = 2*d2.x
    In that case, d1.x was not changed."""
    return copy(data)
