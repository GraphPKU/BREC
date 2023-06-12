# https://github.com/beabevi/ESAN/blob/master/models.py
from typing import Union, Optional
import torch
import torch.nn.functional as F
from torch_geometric.nn import GraphConv, MessagePassing
from torch_geometric.nn import GINConv as PyGINConv
from torch_geometric.nn import global_add_pool, global_mean_pool, global_max_pool, GlobalAttention
from torch_geometric.typing import OptPairTensor, Adj
from torch_scatter import scatter

from .ogb_mol_conv import GINConv


class ZincAtomEncoder(torch.nn.Module):
    def __init__(self, policy=None, emb_dim=64):
        super(ZincAtomEncoder, self).__init__()
        self.policy = policy
        self.num_added = 2
        self.enc = torch.nn.Embedding(28, emb_dim)

    def forward(self, x):
        # my version of data.x is onehot of shape(batchsize, 28)
        if self.policy == 'ego_nets_plus':
            raise NotImplementedError
        else:
            return self.enc(torch.argmax(x, dim=1))

    def reset_parameters(self):
        self.enc.reset_parameters()


class ZINCGINConv(MessagePassing):
    def __init__(self, in_dim, emb_dim):
        super(ZINCGINConv, self).__init__(aggr="add")

        self.mlp = torch.nn.Sequential(torch.nn.Linear(in_dim, emb_dim), torch.nn.BatchNorm1d(emb_dim), torch.nn.ReLU(),
                                       torch.nn.Linear(emb_dim, emb_dim))
        self.eps = torch.nn.Parameter(torch.Tensor([0]))
        self.bond_encoder = torch.nn.Embedding(3, in_dim)

    def forward(self, x, edge_index, edge_attr, edge_weight=None):
        if edge_weight is not None and edge_weight.ndim < 2:
            edge_weight = edge_weight[:, None]

        edge_embedding = self.bond_encoder(torch.argmax(edge_attr, dim=1))
        out = self.mlp((1 + self.eps) * x + self.propagate(edge_index,
                                                           x=x,
                                                           edge_attr=edge_embedding,
                                                           edge_weight=edge_weight))
        return out

    def message(self, x_j, edge_attr, edge_weight):
        m = torch.relu(x_j + edge_attr)
        return m * edge_weight if edge_weight is not None else m

    def update(self, aggr_out):
        return aggr_out

    def reset_parameters(self):
        lst = torch.nn.ModuleList(self.mlp)
        for l in lst:
            if not isinstance(l, torch.nn.ReLU):
                l.reset_parameters()
        self.eps = torch.nn.Parameter(torch.Tensor([0.]).to(self.eps.device))
        self.bond_encoder.reset_parameters()


class MyPyGINConv(PyGINConv):
    def forward(self, x: Union[torch.Tensor, OptPairTensor],
                edge_index: Adj,
                edge_attr: torch.Tensor,
                edge_weight: Optional[torch.Tensor]) -> torch.Tensor:
        """"""
        if isinstance(x, torch.Tensor):
            x = (x, x)

        # propagate_type: (x: OptPairTensor)
        out = self.propagate(edge_index, x=x[0], edge_weight=edge_weight,)

        x_r = x[1]
        if x_r is not None:
            out += (1 + self.eps) * x_r

        return self.nn(out)

    def message(self, x_j: torch.Tensor, edge_weight: Optional[torch.Tensor]) -> torch.Tensor:
        return x_j if edge_weight is None else x_j * edge_weight[:, None]

    # def message_and_aggregate(self, adj_t: SparseTensor, x: OptPairTensor) -> torch.Tensor:
    #     adj_t = adj_t.set_value(None, layout=None)
    #     return matmul(adj_t, x[0], reduce=self.aggr)


class OriginalGINConv(torch.nn.Module):
    def __init__(self, in_dim, emb_dim):
        super(OriginalGINConv, self).__init__()
        mlp = torch.nn.Sequential(
            torch.nn.Linear(in_dim, emb_dim),
            torch.nn.BatchNorm1d(emb_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(emb_dim, emb_dim)
        )
        self.layer = MyPyGINConv(nn=mlp, train_eps=False)

    def forward(self, x, edge_index, edge_attr, edge_weight):
        return self.layer(x, edge_index, edge_attr, edge_weight)

    def reset_parameters(self):
        self.layer.reset_parameters()


class GNN_node(torch.nn.Module):
    """
    Output:
        node representations
    """

    def __init__(self, num_layer, in_dim, emb_dim, drop_ratio=0.5, JK="last", residual=False, gnn_type='gin',
                 num_random_features=0):

        super(GNN_node, self).__init__()
        self.num_layer = num_layer
        self.drop_ratio = drop_ratio
        self.JK = JK
        # add residual connection or not
        self.residual = residual
        self.gnn_type = gnn_type

        if self.num_layer < 2:
            raise ValueError("Number of GNN layers must be greater than 1.")

        self.atom_encoder = torch.nn.Linear(in_dim, emb_dim)
        self.num_random_features = num_random_features

        if num_random_features > 0:
            assert gnn_type == 'graphconv'

            self.initial_layers = torch.nn.ModuleList(
                [GraphConv(in_dim, emb_dim // 2), GraphConv(emb_dim // 2, emb_dim - num_random_features)]
            )
            # now the next layers will have dimension emb_dim
            in_dim = emb_dim

        # List of GNNs
        self.convs = torch.nn.ModuleList()
        self.batch_norms = torch.nn.ModuleList()

        for layer in range(num_layer):
            if gnn_type == 'gin':
                self.convs.append(GINConv(emb_dim, emb_dim))
            elif gnn_type == 'gcn':
                raise NotImplementedError
            elif gnn_type == 'originalgin':
                self.convs.append(OriginalGINConv(emb_dim, emb_dim))
            elif gnn_type == 'zincgin':
                self.convs.append(ZINCGINConv(emb_dim, emb_dim))
            elif gnn_type == 'graphconv':
                raise NotImplementedError
            else:
                raise ValueError('Undefined GNN type called {}'.format(gnn_type))

            self.batch_norms.append(torch.nn.BatchNorm1d(emb_dim))

    def forward(self, data):
        x, edge_index, edge_attr, batch = data.x, data.edge_index, data.edge_attr, data.batch
        edge_weight = data.edge_weight

        if self.num_random_features > 0:
            for layer in self.initial_layers:
                x = F.elu(layer(x, edge_index, edge_attr))

            # Implementation of RNI
            random_dims = torch.empty(x.shape[0], self.num_random_features).to(x.device)
            torch.nn.init.normal_(random_dims)
            x = torch.cat([x, random_dims], dim=1)

        # computing input node embedding
        h_list = [self.atom_encoder(x[:, 0])]

        for layer in range(self.num_layer):

            h = self.convs[layer](h_list[layer], edge_index, edge_attr, edge_weight)

            h = self.batch_norms[layer](h)

            if self.gnn_type not in ['gin', 'gcn'] or layer != self.num_layer - 1:
                h = F.relu(h)  # remove last relu for ogb

            if self.drop_ratio > 0.:
                h = F.dropout(h, self.drop_ratio, training=self.training)

            if self.residual:
                h += h_list[layer]

            h_list.append(h)

        # Different implementations of Jk-concat
        if self.JK == "last":
            node_representation = h_list[-1]
        elif self.JK == "sum":
            node_representation = 0
            for layer in range(self.num_layer + 1):
                node_representation += h_list[layer]
        elif self.JK == "concat":
            node_representation = torch.cat(h_list, dim=1)
        else:
            raise ValueError

        return node_representation

    def reset_parameters(self):
        self.atom_encoder.reset_parameters()
        for c in self.convs:
            c.reset_parameters()
        for b in self.batch_norms:
            b.reset_parameters()


def subgraph_pool(h_node, batched_data, pool):
    # Represent each subgraph as the pool of its node representations
    num_subgraphs = batched_data.num_subgraphs
    tmp = torch.cat([torch.zeros(1, device=num_subgraphs.device, dtype=num_subgraphs.dtype),
                     torch.cumsum(num_subgraphs, dim=0)])
    graph_offset = tmp[batched_data.batch]

    subgraph_idx = graph_offset + batched_data.subgraph_batch

    return pool(h_node, subgraph_idx)


class GNN(torch.nn.Module):

    def __init__(self, num_tasks, num_layer=5, in_dim=300, emb_dim=300,
                 gnn_type='gin', num_random_features=0, residual=False, drop_ratio=0.5, JK="last", graph_pooling="mean"):

        super(GNN, self).__init__()

        self.num_layer = num_layer
        self.drop_ratio = drop_ratio
        self.JK = JK
        self.emb_dim = emb_dim
        self.out_dim = self.emb_dim if self.JK == 'last' else self.emb_dim * (self.num_layer + 1)
        self.num_tasks = num_tasks
        self.graph_pooling = graph_pooling

        if self.num_layer < 2:
            raise ValueError("Number of GNN layers must be greater than 1.")

        # GNN to generate node embeddings
        self.gnn_node = GNN_node(num_layer, in_dim, emb_dim, JK=JK, drop_ratio=drop_ratio, residual=residual,
                                 gnn_type=gnn_type, num_random_features=num_random_features)

        # Pooling function to generate whole-graph embeddings
        if self.graph_pooling == "sum":
            self.pool = global_add_pool
        elif self.graph_pooling == "mean":
            self.pool = global_mean_pool
        elif self.graph_pooling == "max":
            self.pool = global_max_pool
        elif self.graph_pooling == "attention":
            self.pool = GlobalAttention(
                gate_nn=torch.nn.Sequential(torch.nn.Linear(emb_dim, 2 * emb_dim), torch.nn.BatchNorm1d(2 * emb_dim),
                                            torch.nn.ReLU(), torch.nn.Linear(2 * emb_dim, 1)))
        else:
            raise ValueError("Invalid graph pooling type.")

    # in their implementation, they mean the nodes first then the graphs
    # def forward(self, data):
    #     h_node = self.gnn_node(data)
    #     return subgraph_pool(h_node, batched_data, self.pool)

    def forward(self, data):
        h_node = self.gnn_node(data)
        if hasattr(data, 'selected_node_masks') and data.selected_node_masks is not None:
            if data.selected_node_masks.dtype == torch.float:
                h_node = h_node * data.selected_node_masks[:, None]
            elif data.selected_node_masks.dtype == torch.bool:
                h_node = h_node[data.selected_node_masks, :]
            else:
                raise ValueError

            if self.graph_pooling == "mean":
                num_nodes = scatter(data.selected_node_masks.detach(), data.batch, dim=0, reduce="sum")
                h_graph = global_add_pool(h_node, data.batch)
                h_graph = h_graph / num_nodes[:, None]
            else:
                h_graph = self.pool(h_node, data.batch)
        else:
            h_graph = self.pool(h_node, data.batch)
        return h_graph

    def reset_parameters(self):
        self.gnn_node.reset_parameters()
        if self.graph_pooling == 'attention':
            self.pool.reset_parameters()


class DSnetwork(torch.nn.Module):
    def __init__(self, subgraph_gnn, channels, num_tasks, invariant):
        super(DSnetwork, self).__init__()
        self.subgraph_gnn = subgraph_gnn
        self.invariant = invariant

        fc_list = []
        fc_sum_list = []
        for i in range(len(channels)):
            fc_list.append(torch.nn.Linear(in_features=channels[i - 1] if i > 0 else subgraph_gnn.out_dim,
                                           out_features=channels[i]))
            if self.invariant:
                fc_sum_list.append(torch.nn.Linear(in_features=channels[i],
                                                   out_features=channels[i]))
            else:
                fc_sum_list.append(torch.nn.Linear(in_features=channels[i - 1] if i > 0 else subgraph_gnn.out_dim,
                                                   out_features=channels[i]))

        self.fc_list = torch.nn.ModuleList(fc_list)
        self.fc_sum_list = torch.nn.ModuleList(fc_sum_list)

        self.final_layers = torch.nn.Sequential(
            torch.nn.Linear(in_features=channels[-1], out_features=2 * channels[-1]),
            torch.nn.ReLU(),
            torch.nn.Linear(in_features=2 * channels[-1], out_features=num_tasks)
        )

    def forward(self, batched_data):
        h_subgraph = self.subgraph_gnn(batched_data)

        if hasattr(batched_data, 'inter_graph_idx'):
            batch_idx = batched_data.inter_graph_idx
        else:
            batch_idx = torch.arange(h_subgraph.shape[0], device=h_subgraph.device)

        if self.invariant:
            for layer_idx, (fc, fc_sum) in enumerate(zip(self.fc_list, self.fc_sum_list)):
                x1 = fc(h_subgraph)

                h_subgraph = F.elu(x1)

            # aggregate to obtain a representation of the graph given the representations of the subgraphs
            h_graph = scatter(src=h_subgraph, index=batch_idx, dim=0, reduce="mean")
            for layer_idx, fc_sum in enumerate(self.fc_sum_list):
                h_graph = F.elu(fc_sum(h_graph))
        else:
            for layer_idx, (fc, fc_sum) in enumerate(zip(self.fc_list, self.fc_sum_list)):
                x1 = fc(h_subgraph)
                x2 = fc_sum(
                    scatter(src=h_subgraph, index=batch_idx, dim=0, reduce="mean")
                )

                h_subgraph = F.elu(x1 + x2[batch_idx])

            # aggregate to obtain a representation of the graph given the representations of the subgraphs
            h_graph = scatter(src=h_subgraph, index=batch_idx, dim=0, reduce="mean")

        return self.final_layers(h_graph)

    def reset_parameters(self):
        self.subgraph_gnn.reset_parameters()
        for l in self.fc_list:
            l.reset_parameters()
        for l in self.fc_sum_list:
            l.reset_parameters()
        lst = torch.nn.ModuleList(self.final_layers)
        for l in lst:
            if not isinstance(l, torch.nn.ReLU):
                l.reset_parameters()
