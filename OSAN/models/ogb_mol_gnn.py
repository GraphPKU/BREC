# credits to OGB team
# https://github.com/snap-stanford/ogb/blob/master/examples/graphproppred/mol/gnn.py
import torch
from torch_geometric.nn import global_add_pool, global_mean_pool, global_max_pool, GlobalAttention, Set2Set
from torch_scatter import scatter

from .ogb_mol_conv import GNN_node, GNN_node_Virtualnode, GNN_node_order


class OGBGNN(torch.nn.Module):

    def __init__(self, num_tasks, num_layer=5, emb_dim=300,
                 gnn_type='gin', virtual_node=True, residual=False, drop_ratio=0.5, JK="last", graph_pooling="mean"):
        """
            num_tasks (int): number of labels to be predicted
            virtual_node (bool): whether to add virtual node or not
        """

        super(OGBGNN, self).__init__()

        self.num_layer = num_layer
        self.drop_ratio = drop_ratio
        self.JK = JK
        self.emb_dim = emb_dim
        self.num_tasks = num_tasks
        self.graph_pooling = graph_pooling

        if self.num_layer < 2:
            raise ValueError("Number of GNN layers must be greater than 1.")

        # GNN to generate node embeddings
        if virtual_node:
            self.gnn_node = GNN_node_Virtualnode(num_layer, emb_dim, JK=JK, drop_ratio=drop_ratio, residual=residual,
                                                 gnn_type=gnn_type)
        else:
            self.gnn_node = GNN_node(num_layer, emb_dim, JK=JK, drop_ratio=drop_ratio, residual=residual,
                                     gnn_type=gnn_type)

        # Pooling function to generate whole-graph embeddings
        if self.graph_pooling == "sum":
            self.pool = global_add_pool
        elif self.graph_pooling == "mean":
            self.pool = global_mean_pool
        elif self.graph_pooling == "max":
            self.pool = global_max_pool
        elif self.graph_pooling == "attention":
            self.pool = GlobalAttention(
                gate_nn=torch.nn.Sequential(torch.nn.Linear(emb_dim, 2 * emb_dim),
                                            torch.nn.BatchNorm1d(2 * emb_dim),
                                            torch.nn.ReLU(),
                                            torch.nn.Linear(2 * emb_dim, 1)))
        elif self.graph_pooling == "set2set":
            self.pool = Set2Set(emb_dim, processing_steps=2)
        else:
            raise ValueError("Invalid graph pooling type.")

        if graph_pooling == "set2set":
            self.graph_pred_linear = torch.nn.Linear(2 * self.emb_dim, self.num_tasks)
        else:
            self.graph_pred_linear = torch.nn.Linear(self.emb_dim, self.num_tasks)

    def forward(self, data):
        h_node = self.gnn_node(data)

        if hasattr(data, 'selected_node_masks') and data.selected_node_masks is not None:
            if data.selected_node_masks.dtype == torch.float:
                h_node = h_node * data.selected_node_masks[:, None]
            elif data.selected_node_masks.dtype == torch.bool:
                h_node = h_node[data.selected_node_masks, :]
            else:
                raise ValueError

            num_nodes = scatter(data.selected_node_masks.detach(), data.batch, dim=0, reduce="sum")
            h_graph = global_add_pool(h_node, data.batch)
            h_graph = h_graph / num_nodes[:, None]
        else:
            h_graph = self.pool(h_node, data.batch)

        if hasattr(data, 'inter_graph_idx') and data.inter_graph_idx is not None:
            h_graph = global_mean_pool(h_graph, data.inter_graph_idx)
        return self.graph_pred_linear(h_graph)

    def reset_parameters(self):
        self.gnn_node.reset_parameters()
        if self.graph_pooling in ['set2set', 'attention']:
            self.pool.reset_parameters()
        self.graph_pred_linear.reset_parameters()


class OGBGNN_order(OGBGNN):
    def __init__(self, num_tasks, num_layer=5, emb_dim=300, extra_in_dim=1, extra_encode_dim=300,
                 gnn_type='gin', virtual_node=True, residual=False, drop_ratio=0.5, JK="last", graph_pooling="mean"):
        """
            num_tasks (int): number of labels to be predicted
            virtual_node (bool): whether to add virtual node or not
        """

        super(OGBGNN, self).__init__()

        self.num_layer = num_layer
        self.drop_ratio = drop_ratio
        self.JK = JK
        self.emb_dim = emb_dim
        self.num_tasks = num_tasks
        self.graph_pooling = graph_pooling

        if self.num_layer < 2:
            raise ValueError("Number of GNN layers must be greater than 1.")

        # GNN to generate node embeddings
        if virtual_node:
            raise NotImplementedError
        else:
            self.gnn_node = GNN_node_order(num_layer,
                                           emb_dim,
                                           extra_in_dim,
                                           extra_encode_dim,
                                           JK=JK,
                                           drop_ratio=drop_ratio,
                                           residual=residual,
                                           gnn_type=gnn_type)

        # Pooling function to generate whole-graph embeddings
        if self.graph_pooling == "sum":
            self.pool = global_add_pool
        elif self.graph_pooling == "mean":
            self.pool = global_mean_pool
        elif self.graph_pooling == "max":
            self.pool = global_max_pool
        elif self.graph_pooling == "attention":
            self.pool = GlobalAttention(
                gate_nn=torch.nn.Sequential(torch.nn.Linear(emb_dim, 2 * emb_dim),
                                            torch.nn.BatchNorm1d(2 * emb_dim),
                                            torch.nn.ReLU(),
                                            torch.nn.Linear(2 * emb_dim, 1)))
        elif self.graph_pooling == "set2set":
            self.pool = Set2Set(emb_dim, processing_steps=2)
        else:
            raise ValueError("Invalid graph pooling type.")

        if graph_pooling == "set2set":
            self.graph_pred_linear = torch.nn.Linear(2 * self.emb_dim, self.num_tasks)
        else:
            self.graph_pred_linear = torch.nn.Linear(self.emb_dim, self.num_tasks)

    def forward(self, data):
        h_node = self.gnn_node(data)

        assert data.selected_node_masks.dtype == torch.float
        h_node = h_node * data.selected_node_masks[:, None]
        h_node = self.pool(h_node, data.original_node_mask)
        h_graph = global_mean_pool(h_node, data.inter_graph_idx)
        return self.graph_pred_linear(h_graph)
