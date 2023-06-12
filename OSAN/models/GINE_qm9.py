from torch.nn import Sequential, Linear, ReLU
import torch
from torch_geometric.nn import MessagePassing, Set2Set, global_mean_pool


class GINConv(MessagePassing):
    def __init__(self, emb_dim, dim1, dim2):
        super(GINConv, self).__init__(aggr="add")

        self.bond_encoder = Sequential(Linear(emb_dim, dim1), torch.nn.BatchNorm1d(dim1), ReLU(), Linear(dim1, dim1),
                                       torch.nn.BatchNorm1d(dim1), ReLU())

        self.mlp = Sequential(Linear(dim1, dim2), torch.nn.BatchNorm1d(dim2), ReLU(), Linear(dim2, dim2),
                              torch.nn.BatchNorm1d(dim2), ReLU())

        self.eps = torch.nn.Parameter(torch.Tensor([0]))

    def forward(self, x, edge_index, edge_attr, edge_weight):
        if edge_weight is not None and edge_weight.ndim < 2:
            edge_weight = edge_weight[:, None]

        edge_embedding = self.bond_encoder(edge_attr)

        out = self.mlp((1 + self.eps) * x + self.propagate(edge_index,
                                                           x=x,
                                                           edge_attr=edge_embedding,
                                                           edge_weight=edge_weight))

        return out

    def message(self, x_j, edge_attr, edge_weight):
        # x_j has shape [E, out_channels]
        m = torch.relu(x_j + edge_attr)
        return m * edge_weight if edge_weight is not None else m

    def update(self, aggr_out):
        return aggr_out

    def reset_parameters(self):
        seq_lst1 = torch.nn.ModuleList(self.bond_encoder)
        for l in seq_lst1:
            if not isinstance(l, torch.nn.ReLU):
                l.reset_parameters()
        seq_lst2 = torch.nn.ModuleList(self.mlp)
        for l in seq_lst2:
            if not isinstance(l, torch.nn.ReLU):
                l.reset_parameters()
        self.eps = torch.nn.Parameter(torch.Tensor([0.]).to(self.eps.device))


class NetGINE_QM(torch.nn.Module):
    def __init__(self, input_dims,
                 edge_features,
                 dim,
                 num_convlayers,
                 num_class=1):
        super(NetGINE_QM, self).__init__()

        self.conv = torch.nn.ModuleList([GINConv(edge_features, input_dims, dim)])

        for _ in range(num_convlayers - 1):
            self.conv.append(GINConv(edge_features, dim, dim))

        self.set2set = Set2Set(1 * dim, processing_steps=6)

        self.fc1 = Linear(2 * dim, dim)
        self.fc4 = Linear(dim, num_class)

    def forward(self, data):
        x, edge_index, edge_attr, edge_weight = data.x, data.edge_index, data.edge_attr, data.edge_weight
        batch = data.batch if hasattr(data, 'batch') and data.batch is not None \
            else torch.zeros(x.shape[0], dtype=torch.long)

        for l in self.conv:
            x = torch.relu(l(x, edge_index, edge_attr, edge_weight))

        if hasattr(data, 'selected_node_masks') and data.selected_node_masks is not None:
            if data.selected_node_masks.dtype == torch.float:
                x = x * data.selected_node_masks[:, None]
            elif data.selected_node_masks.dtype == torch.bool:
                x = x[data.selected_node_masks, :]
            else:
                raise ValueError

        x = self.set2set(x, batch)
        x = torch.relu(self.fc1(x))
        if hasattr(data, 'inter_graph_idx') and data.inter_graph_idx is not None:
            x = global_mean_pool(x, data.inter_graph_idx)
        x = self.fc4(x)
        return x

    def reset_parameters(self):
        for l in self.conv:
            l.reset_parameters()
        self.set2set.reset_parameters()
        self.fc1.reset_parameters()
        self.fc4.reset_parameters()


class NetGINE_QM_ordered(torch.nn.Module):
    def __init__(self, input_dims,
                 edge_features,
                 dim,
                 num_convlayers,
                 extra_in_dim,
                 extra_encode_dim,
                 num_class=1):
        super(NetGINE_QM_ordered, self).__init__()

        self.encode_x = torch.nn.Linear(input_dims, dim)
        self.encode_extra = torch.nn.Linear(extra_in_dim, extra_encode_dim)

        self.conv = torch.nn.ModuleList([GINConv(edge_features, dim + extra_encode_dim, dim)])

        for _ in range(num_convlayers - 1):
            self.conv.append(GINConv(edge_features, dim, dim))

        self.set2set = Set2Set(1 * dim, processing_steps=6)

        self.fc1 = Linear(2 * dim, dim)
        self.fc4 = Linear(dim, num_class)

    def forward(self, data):
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        extra_feature = data.extra_feature
        x = torch.relu(self.encode_x(x))
        extra_feature = torch.relu(self.encode_extra(extra_feature))
        x = torch.cat([x, extra_feature], dim=-1)

        for l in self.conv:
            x = torch.relu(l(x, edge_index, edge_attr, None))

        assert data.selected_node_masks.dtype == torch.float
        x = x * data.selected_node_masks[:, None]

        x = self.set2set(x, data.original_node_mask)
        x = torch.relu(self.fc1(x))
        x = global_mean_pool(x, data.inter_graph_idx)
        x = self.fc4(x)
        return x

    def reset_parameters(self):
        self.encode_x.reset_parameters()
        self.encode_extra.reset_parameters()
        for l in self.conv:
            l.reset_parameters()
        self.set2set.reset_parameters()
        self.fc1.reset_parameters()
        self.fc4.reset_parameters()
