import torch
from torch.nn import Linear, ReLU
import torch.nn.functional as F
from torch_geometric.nn import global_mean_pool, MessagePassing, global_add_pool
from torch_scatter import scatter

from .nn_utils import residual


class GINEConv(MessagePassing):
    def __init__(self, emb_dim, dim1, dim2):
        super(GINEConv, self).__init__(aggr="add")

        # disable the bias, otherwise the information will be nonzero
        self.bond_encoder = torch.nn.ModuleList([Linear(emb_dim, dim1), ReLU(), Linear(dim1, dim1)])
        self.mlp = torch.nn.ModuleList([Linear(dim1, dim1), ReLU(), Linear(dim1, dim2)])
        self.eps = torch.nn.Parameter(torch.Tensor([0]))

    def forward(self, x, edge_index, edge_attr, edge_weight):
        if edge_weight is not None and edge_weight.ndim < 2:
            edge_weight = edge_weight[:, None]

        edge_embedding = edge_attr
        for l in self.bond_encoder:
            edge_embedding = l(edge_embedding)
        out = (1 + self.eps) * x + self.propagate(edge_index, x=x, edge_attr=edge_embedding, edge_weight=edge_weight)
        for l in self.mlp:
            out = l(out)

        return out

    def message(self, x_j, edge_attr, edge_weight):
        # x_j has shape [E, out_channels]
        m = torch.relu(x_j + edge_attr)
        return m * edge_weight if edge_weight is not None else m

    def update(self, aggr_out):
        return aggr_out

    def reset_parameters(self):
        for l in self.bond_encoder:
            if isinstance(l, torch.nn.Linear):
                l.reset_parameters()
        for l in self.mlp:
            if isinstance(l, torch.nn.Linear):
                l.reset_parameters()
        self.eps = torch.nn.Parameter(torch.Tensor([0]).to(self.eps.device))


class NetGINE(torch.nn.Module):
    def __init__(self, input_dims,
                 edge_features,
                 dim,
                 dropout,
                 num_convlayers,
                 jk=None,
                 num_class=1):
        super(NetGINE, self).__init__()

        self.dropout = dropout
        self.jk = jk
        assert num_convlayers > 1

        self.conv = torch.nn.ModuleList([GINEConv(edge_features, input_dims, dim)])
        self.bn = torch.nn.ModuleList([torch.nn.BatchNorm1d(dim)])

        for _ in range(num_convlayers - 1):
            self.conv.append(GINEConv(edge_features, dim, dim))
            self.bn.append(torch.nn.BatchNorm1d(dim))

        if self.jk == 'concat':
            self.fc1 = Linear(num_convlayers * dim, dim)
        elif self.jk == 'residual' or self.jk is None:
            self.fc1 = Linear(dim, dim)
        else:
            raise ValueError(f"Unsupported jumping connection type{self.jk}")

        self.fc2 = Linear(dim, dim)
        self.fc3 = Linear(dim, dim)
        self.fc4 = Linear(dim, num_class)

    def forward(self, data):
        x, edge_index, edge_attr, edge_weight = data.x, data.edge_index, data.edge_attr, data.edge_weight
        batch = data.batch if hasattr(data, 'batch') and data.batch is not None \
            else torch.zeros(x.shape[0], dtype=torch.long)

        intermediate_x = [] if self.jk == 'concat' else None
        for i, (conv, bn) in enumerate(zip(self.conv, self.bn)):
            x_new = conv(x, edge_index, edge_attr, edge_weight)
            x_new = bn(x_new)
            x_new = torch.relu(x_new)
            x_new = torch.dropout(x_new, p=self.dropout, train=self.training)
            x = residual(x, x_new) if self.jk == 'residual' else x_new
            if intermediate_x is not None:
                intermediate_x.append(x)

        x = torch.cat(intermediate_x, dim=-1) if intermediate_x is not None else x
        if hasattr(data, 'selected_node_masks') and data.selected_node_masks is not None:
            if data.selected_node_masks.dtype == torch.float:
                x = x * data.selected_node_masks[:, None]
            elif data.selected_node_masks.dtype == torch.bool:
                x = x[data.selected_node_masks, :]
            else:
                raise ValueError

            num_nodes = scatter(data.selected_node_masks.detach(), batch, dim=0, reduce="sum")
            x = global_add_pool(x, batch)
            x = x / num_nodes[:, None]
        else:
            x = global_mean_pool(x, batch)

        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        if hasattr(data, 'inter_graph_idx') and data.inter_graph_idx is not None:
            x = global_mean_pool(x, data.inter_graph_idx)
        x = self.fc4(x)
        return x

    def reset_parameters(self):
        for l in self.conv:
            l.reset_parameters()
        for l in self.bn:
            l.reset_parameters()
        self.fc1.reset_parameters()
        self.fc2.reset_parameters()
        self.fc3.reset_parameters()
        self.fc4.reset_parameters()


class NetGINE_ordered(NetGINE):
    def __init__(self, input_dims,
                 edge_features,
                 dim,
                 extra_in_dim,
                 extra_encode_dim,
                 dropout,
                 num_convlayers,
                 jk=None,
                 num_class=1):
        super(NetGINE, self).__init__()

        self.dropout = dropout
        self.jk = jk
        assert num_convlayers > 1

        self.encode_x = torch.nn.Linear(input_dims, dim)
        self.encode_extra = torch.nn.Linear(extra_in_dim, extra_encode_dim)

        self.conv = torch.nn.ModuleList([GINEConv(edge_features, dim + extra_encode_dim, dim)])
        self.bn = torch.nn.ModuleList([torch.nn.BatchNorm1d(dim)])

        for _ in range(num_convlayers - 1):
            self.conv.append(GINEConv(edge_features, dim, dim))
            self.bn.append(torch.nn.BatchNorm1d(dim))

        if self.jk == 'concat':
            self.fc1 = Linear(num_convlayers * dim, dim)
        elif self.jk == 'residual' or self.jk is None:
            self.fc1 = Linear(dim, dim)
        else:
            raise ValueError(f"Unsupported jumping connection type{self.jk}")

        self.fc2 = Linear(dim, dim)
        self.fc3 = Linear(dim, dim)
        self.fc4 = Linear(dim, num_class)

    def forward(self, data):
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        extra_feature = data.extra_feature

        x = F.relu(self.encode_x(x))
        extra_feature = F.relu(self.encode_extra(extra_feature))
        x = torch.cat([x, extra_feature], dim=-1)

        intermediate_x = [] if self.jk == 'concat' else None
        for i, (conv, bn) in enumerate(zip(self.conv, self.bn)):
            x_new = conv(x, edge_index, edge_attr, None)
            x_new = bn(x_new)
            x_new = torch.relu(x_new)
            x_new = torch.dropout(x_new, p=self.dropout, train=self.training)
            x = residual(x, x_new) if self.jk == 'residual' else x_new
            if intermediate_x is not None:
                intermediate_x.append(x)

        x = torch.cat(intermediate_x, dim=-1) if intermediate_x is not None else x
        assert data.selected_node_masks.dtype == torch.float
        x = x * data.selected_node_masks[:, None]

        x = global_add_pool(x, data.original_node_mask)

        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        x = global_mean_pool(x, data.inter_graph_idx)
        x = self.fc4(x)
        return x

    def reset_parameters(self):
        self.encode_extra.reset_parameters()
        self.encode_x.reset_parameters()

        for l in self.conv:
            l.reset_parameters()
        for l in self.bn:
            l.reset_parameters()
        self.fc1.reset_parameters()
        self.fc2.reset_parameters()
        self.fc3.reset_parameters()
        self.fc4.reset_parameters()
