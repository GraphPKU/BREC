import torch
from torch_geometric.nn import Set2Set, global_mean_pool

from .GINE_gnn import GINEConv


class NetGINEAlchemy(torch.nn.Module):
    def __init__(self, input_dims, edge_features, dim, num_class, num_layers):
        super(NetGINEAlchemy, self).__init__()
        assert num_layers >= 1

        self.conv = torch.nn.ModuleList([GINEConv(edge_features, input_dims, dim)])
        for _ in range(num_layers - 1):
            self.conv.append(GINEConv(edge_features, dim, dim))

        self.set2set = Set2Set(1 * dim, processing_steps=6)

        self.fc1 = torch.nn.Linear(2 * dim, dim)
        self.fc4 = torch.nn.Linear(dim, num_class)

    def forward(self, data):
        x, edge_index, edge_attr, edge_weight = data.x, data.edge_index, data.edge_attr, data.edge_weight
        batch = data.batch if hasattr(data, 'batch') and data.batch is not None \
            else torch.zeros(x.shape[0], dtype=torch.long)

        for conv in self.conv:
            x = torch.relu(conv(x, edge_index, edge_attr, edge_weight))

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
