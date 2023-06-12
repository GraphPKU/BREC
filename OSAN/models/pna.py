import torch
from torch import Tensor
import torch.nn.functional as F
from torch.nn import Linear, ModuleList, ReLU, Sequential

from torch_geometric.nn import BatchNorm, PNAConv, global_add_pool, global_mean_pool
from torch_geometric.utils import degree
from torch_geometric.typing import Adj, OptTensor


class MyPNAConv(PNAConv):
    def forward(self, x: Tensor, edge_index: Adj, edge_attr: OptTensor = None, edge_weight: OptTensor = None) -> Tensor:
        """"""

        if self.divide_input:
            x = x.view(-1, self.towers, self.F_in)
        else:
            x = x.view(-1, 1, self.F_in).repeat(1, self.towers, 1)

        # propagate_type: (x: Tensor, edge_attr: OptTensor)
        out = self.propagate(edge_index, x=x, edge_attr=edge_attr, size=None, edge_weight=edge_weight)

        out = torch.cat([x, out], dim=-1)
        outs = [nn(out[:, i]) for i, nn in enumerate(self.post_nns)]
        out = torch.cat(outs, dim=1)

        return self.lin(out)

    def message(self, x_i: Tensor, x_j: Tensor,
                edge_attr: OptTensor, edge_weight: OptTensor) -> Tensor:

        h: Tensor = x_i  # Dummy.
        if edge_attr is not None:
            edge_attr = self.edge_encoder(edge_attr)
            edge_attr = edge_attr.view(-1, 1, self.F_in)
            edge_attr = edge_attr.repeat(1, self.towers, 1)
            h = torch.cat([x_i, x_j, edge_attr], dim=-1)
        else:
            h = torch.cat([x_i, x_j], dim=-1)

        hs = torch.stack([nn(h[:, i]) for i, nn in enumerate(self.pre_nns)], dim=1)
        return hs if edge_weight is None else hs * edge_weight[:, None, None]


class PNANet(torch.nn.Module):
    def __init__(self, train_dataset, input_dims, edge_features, hidden_dim, edge_dim, num_layer, num_class=1):
        super().__init__()

        max_degree = -1
        for data in train_dataset:
            if isinstance(data, list):
                g = data[0]
            else:
                g = data
            d = degree(g.edge_index[1], num_nodes=g.num_nodes, dtype=torch.long)
            max_degree = max(max_degree, int(d.max()))

        # Compute the in-degree histogram tensor
        deg = torch.zeros(max_degree + 1, dtype=torch.long)
        for data in train_dataset:
            if isinstance(data, list):
                g = data[0]
            else:
                g = data
            d = degree(g.edge_index[1], num_nodes=g.num_nodes, dtype=torch.long)
            deg += torch.bincount(d, minlength=deg.numel())

        self.node_emb = Linear(input_dims, hidden_dim)
        self.edge_emb = Linear(edge_features, edge_dim)

        aggregators = ['mean', 'min', 'max', 'std']
        scalers = ['identity', 'amplification', 'attenuation']

        self.convs = ModuleList()
        self.batch_norms = ModuleList()
        for _ in range(num_layer):
            conv = MyPNAConv(in_channels=hidden_dim, out_channels=hidden_dim,
                             aggregators=aggregators, scalers=scalers, deg=deg,
                             edge_dim=edge_dim, towers=4, pre_layers=1, post_layers=1,
                             divide_input=False)
            self.convs.append(conv)
            self.batch_norms.append(BatchNorm(hidden_dim))

        self.mlp = Sequential(Linear(hidden_dim, hidden_dim),
                              ReLU(),
                              Linear(hidden_dim, hidden_dim // 2),
                              ReLU(),
                              Linear(hidden_dim // 2, num_class))

    def forward(self, data):
        x, edge_index, edge_attr, edge_weight = data.x, data.edge_index, data.edge_attr, data.edge_weight
        batch = data.batch if hasattr(data, 'batch') and data.batch is not None \
            else torch.zeros(x.shape[0], dtype=torch.long)

        x = self.node_emb(x)
        edge_attr = self.edge_emb(edge_attr)

        for conv, batch_norm in zip(self.convs, self.batch_norms):
            x = F.relu(batch_norm(conv(x, edge_index, edge_attr, edge_weight)))

        if hasattr(data, 'selected_node_masks') and data.selected_node_masks is not None:
            if data.selected_node_masks.dtype == torch.float:
                x = x * data.selected_node_masks[:, None]
            elif data.selected_node_masks.dtype == torch.bool:
                x = x[data.selected_node_masks, :]
            else:
                raise ValueError

        x = global_add_pool(x, batch)

        x = self.mlp(x)
        if hasattr(data, 'inter_graph_idx') and data.inter_graph_idx is not None:
            x = global_mean_pool(x, data.inter_graph_idx)
        return x

    def reset_parameters(self):
        lst = torch.nn.ModuleList(self.mlp)
        for l in lst:
            if not isinstance(l, ReLU):
                l.reset_parameters()

        for c in self.convs:
            c.reset_parameters()

        for b in self.batch_norms:
            b.reset_parameters()

        self.node_emb.reset_parameters()
        self.edge_emb.reset_parameters()


class PNANet_order(torch.nn.Module):
    def __init__(self, train_dataset, input_dims, extra_in_dim,
                 extra_encode_dim, edge_features, hidden_dim, edge_dim, num_layer, num_class=1):
        super().__init__()

        max_degree = -1
        for data in train_dataset:
            if isinstance(data, list):
                g = data[0]
            else:
                g = data
            d = degree(g.edge_index[1], num_nodes=g.num_nodes, dtype=torch.long)
            max_degree = max(max_degree, int(d.max()))

        # Compute the in-degree histogram tensor
        deg = torch.zeros(max_degree + 1, dtype=torch.long)
        for data in train_dataset:
            if isinstance(data, list):
                g = data[0]
            else:
                g = data
            d = degree(g.edge_index[1], num_nodes=g.num_nodes, dtype=torch.long)
            deg += torch.bincount(d, minlength=deg.numel())

        self.node_emb = Linear(input_dims, hidden_dim)
        self.edge_emb = Linear(edge_features, edge_dim)
        self.extra_emb = Linear(extra_in_dim, extra_encode_dim)

        aggregators = ['mean', 'min', 'max', 'std']
        scalers = ['identity', 'amplification', 'attenuation']

        self.convs = ModuleList()
        self.batch_norms = ModuleList()
        for i in range(num_layer):
            conv = MyPNAConv(in_channels=hidden_dim if i > 0 else hidden_dim + extra_encode_dim, out_channels=hidden_dim,
                             aggregators=aggregators, scalers=scalers, deg=deg,
                             edge_dim=edge_dim, towers=4, pre_layers=1, post_layers=1,
                             divide_input=False)
            self.convs.append(conv)
            self.batch_norms.append(BatchNorm(hidden_dim))

        self.mlp = Sequential(Linear(hidden_dim, hidden_dim),
                              ReLU(),
                              Linear(hidden_dim, hidden_dim // 2),
                              ReLU(),
                              Linear(hidden_dim // 2, num_class))

    def forward(self, data):
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        extra_feature = data.extra_feature
        extra_feature = F.relu(self.extra_emb(extra_feature))
        x = self.node_emb(x)
        x = torch.cat([x, extra_feature], dim=-1)
        edge_attr = self.edge_emb(edge_attr)

        for conv, batch_norm in zip(self.convs, self.batch_norms):
            x = F.relu(batch_norm(conv(x, edge_index, edge_attr, None)))

        assert data.selected_node_masks.dtype == torch.float
        x = x * data.selected_node_masks[:, None]

        x = global_add_pool(x, data.original_node_mask)

        x = self.mlp(x)
        if hasattr(data, 'inter_graph_idx') and data.inter_graph_idx is not None:
            x = global_mean_pool(x, data.inter_graph_idx)
        return x

    def reset_parameters(self):
        lst = torch.nn.ModuleList(self.mlp)
        for l in lst:
            if not isinstance(l, ReLU):
                l.reset_parameters()

        for c in self.convs:
            c.reset_parameters()

        for b in self.batch_norms:
            b.reset_parameters()

        self.node_emb.reset_parameters()
        self.edge_emb.reset_parameters()
        self.extra_emb.reset_parameters()
