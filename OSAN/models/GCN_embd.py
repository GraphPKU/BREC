import torch
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import degree

from .encoder import AtomEncoder, BondEncoder


class GCNConv(MessagePassing):
    def __init__(self, edge_features, node_features, emb_dim, update_edge=False):
        super(GCNConv, self).__init__(aggr='add')

        self.update_edge = update_edge

        self.linear = torch.nn.Linear(node_features, emb_dim)
        self.edge_encoder = torch.nn.Linear(edge_features, emb_dim)
        if update_edge:
            self.edge_updating = torch.nn.Linear(emb_dim, emb_dim)

    def forward(self, x, edge_index, edge_embedding):
        x = self.linear(x)
        if edge_embedding is not None:
            edge_embedding = self.edge_encoder(edge_embedding)

        row, col = edge_index

        # edge_weight = torch.ones((edge_index.size(1), ), device=edge_index.device)
        deg = degree(row, x.size(0), dtype=x.dtype) + 1
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0

        norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]

        new_x = self.propagate(edge_index, x=x, edge_attr=edge_embedding, norm=norm)
        if self.update_edge:
            new_edge_attr = self.edge_updating(new_x[edge_index[0]] + new_x[edge_index[1]])
            if edge_embedding is not None:
                new_edge_attr += edge_embedding
        else:
            new_edge_attr = None
        return new_x, new_edge_attr

    def message(self, x_j, edge_attr, norm):
        return norm.view(-1, 1) * torch.relu(x_j + edge_attr) if edge_attr is not None else norm.view(-1, 1) * x_j

    def update(self, aggr_out):
        return aggr_out

    def reset_parameters(self):
        self.linear.reset_parameters()
        self.edge_encoder.reset_parameters()
        if self.update_edge:
            self.edge_updating.reset_parameters()


class GCN_emb(torch.nn.Module):
    def __init__(self, input_dim, edge_features, hid_dim, emb_dim, normalize=False, encoder=False):
        super(GCN_emb, self).__init__()

        self.encoder = encoder
        if encoder:
            self.atom_encoder = AtomEncoder(hid_dim)
            self.bond_encoder = BondEncoder(hid_dim)
            input_dim = hid_dim
            edge_features = hid_dim

        self.layers = torch.nn.ModuleList([])
        self.conv1 = GCNConv(edge_features, input_dim, hid_dim, update_edge=False)
        self.layers.append(self.conv1)
        self.bn1 = torch.nn.BatchNorm1d(hid_dim)
        self.layers.append(self.bn1)

        self.conv2 = GCNConv(edge_features, hid_dim, hid_dim, update_edge=False)
        self.layers.append(self.conv2)
        self.bn2 = torch.nn.BatchNorm1d(hid_dim)
        self.layers.append(self.bn2)

        self.conv3 = GCNConv(edge_features, hid_dim, hid_dim, update_edge=True)
        self.layers.append(self.conv3)
        self.bn3 = torch.nn.BatchNorm1d(hid_dim)
        self.layers.append(self.bn3)
        self.bn_edge3 = torch.nn.BatchNorm1d(hid_dim)
        self.layers.append(self.bn_edge3)

        self.lin_node = torch.nn.Linear(hid_dim, emb_dim)
        self.layers.append(self.lin_node)
        self.lin_edge = torch.nn.Linear(hid_dim, emb_dim)
        self.layers.append(self.lin_edge)
        self.normalize = normalize
        if normalize:
            self.node_bn = torch.nn.BatchNorm1d(emb_dim, affine=False)
            self.layers.append(self.node_bn)
            self.edge_bn = torch.nn.BatchNorm1d(emb_dim, affine=False)
            self.layers.append(self.edge_bn)

    def forward(self, data):
        x, edge_attr = data.x, data.edge_attr

        if self.encoder:
            x = self.atom_encoder(x)
            if edge_attr is not None:
                edge_attr = self.bond_encoder(edge_attr)

        x, _ = self.conv1(x, data.edge_index, edge_attr)
        x = self.bn1(torch.relu(x))
        x, _ = self.conv2(x, data.edge_index, edge_attr)
        x = self.bn2(torch.relu(x))
        x, edge_attr = self.conv3(x, data.edge_index, edge_attr)
        x = self.lin_node(self.bn3(torch.relu(x)))
        edge_attr = self.lin_edge(self.bn_edge3(torch.relu(edge_attr)))
        if self.normalize:
            x = self.node_bn(x)
            edge_attr = self.edge_bn(edge_attr)

        return x, edge_attr

    def reset_parameters(self):
        for l in self.layers:
            l.reset_parameters()
        if self.encoder:
            self.atom_encoder.reset_parameters()
            self.bond_encoder.reset_parameters()
