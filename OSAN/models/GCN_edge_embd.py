import torch
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import degree

from .encoder import AtomEncoder, BondEncoder


class GCNConv(MessagePassing):
    def __init__(self, edge_features, node_features, emb_dim):
        super(GCNConv, self).__init__(aggr='add')
        self.linear = torch.nn.Linear(node_features, emb_dim)
        self.edge_encoder = torch.nn.Linear(edge_features, emb_dim)

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

        return new_x

    def message(self, x_j, edge_attr, norm):
        return norm.view(-1, 1) * torch.relu(x_j + edge_attr) if edge_attr is not None else norm.view(-1, 1) * x_j

    def update(self, aggr_out):
        return aggr_out

    def reset_parameters(self):
        self.linear.reset_parameters()
        self.edge_encoder.reset_parameters()


class GCN_edge_emb(torch.nn.Module):
    def __init__(self, input_dim, edge_features, hid_dim, emb_dim, normalize=False, encoder=False):
        super(GCN_edge_emb, self).__init__()

        if encoder:
            raise NotImplementedError

        self.encoder = encoder
        if encoder:
            self.atom_encoder = AtomEncoder(hid_dim)
            self.bond_encoder = BondEncoder(hid_dim)
            input_dim = hid_dim
            edge_features = hid_dim

        self.layers = torch.nn.ModuleList([])
        self.conv1 = GCNConv(edge_features, input_dim, hid_dim)
        self.layers.append(self.conv1)
        self.bn1 = torch.nn.BatchNorm1d(hid_dim)
        self.layers.append(self.bn1)

        self.conv2 = GCNConv(edge_features, hid_dim, hid_dim)
        self.layers.append(self.conv2)
        self.bn2 = torch.nn.BatchNorm1d(hid_dim)
        self.layers.append(self.bn2)

        self.conv3 = GCNConv(edge_features, hid_dim, hid_dim)
        self.layers.append(self.conv3)
        self.bn3 = torch.nn.BatchNorm1d(hid_dim)
        self.layers.append(self.bn3)

        self.lin_node = torch.nn.Linear(hid_dim, emb_dim)
        self.layers.append(self.lin_node)
        self.normalize = normalize
        if normalize:
            self.node_bn = torch.nn.BatchNorm1d(emb_dim, affine=False)
            self.layers.append(self.node_bn)

    def forward(self, data):
        x, edge_attr, edge_index = data.lin_x, data.lin_edge_attr, data.lin_edge_index

        if self.encoder:
            x = self.atom_encoder(x)
            if edge_attr is not None:
                edge_attr = self.bond_encoder(edge_attr)

        x = self.conv1(x, edge_index, edge_attr)
        x = self.bn1(torch.relu(x))
        x = self.conv2(x, edge_index, edge_attr)
        x = self.bn2(torch.relu(x))
        x = self.conv3(x, edge_index, edge_attr)
        x = self.lin_node(self.bn3(torch.relu(x)))
        if self.normalize:
            x = self.node_bn(x)

        return None, x

    def reset_parameters(self):
        for l in self.layers:
            l.reset_parameters()
        if self.encoder:
            self.atom_encoder.reset_parameters()
            self.bond_encoder.reset_parameters()
