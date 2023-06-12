import torch
from torch.nn import Linear, ReLU
from torch_geometric.nn import MessagePassing


class GINEConv(MessagePassing):
    def __init__(self, emb_dim, dim1, dim2, update_edge=False):
        super(GINEConv, self).__init__(aggr="add")

        # disable the bias, otherwise the information will be nonzero
        self.bond_encoder = torch.nn.ModuleList([Linear(emb_dim, dim1), ReLU(), Linear(dim1, dim1)])
        self.mlp = torch.nn.ModuleList([Linear(dim1, dim1), ReLU(), Linear(dim1, dim2)])
        self.eps = torch.nn.Parameter(torch.Tensor([0]))

    def forward(self, x, edge_index, edge_attr):
        edge_embedding = edge_attr
        for l in self.bond_encoder:
            edge_embedding = l(edge_embedding)
        out = (1 + self.eps) * x + self.propagate(edge_index, x=x, edge_attr=edge_embedding)
        for l in self.mlp:
            out = l(out)

        return out

    def message(self, x_j, edge_attr):
        # x_j has shape [E, out_channels]
        return torch.relu(x_j + edge_attr)

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


class GINE_embd(torch.nn.Module):
    def __init__(self, input_dims,
                 edge_features,
                 dim,
                 num_convlayers=4,
                 num_class=1):
        super(GINE_embd, self).__init__()

        assert num_convlayers > 1

        self.conv = torch.nn.ModuleList([GINEConv(edge_features, input_dims, dim)])
        self.bn = torch.nn.ModuleList([torch.nn.BatchNorm1d(dim)])

        for _ in range(num_convlayers - 1):
            self.conv.append(GINEConv(edge_features, dim, dim))
            self.bn.append(torch.nn.BatchNorm1d(dim))

        self.fc1 = Linear(dim, dim)
        self.bn_last = torch.nn.BatchNorm1d(dim)
        self.fc2 = Linear(dim, num_class)

    def forward(self, data):
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr

        for i, (conv, bn) in enumerate(zip(self.conv, self.bn)):
            x = conv(x, edge_index, edge_attr)
            x = bn(x)
            x = torch.relu(x)

        x = torch.relu(self.bn_last(self.fc1(x)))
        x = self.fc2(x)
        return x, None

    def reset_parameters(self):
        for l in self.conv:
            l.reset_parameters()
        for l in self.bn:
            l.reset_parameters()
        self.fc1.reset_parameters()
        self.fc2.reset_parameters()
        self.bn_last.reset_parameters()
