import torch
import torch.nn as nn
import torch.nn.functional as F
from core.model.model_utils.elements import MLP, DiscreteEncoder, Identity, BN
from torch_scatter import scatter
from core.model.model_utils.pyg_gnn_wrapper import GINEConv
import core.model.model_utils.pyg_gnn_wrapper as gnn_wrapper 


class GNN(nn.Module):
    # this version use nin as hidden instead of nout, resulting a larger model
    def __init__(self, nfeat_node, nfeat_edge, nhid, nout, nlayer, gnn_type, dropout=0, pooling='add', bn=BN, res=True):
        super().__init__()
        d2c_dim=0
        self.input_encoder = DiscreteEncoder(nhid-d2c_dim) if nfeat_node is None else MLP(nfeat_node, nhid-d2c_dim, 1)
        self.edge_encoder = DiscreteEncoder(nhid) if nfeat_edge is None else MLP(nfeat_edge, nhid, 1)
        self.pos_encoder = MLP(3, nhid, 1)
        self.convs = nn.ModuleList([getattr(gnn_wrapper, gnn_type)(nhid, nhid, bias=not bn) for _ in range(nlayer)]) # set bias=False for BN
        self.norms = nn.ModuleList([nn.BatchNorm1d(nhid) if bn else Identity() for _ in range(nlayer)])
        self.output_encoder = MLP(nhid, nout, nlayer=2, with_final_activation=False, with_norm=False if pooling=='mean' else True)
        self.pooling = pooling
        self.dropout = dropout
        self.res = res

    def reset_parameters(self):
        self.input_encoder.reset_parameters()
        self.edge_encoder.reset_parameters()
        self.pos_encoder.reset_parameters()
        self.output_encoder.reset_parameters()
        for conv, norm in zip(self.convs, self.norms):
            conv.reset_parameters()
            norm.reset_parameters()
     
    def forward(self, data):
        # encode x and edges
        x = data.x if len(data.x.shape) <= 2 else data.x.squeeze(-1)
        x = self.input_encoder(x)
        edge_attr = data.edge_index.new_zeros(data.edge_index.size(-1)) if data.edge_attr is None else data.edge_attr 
        edge_attr = self.edge_encoder(edge_attr)
        # encode 3d position 
        if data.pos is not None:
            dis = data.pos[data.edge_index[0]] - data.pos[data.edge_index[1]]
            edge_attr = edge_attr + self.pos_encoder(dis.abs())

        previous_x = 0#x
        for layer, norm in zip(self.convs, self.norms):
            x = layer(x, data.edge_index, edge_attr, data.batch)
            x = norm(x)
            x = F.relu(x)
            x = F.dropout(x, self.dropout, training=self.training)
            if self.res:
                x = x + previous_x 
                previous_x = x

        x = scatter(x, data.batch, dim=0, reduce='add')
        x = self.output_encoder(x)
        return x



# Step 1: implement a layer that (x,edge_attr) -> (x,edge_attr)
class EdgeUpdater(nn.Module):
    def __init__(self, nin, nout, mlp_layers=2):
        super().__init__()
        self.linear = nn.Linear(nin, nin)
        self.combiner = MLP(2*nin, nout, nlayer=mlp_layers, with_final_activation=True)

    def reset_parameters(self):
        self.linear.reset_parameters()
        self.combiner.reset_parameters()
        
    def forward(self, x, edge_index, edge_attr):
        x = self.linear(x)
        aggregated = x[edge_index[0]] + x[edge_index[1]]
        return self.combiner(torch.cat([aggregated, edge_attr], dim=-1))


class EdgeConv(nn.Module):
    def __init__(self, nhid, mlp_layers=2):
        super().__init__()
        self.node_conv = GINEConv(nhid, nhid)
        self.edge_conv = EdgeUpdater(nhid, nhid, mlp_layers)
        self.norm = nn.BatchNorm1d(nhid)

    def reset_parameters(self):
        self.node_conv.reset_parameters()
        self.edge_conv.reset_parameters()
        self.norm.reset_parameters()

    def forward(self, x, edge_index, edge_attr):
        # from nodes to edges
        edge_attr = self.edge_conv(x, edge_index, edge_attr)
        # from edges to nodes 
        x = self.node_conv(x, edge_index, edge_attr)
        x = self.norm(x)
        x = F.relu(x)
        # edge_attr = self.edge_conv(x, edge_index, edge_attr)
        return x, edge_attr


class EdgeGNN(nn.Module):
    # this version use nin as hidden instead of nout, resulting a larger model
    def __init__(self, nfeat_node, nfeat_edge, nhid, nout, nlayer, dropout=0, pooling='add', res=True):
        super().__init__()
        self.input_encoder = DiscreteEncoder(nhid) if nfeat_node is None else MLP(nfeat_node, nhid, 1)
        self.edge_encoder = DiscreteEncoder(nhid) if nfeat_edge is None else MLP(nfeat_edge, nhid, 1)
        self.convs = nn.ModuleList([EdgeConv(nhid) for _ in range(nlayer)]) 
        self.output_encoder1 = MLP(nhid, nhid, nlayer=1, with_final_activation=False, with_norm=True)
        self.output_encoder2 = MLP(nhid, nhid, nlayer=1, with_final_activation=True, with_norm=True)
        self.output_encoder = MLP(2*nhid, nout, nlayer=2, with_final_activation=True, with_norm=True)

        self.pooling = pooling
        self.dropout = dropout
        self.res = res

    def reset_parameters(self):
        self.input_encoder.reset_parameters()
        self.edge_encoder.reset_parameters()
        self.output_encoder1.reset_parameters()
        self.output_encoder2.reset_parameters()
        self.output_encoder.reset_parameters()
        for conv in self.convs:
            conv.reset_parameters()
     
    def forward(self, data):
        # encode x and edges
        x = data.x if len(data.x.shape) <= 2 else data.x.squeeze(-1)
        x = self.input_encoder(x)
        edge_attr = data.edge_index.new_zeros(data.edge_index.size(-1)) if data.edge_attr is None else data.edge_attr 
        edge_attr = self.edge_encoder(edge_attr)

        px, pe = 0, 0
        for layer in self.convs:
            x, edge_attr = layer(x, data.edge_index, edge_attr)
            x = F.dropout(x, self.dropout, training=self.training)
            edge_attr = F.dropout(edge_attr, self.dropout, training=self.training)
            if self.res:
                x = x + px
                # edge_attr += pe 
                px, pe = x, edge_attr

        # aggregate to graph level
        x = scatter(x, data.batch, dim=0, reduce=self.pooling)
        edge_batch = data.batch[data.edge_index[0]]
        edge = scatter(edge_attr, edge_batch, dim=0, reduce=self.pooling)
        
        x = self.output_encoder(torch.cat([self.output_encoder1(x), self.output_encoder2(edge)], dim=1))
        return x