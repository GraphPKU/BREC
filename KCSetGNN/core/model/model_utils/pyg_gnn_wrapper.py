import torch
import torch.nn as nn
import torch_geometric.nn as gnn
from core.model.model_utils.elements import MLP
import torch.nn.functional as F
from torch_scatter import scatter
from core.model.model_utils.generalized_scatter import generalized_scatter

class GINEConv(nn.Module):
    def __init__(self, nin, nout, bias=True):
        super().__init__()
        self.nn = MLP(nin, nout, 2, False, bias=bias)
        self.layer = gnn.GINEConv(self.nn, train_eps=True)
        self.edge_linear = nn.Linear(nin, nin)
    def reset_parameters(self):
        self.nn.reset_parameters()
        self.layer.reset_parameters()
    def forward(self, x, edge_index, edge_attr, batch=None):
        edge_attr = self.edge_linear(edge_attr) # TODO: mark the update
        return self.layer(x, edge_index, edge_attr)
    
class SetConv(nn.Module):
    def __init__(self, nin, nout, bias=True):
        super().__init__()
        self.nn = MLP(nin, nout, 2, False)
        self.linear = nn.Linear(nin, nin, bias=False)
        self.bn = nn.BatchNorm1d(nin)
    def reset_parameters(self):
        self.nn.reset_parameters()
        self.bn.reset_parameters()
    def forward(self, x, edge_index, edge_attr, batch):
        # print(x.shape, batch.shape, batch.max())
        summation = scatter(x, batch, dim=0)
        summation = self.linear(summation)
        summation = self.bn(summation)
        summation = F.relu(summation)
        return self.nn(x + summation[batch])


class GATConv(nn.Module):
    def __init__(self, nin, nout, bias=True, nhead=1):
        super().__init__()
        self.layer = gnn.GATConv(nin, nout//nhead, nhead, bias=bias)
    def reset_parameters(self):
        self.layer.reset_parameters()
    def forward(self, x, edge_index, edge_attr, batch=None):
        return self.layer(x, edge_index)

class GCNConv(nn.Module):
    def __init__(self, nin, nout, bias=True):
        super().__init__()
        # self.nn = MLP(nin, nout, 2, False, bias=bias)
        # self.layer = gnn.GCNConv(nin, nin, bias=True)
        self.layer = gnn.GCNConv(nin, nout, bias=bias)
    def reset_parameters(self):
        self.layer.reset_parameters()
    def forward(self, x, edge_index, edge_attr, batch=None):
        return self.layer(x, edge_index)
        # return self.nn(F.relu(self.layer(x, edge_index)))

#TODO: define general message passing layers like DGN's paper, 
# or use the meta-layer
from torch_scatter import scatter
from torch_geometric.utils import degree
class SimplifiedPNAConv(gnn.MessagePassing):
    def __init__(self, nin, nout, bias=True, aggregators=['mean', 'min', 'max', 'std'], **kwargs): # ['mean', 'min', 'max', 'std'],
        kwargs.setdefault('aggr', None)
        super().__init__(node_dim=0, **kwargs)
        self.aggregators = aggregators
        self.pre_nn = MLP(3*nin, nin, 2, False)
        self.post_nn = MLP((len(aggregators) + 1 +1) * nin, nout, 2, False, bias=bias)
        # self.post_nn = MLP((len(aggregators) + 1 ) * nin, nout, 2, False)
        self.deg_embedder = nn.Embedding(200, nin) 

    def reset_parameters(self):
        self.pre_nn.reset_parameters()
        self.post_nn.reset_parameters()
        self.deg_embedder.reset_parameters()

    def forward(self, x, edge_index, edge_attr, batch=None):
        out = self.propagate(edge_index, x=x, edge_attr=edge_attr)
        out = torch.cat([x, out], dim=-1)
        out = self.post_nn(out)
        # return x + out
        return out

    def message(self, x_i, x_j, edge_attr):
        if edge_attr is not None:
            h = torch.cat([x_i, x_j, edge_attr], dim=-1)
        else:
            h = torch.cat([x_i, x_j], dim=-1)
        return self.pre_nn(h)

    def aggregate(self, inputs, index, dim_size=None):
        outs = []
        for aggregator in self.aggregators:
            if aggregator == 'sum':
                out = scatter(inputs, index, 0, None, dim_size, reduce='sum')
            elif aggregator == 'mean':
                out = scatter(inputs, index, 0, None, dim_size, reduce='mean')
            elif aggregator == 'min':
                out = scatter(inputs, index, 0, None, dim_size, reduce='min')
            elif aggregator == 'max':
                out = scatter(inputs, index, 0, None, dim_size, reduce='max')
            elif aggregator == 'var' or aggregator == 'std':
                mean = scatter(inputs, index, 0, None, dim_size, reduce='mean')
                mean_squares = scatter(inputs * inputs, index, 0, None, dim_size, reduce='mean')
                out = mean_squares - mean * mean
                if aggregator == 'std':
                    out = torch.sqrt(F.relu_(out) + 1e-5)
            else:
                raise ValueError(f'Unknown aggregator "{aggregator}".')  
            outs.append(out)

        outs.append(self.deg_embedder(degree(index, dim_size, dtype=index.dtype)))
        out = torch.cat(outs, dim=-1)

        return out


# Next test PNA without any normalization layer

class GINEDegConv(gnn.MessagePassing):
    def __init__(self, nin, nout, bias=True, **kwargs):
        kwargs.setdefault('aggr', None)
        super().__init__(node_dim=0, **kwargs)
        self.nn = MLP(2*nin, nout, 2, False, bias=bias)
        self.eps = torch.nn.Parameter(torch.Tensor([0]))
        self.deg_embedder = nn.Embedding(200, nin) 

        # self.nn = MLP(nin, nout, 2, False, bias=bias)

        self.nn = MLP(3*nin, nout, 2, False, bias=bias)

        self.linear1 = nn.Linear(nin, nin) 
        self.linear2 = nn.Linear(nin, nin) 
        # self.nn = MLP(nin, nout, 2, False, bias=bias)


        # self.attention = MultiHeadAttention(2*nin, nin)
        self.nn = MLP(nin, nout, 2, False, bias=bias)

    def reset_parameters(self):
        self.nn.reset_parameters()
        self.eps.data.fill_(0)
        self.deg_embedder.reset_parameters()

    def forward(self, x, edge_index, edge_attr, batch=None):
        out, deg = self.propagate(edge_index, x=x, edge_attr=edge_attr)
        out = out + (1+self.eps) * x
        # out = torch.cat([out,deg], dim=-1)
        # out = torch.cat([out, deg, out*deg], dim=-1)
        
        out = out + deg + self.linear1(out) * self.linear2(deg)

        # return out 

        # out, deg = self.propagate(edge_index, x=x, edge_attr=edge_attr)
        # out += (1+self.eps) * x


        return self.nn(out)
    
    def message(self, x_j, edge_attr):
        return (x_j + edge_attr).relu()

    def aggregate(self, inputs, index, dim_size=None):
        out = scatter(inputs, index, 0, None, dim_size, reduce='mean')
        deg = self.deg_embedder(degree(index, dim_size, dtype=index.dtype))

        # out = scatter(inputs, index, 0, None, dim_size, reduce='add')
        return out, deg