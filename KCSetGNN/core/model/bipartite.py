"""
    Focus on bipartite graph convolution.
    Assume every part has node features. 

    Bidirectional propagation? Because it actually has order. 
"""
import torch
import torch.nn as nn 
import torch.nn.functional as F
from core.model.model_utils.elements import MLP, Identity
from torch_scatter import scatter
from core.model.model_utils.generalized_scatter import generalized_scatter
from torch_geometric.nn.inits import reset

class BipartiteGNN(nn.Module):
    def __init__(self, nin, nout, nlayer, type, mlp_layers=2, dropout=0, bias=True, pools=['add'], num_bipartites=1, half_step=False):
        super().__init__()
        assert type in ['Sequential', 'Parallel']
        self.type = type 
        self.dropout = dropout
        Layer = SequentialLayer if type == 'Sequential' else ParallelLayer
        self.layers = nn.ModuleList(Layer(nin, mlp_num_layers=mlp_layers, pools=pools, num_bipartites=num_bipartites, half_step=half_step)  for _ in range(nlayer))
        self.output_encoder = MLP(nin, nout, nlayer=1, with_final_activation=True, bias=bias) if nin!=nout else Identity() 

    def reset_parameters(self):
        for layer in self.layers:
            layer.reset_parameters()
        self.output_encoder.reset_parameters()

    def forward(self, xs, k_batch, bipartites_list, x):
        n = len(bipartites_list)
        if self.type == 'Parallel':
            combined_xs = torch.cat([xs[k_batch==i] for i in range(n+1)])
            combined_batch = torch.cat([k_batch[k_batch==i] for i in range(n+1)])
            combined_bipartites = torch.clone(bipartites_list[0])
            for k_to_kplus1 in bipartites_list[1:]:
                # update = combined_bipartites.max(dim=1, keepdim=True)[0] + k_to_kplus1 + 1
                update = torch.clone(k_to_kplus1)
                update[:2] =  update[:2] + 1 + combined_bipartites[:2].max(dim=1, keepdim=True)[0] #TODO: this is incorrect for empty case
                combined_bipartites = torch.cat([combined_bipartites, update], dim=1) 
        else:
            combined_xs, combined_batch, combined_bipartites = xs, k_batch, bipartites_list

        for layer in self.layers:
            combined_xs = combined_xs + F.dropout(layer(combined_xs, combined_batch, combined_bipartites, x), self.dropout, training=self.training)

        if self.type == 'Parallel':
            xs = combined_xs.clone()
            for i in range(n + 1):
                xs[k_batch==i] = combined_xs[combined_batch==i]
        else:
            xs = combined_xs

        return self.output_encoder(xs)

class ParallelLayer(nn.Module):
    def __init__(self, nhid, mlp_num_layers=2, pools=['add'], num_bipartites=1,  half_step=False):
        super().__init__()
        self.pools = pools 
        self.half_step = half_step 
        self.nn1 = MLP(nhid, nhid, mlp_num_layers, with_final_activation=True) 
        self.nn2 = MLP(len(pools)*nhid, nhid, mlp_num_layers, with_final_activation=True) 
        self.linear1 = nn.Sequential(nn.Linear(nhid, nhid//2), nn.BatchNorm1d(nhid//2, momentum=1.0, affine=False), nn.ReLU())
        self.linear2 = nn.Sequential(nn.Linear(len(pools)*nhid, nhid//2), nn.BatchNorm1d(nhid//2, momentum=1.0, affine=False), nn.ReLU())
        self.linear3 = nn.Sequential(nn.Linear(len(pools)*nhid, nhid//2), nn.BatchNorm1d(nhid//2, momentum=1.0, affine=False), nn.ReLU())
        self.linear4 = nn.Sequential(nn.Linear(len(pools)*nhid, nhid//2), nn.BatchNorm1d(nhid//2, momentum=1.0, affine=False), nn.ReLU())
        self.nn_out = MLP(2*nhid, nhid, mlp_num_layers, with_final_activation=True) 

    def reset_parameters(self):
        self.nn1.reset_parameters()
        self.nn2.reset_parameters()
        self.nn_out.reset_parameters()
        reset(self.linear1)
        reset(self.linear2)
        reset(self.linear3)
        reset(self.linear4)

    def forward(self, combined_xs, combined_batch, combined_bipartities, x=None):
        last = combined_batch.max().item()
        x_right = combined_xs[combined_batch != 0]
        x_left = combined_xs[combined_batch != last]

        # right to left
        message = x_right[combined_bipartities[1]]
        # right_info = scatter(message, combined_bipartities[0], dim=0, dim_size=x_left.size(0), reduce=self.pooling)
        right_info = generalized_scatter(message, combined_bipartities[0], dim_size=x_left.size(0), aggregators=self.pools) # len(pools) * nhid

        # left to right
        ## apply nn to get new x_left
        x_left_new = self.nn1(combined_xs)[combined_batch != last]
        message = x_left_new[combined_bipartities[0]]
        # left_info = scatter(message, combined_bipartities[1], dim=0, dim_size=x_right.size(0), reduce=self.pooling)
        left_info = generalized_scatter(message, combined_bipartities[1], dim_size=x_right.size(0), aggregators=self.pools) # len(pools) * nhid
        
        # right_info to right
        ## apply nn to get new right_info
        right_info_new = self.nn2(right_info)
        message =  right_info_new[combined_bipartities[0]]
        # right_left_info = scatter(message, combined_bipartities[1], dim=0, dim_size=x_right.size(0), reduce=self.pooling)
        right_left_info = generalized_scatter(message, combined_bipartities[1], dim_size=x_right.size(0), aggregators=self.pools) # len(pools) * nhid
        
        ## apply nn to transform  x, right_info, left_info, right_left_info to low dim first
        x = self.linear1(combined_xs)
        right_info_tmp = self.linear2(right_info)
        left_info_tmp = self.linear3(left_info)
        right_left_info_tmp = self.linear4(right_left_info)

        right_info = torch.clone(x)
        left_info = torch.clone(x)
        right_left_info = torch.clone(x)
        right_info[combined_batch != last] = right_info_tmp
        left_info[combined_batch != 0] = left_info_tmp
        right_left_info[combined_batch != 0] = right_left_info_tmp

        # combine and transform 
        return self.nn_out(torch.cat([x, right_info, left_info, right_left_info], -1))

class SequentialLayer(nn.Module):
    def __init__(self, nhid, mlp_num_layers=2, pools=['add'], num_bipartites=1, half_step=False):
        super().__init__()
        # share MLP to save number of paramaters (like RNN) 
        # later can also test don't share parameters
        self.pools = pools
        self.combine1 = nn.ModuleList(SelfCombiner(nhid, mlp_num_layers, second_nhid=len(pools)*nhid) for _ in range(num_bipartites))
        self.combine2 = nn.ModuleList(SelfCombiner(nhid, mlp_num_layers, second_nhid=len(pools)*nhid) for _ in range(num_bipartites))    
        self.half_step = half_step
    def reset_parameters(self):
        for c1,c2 in zip(self.combine1, self.combine2):
            c1.reset_parameters()
            c2.reset_parameters()

    def forward(self, xs, k_batch, bipartites_list, x=None): 
        # Backfward first: from right to left 
        xs_out = xs.clone()
        x_right = xs[k_batch == len(bipartites_list)]
        for i in reversed(range(len(bipartites_list))):
            edge_index = bipartites_list[i]
            x_left = xs[k_batch == i]
            message = x_right[edge_index[1]]
            aggregated = generalized_scatter(message, edge_index[0], dim=0, dim_size=x_left.size(0), aggregators=self.pools)
            x_right = self.combine1[i](x_left, aggregated)
            xs_out[k_batch == i] = x_right

        if not self.half_step:
            # Then foward: from left to right
            xs = xs_out.clone()
            x_left = xs[k_batch == 0]
            for i in range(len(bipartites_list)):
                edge_index = bipartites_list[i]
                x_right = xs[k_batch == i+1]
                message = x_left[edge_index[0]]
                # aggregated = scatter(message, edge_index[1], dim=0, dim_size=x_right.size(0), reduce=self.pooling)
                aggregated = generalized_scatter(message, edge_index[1], dim=0, dim_size=x_right.size(0), aggregators=self.pools)
                x_left = self.combine2[i](x_right, aggregated)
                xs_out[k_batch == i+1] = x_left
        return xs_out


class SelfCombiner(nn.Module):
    def __init__(self, nhid, mlp_num_layers=2, type='2-way', second_nhid=None):
        super().__init__()
        assert type in ['2-way', '3-way', '4-way']
        if second_nhid is None:
            second_nhid = nhid 

        self.nn1 = MLP(second_nhid, nhid, mlp_num_layers, with_final_activation=True)
        if type == '4-way':
            self.nn2 = MLP(second_nhid, nhid, mlp_num_layers, with_final_activation=True)
            self.nn3 = MLP(second_nhid, nhid, mlp_num_layers, with_final_activation=True)
            self.combine = MLP(4*nhid, nhid, 1, with_final_activation=True)
        elif type == '3-way':
            self.nn2 = MLP(second_nhid, nhid, mlp_num_layers, with_final_activation=True)
            self.combine = MLP(3*nhid, nhid, 1, with_final_activation=True)
        else:
            self.combine = MLP(2*nhid, nhid, 1, with_final_activation=True)
        self.type = type

    def reset_parameters(self):
        self.nn1.reset_parameters()
        self.combine.reset_parameters()
        if self.type == '3-way':
            self.nn2.reset_parameters()
        if self.type == '4-way':
            self.nn2.reset_parameters()
            self.nn3.reset_parameters()

    def forward(self, x, y, z=None, r=None):
        if self.type == '4-way':
            out = torch.cat([x, self.nn1(y), self.nn2(z), self.nn3(r)], dim=-1)
        elif self.type == '3-way':
            out = torch.cat([x, self.nn1(y), self.nn2(z)], dim=-1)
        else:
            out = torch.cat([x, self.nn1(y)], dim=-1)
        out = self.combine(out)
        return out