import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_scatter import scatter
from core.model.model_utils.generalized_scatter import generalized_scatter
import core.model.model_utils.pyg_gnn_wrapper as gnn_wrapper 
from core.model.model_utils.elements import MLP, DiscreteEncoder, Identity, BN
from core.model.model_utils.ppgn import PPGNLayer
from torch_geometric.nn.inits import reset
from core.model.bipartite import BipartiteGNN

class KCSetGNN(nn.Module):
    def __init__(self, nfeat_node, nfeat_edge, nhid, nout, 
                    nlayer_intra, nlayer_inter, gnn_type, bgnn_type,
                    dropout=0, 
                    res=True, 
                    pools=['add'], # notice that pools now is only used for Bipartite Propagation. All others are fixed.
                    mlp_layers=2,
                    num_bipartites=1,
                    half_step=False):
        super().__init__()
        encoding_dim = 16
        # encode node feature and edge feature
        self.input_encoder = DiscreteEncoder(nhid) if nfeat_node is None else MLP(nfeat_node, nhid, 1)
        self.edge_encoder = DiscreteEncoder(nhid) if nfeat_edge is None else MLP(nfeat_edge, nhid, 1)
        self.pos_encoder = MLP(3, nhid, 1)

        # encode subgraphs 
        self.subgraph_encoder = GraphToSubgraphs(nhid, nlayer_intra, mlp_layers, gnn_type, dropout)

        # propagate among subgraphs
        self.bipartite_gnn = BipartiteGNN(nhid, nhid, 
                                          nlayer_inter, 
                                          mlp_layers=mlp_layers, 
                                          type=bgnn_type, 
                                          dropout=dropout, 
                                          bias=True, 
                                          pools=pools, 
                                          num_bipartites=num_bipartites, 
                                          half_step=half_step)
        # output
        self.output_decoder0 = MLP(2*nhid, nhid, nlayer=1, with_final_activation=True) # MarK 2-> 1
        self.output_decoder = MLP((num_bipartites+1)*(nhid), nout, nlayer=2, with_final_activation=False, n_hid=nhid)
        
        self.num_bipartites = num_bipartites
        self.pools = pools

        self.num_components_encoder = DiscreteEncoder(encoding_dim, max_num_values=20)
        self.num_nodes_encoder = DiscreteEncoder(encoding_dim, max_num_values=20)
        self.kc_gate = nn.Sequential(nn.Linear(2*encoding_dim, nhid, bias=False), nn.Sigmoid())
                                         
    def reset_parameters(self):
        self.input_encoder.reset_parameters()
        self.edge_encoder.reset_parameters()
        self.pos_encoder.reset_parameters()
        self.subgraph_encoder.reset_parameters()
        self.bipartite_gnn.reset_parameters()
        self.output_decoder.reset_parameters()
        self.output_decoder0.reset_parameters()
        self.num_components_encoder.init_constant()
        self.num_nodes_encoder.init_constant()
        reset(self.kc_gate)

    def forward(self, data):
        # generate data.ks and data.num_subgraphs from data.num_ks
        num_ks = data.num_ks.reshape(data.num_graphs, self.num_bipartites+1)
        data.num_subgraphs = num_ks.sum(dim=1)
        data.ks = torch.repeat_interleave(torch.arange(self.num_bipartites+1, device=data.x.device).repeat(data.num_graphs), data.num_ks)

        # encode x and edges
        x = data.x if len(data.x.shape) <= 2 else data.x.squeeze(-1)
        x = self.input_encoder(x)
        
        edge_attr = data.edge_index.new_zeros(data.edge_index.size(-1), 1) if data.edge_attr is None else data.edge_attr 
        edge_attr = self.edge_encoder(edge_attr)
        # encode 3d position into edge_attr 
        if data.pos is not None:
            dis = data.pos[data.edge_index[0]] - data.pos[data.edge_index[1]]
            edge_attr = edge_attr + self.pos_encoder(dis.abs())
            
        # get init color of k-sets
        x, subgraphs = self.subgraph_encoder(x, edge_attr, data)
    
        # message passing among k-set graphs.
        subgraphs = self.bipartite_gnn(subgraphs, data.ks, [getattr(data, f'bipartite_{i}') for i in range(data.ks.max())], x)
        
        # generate graph embedding based on all k-sets. 
        num_graphs = data.batch[-1]+1
        batch_subgraphs = torch.arange(num_graphs, device=x.device)
        batch_subgraphs = batch_subgraphs.repeat_interleave(data.num_subgraphs)
        # aggregate to graph level representation 
        kmax = self.num_bipartites + 1
        within_graph_kbatch = data.ks + batch_subgraphs * kmax

        # gate mechanism, to balance scale by learning
        kc_encoding = torch.cat([self.num_nodes_encoder(data.ks), self.num_components_encoder(data.num_components)], dim=-1)
        max_info = scatter(subgraphs, within_graph_kbatch, dim=0, dim_size=num_graphs*kmax, reduce='max')
        sum_info = scatter(subgraphs * self.kc_gate(kc_encoding), within_graph_kbatch, dim=0, dim_size=num_graphs*kmax, reduce='add')

        x = self.output_decoder0(torch.cat([max_info, sum_info], -1)).reshape(num_graphs, -1) # num_graphs x (kmax*nhid)
        return self.output_decoder(x) 

class BaseGNN(nn.Module):
    # this version use nin as hidden instead of nout, resulting a larger model
    def __init__(self, nin, nout, nlayer, gnn_type, mlp_layers=1, dropout=0, bn=BN, bias=True, res=True):
        super().__init__()
        self.convs = nn.ModuleList([getattr(gnn_wrapper, gnn_type)(nin, nin, bias=not bn) for _ in range(nlayer)]) # set bias=False for BN
        self.norms = nn.ModuleList([nn.BatchNorm1d(nin, momentum=1.0, affine=False) if bn else Identity() for _ in range(nlayer)])
        self.output_encoder = MLP(nin, nout, nlayer=1, with_final_activation=True, bias=bias) #if nin!=nout else Identity() 
        self.dropout = dropout
        self.res = res

    def reset_parameters(self):
        self.output_encoder.reset_parameters()
        for conv, norm in zip(self.convs, self.norms):
            conv.reset_parameters()
            norm.reset_parameters()
     
    def forward(self, x, edge_index, edge_attr, batch):
        previous_x = x
        for layer, norm in zip(self.convs, self.norms):
            x = layer(x, edge_index, edge_attr, batch)
            # TODO: update edge_attr ?
            x = norm(x)
            x = F.relu(x)
            x = F.dropout(x, self.dropout, training=self.training)
            if self.res:
                x = x + previous_x 
                previous_x = x

        x = self.output_encoder(x)
        return x

class GraphToSubgraphs(nn.Module):
    def __init__(self, nhid, nlayer, mlp_layers, gnn_type, dropout=0):
        super().__init__()
        encoding_dim = 16
        encoding_dim = 0
        if gnn_type == 'PPGNConv':
            self.subgraphs_conv = PPGNLayer(nhid, nhid, nlayer=nlayer, depth_of_mlp=mlp_layers)
        else:
            self.subgraphs_conv = BaseGNN(nhid, nhid, nlayer, gnn_type, mlp_layers=mlp_layers, dropout=dropout, res=True)

        self.num_components_encoder = DiscreteEncoder(encoding_dim if encoding_dim>0 else nhid, max_num_values=20)
        self.num_nodes_encoder = DiscreteEncoder(encoding_dim if encoding_dim>0 else nhid, max_num_values=20)
        self.out_encoder =  MLP(nhid+2*encoding_dim, nhid, nlayer=1, with_final_activation=True)

        self.dropout = dropout
        self.encoding_dim = encoding_dim

    def reset_parameters(self):
        self.subgraphs_conv.reset_parameters()
        self.num_components_encoder.reset_parameters()
        self.num_nodes_encoder.reset_parameters()
        self.out_encoder.reset_parameters()

    def forward(self, x, edge_attr, data):
        ### graph_x => subgraphs_x
        subgraphs_x = x[data.subgraphs_nodes_mapper] # lift up the embeddings, positional encoding
        # else: # use previous subgraphs x
        #     subgraphs_x = self.combine_subgraphs_x(torch.cat([subgraphs_x, x[data.subgraphs_nodes_mapper]], dim=1))

        subgraphs_edge_index = data.combined_subgraphs
        subgraphs_edge_attr = edge_attr[data.subgraphs_edges_mapper]
        subgraphs_batch = data.subgraphs_batch

        # subgraphs_x <=> subgraphs_x
        subgraphs_x = self.subgraphs_conv(subgraphs_x, subgraphs_edge_index, subgraphs_edge_attr, subgraphs_batch)

        # # subgraphs_x => x 
        # context_x = scatter(self.pre_scatter(subgraphs_x), data.subgraphs_nodes_mapper, dim=0, dim_size=x.size(0), reduce=self.pooling)
        # x = self.combine_x(torch.cat([x, self.pre_combine_x(context_x)], dim=1))

        # subgraphs_x => subgraphs
        subgraphs = scatter(subgraphs_x, subgraphs_batch, dim=0, reduce='add')

        # deal with sets with >1 number of components [TODO: currently only support add pooling]
        full_subgraphs = subgraphs.new_zeros((len(data.num_components), subgraphs.size(-1)))
        full_subgraphs[data.num_components==1] = subgraphs
        
        if hasattr(data, 'components_graph'):
            # inference sets with multiple components
            # TODO: the current version is simple add which mimicing message passing GNNs, need to test more powerful set encoder
            scatter(full_subgraphs[data.components_graph[0]], data.components_graph[1], dim=0, reduce='add', out=full_subgraphs) 

        # if data.ks.max() > 1:
        #     # only do it for large number of ks. To avoid do this for no stacking setting.
        full_subgraphs = full_subgraphs / (data.ks+1).float().unsqueeze(1)
        if self.encoding_dim ==0:
            subgraphs = full_subgraphs + self.num_nodes_encoder(data.ks) + self.num_components_encoder(data.num_components)
        else:
            subgraphs = torch.cat([full_subgraphs, self.num_nodes_encoder(data.ks), self.num_components_encoder(data.num_components)], -1)

        subgraphs = self.out_encoder(subgraphs) # this maybe not necessary
        return x, subgraphs