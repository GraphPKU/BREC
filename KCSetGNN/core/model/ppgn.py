import torch.nn as nn
from torch_scatter import scatter
from core.model.model_utils.elements import MLP, DiscreteEncoder
from core.model.model_utils.ppgn import PPGNLayer

class PPGN(nn.Module):
    def __init__(self, nfeat_node, nfeat_edge, nhid, nout, nlayer, pooling='add'):
        super().__init__()
        self.input_encoder = DiscreteEncoder(nhid) if nfeat_node is None else MLP(nfeat_node, 1)
        self.edge_encoder = DiscreteEncoder(nhid) if nfeat_edge is None else MLP(nfeat_edge, nhid, 1)
        self.layer = PPGNLayer(nhid, nhid, nlayer, depth_of_mlp=2)
        self.output_encoder = MLP(nhid, nout, nlayer=2, with_final_activation=False)

    def reset_parameters(self):
        self.input_encoder.reset_parameters()
        self.edge_encoder.reset_parameters()
        self.layer.reset_parameters()
        self.output_encoder.reset_parameters()    

    def forward(self, data):
        # encode x and edges
        x = data.x if len(data.x.shape) <= 2 else data.x.squeeze(-1)
        x = self.input_encoder(x)
        edge_attr = data.edge_index.new_zeros(data.edge_index.size(-1)) if data.edge_attr is None else data.edge_attr 
        edge_attr = self.edge_encoder(edge_attr)

        x = self.layer(x, data.edge_index, edge_attr, data.batch)
        x = scatter(x, data.batch, dim=0, reduce='add')
        x = self.output_encoder(x)
        return x