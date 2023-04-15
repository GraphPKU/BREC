# PPGN layer
import torch 
import torch.nn as nn
from core.model.model_utils.elements import MLP, Identity
import torch.nn.functional as F

class PPGNLayer(nn.Module):
    def __init__(self, nin, nout, nlayer, depth_of_mlp=2):
        super().__init__()
        # First part - sequential mlp blocks
        self.combine_node_edge = nn.Linear(2*nin, nin)
        self.reg_blocks = nn.ModuleList([RegularBlock(nin, nin, depth_of_mlp) for i in range(nlayer)])
        # Second part
        # self.norm = Identity() # 
        # self.norm = nn.BatchNorm1d(3*nin)
        self.norm = Identity()
        self.output_encoder = MLP(3*nin, nout, nlayer=depth_of_mlp, with_final_activation=True)
        # self.output_encoder = MLP(2*nin, nout, nlayer=depth_of_mlp, with_final_activation=True)

    def reset_parameters(self):
        self.combine_node_edge.reset_parameters()
        self.norm.reset_parameters()
        self.output_encoder.reset_parameters()
        for reg in self.reg_blocks:
            reg.reset_parameters()

    def forward(self, x, edge_index, edge_attr, batch):
        # to dense first
        x, adj, mask_x = to_dense_batch(x, edge_index, edge_attr, batch) # B x N_max x N_max x F
        ### TODO: for PPGN-AK we need to make N_max smaller, by make batch hasing more disconnected component

        # combine x and adj 
        x_mat = torch.zeros_like(adj)
        idx_tmp = range(x.size(1))
        x_mat[:, idx_tmp, idx_tmp, :] = x
        # adj[:, idx_tmp, idx_tmp, :] = x
        # print(x_mat.shape, adj.shape)
        # torch.cat([x_mat, adj], dim=-1)
        # exit(0)
        x = self.combine_node_edge(torch.cat([x_mat, adj], dim=-1))
        x = torch.transpose(x, 1, 3) # B x F x N_max x N_max 

        # create new mask 
        mask_adj = mask_x.unsqueeze(2) * mask_x.unsqueeze(1) # Bx N_max x N_max

        for block in self.reg_blocks:
            # consider add residual connection here?
            x = block(x, mask_adj)

        # 2nd order to 1st order matrix
        diag_x = x[:, :, idx_tmp, idx_tmp] # B x F x N_max
        x[:, :, idx_tmp, idx_tmp] = 0
        offdiag_x_max = x.max(dim=-1)[0] # B x F x N_max,  use summation here, can change to mean or max.
        # offdiag_x_max = x.mean(dim=-1)
        offdiag_x_mean = x.mean(dim=-1)
        x = torch.cat([diag_x, offdiag_x_max, offdiag_x_mean], dim=1).transpose(1, 2) # B x N_max x 2F
        # x = torch.cat([diag_x, offdiag_x_mean], dim=1).transpose(1, 2)

        # to sparse x 
        x = x.reshape(-1, x.size(-1))[mask_x.reshape(-1)] # BN x F

        x = self.norm(x)

        # transform feature by mlp
        x = self.output_encoder(x) # BN x F
        return x


#######################################################################################################
# Helpers for PPGN, from original repo: https://github.com/hadarser/ProvablyPowerfulGraphNetworks_torch

class RegularBlock(nn.Module):
    """
    Imputs: N x input_depth x m x m
    Take the input through 2 parallel MLP routes, multiply the result, and add a skip-connection at the end.
    At the skip-connection, reduce the dimension back to output_depth
    :return: Tensor of shape N x output_depth x m x m
    """
    def __init__(self, in_features, out_features, depth_of_mlp=2):
        super().__init__()
        self.mlp1 = MlpBlock(in_features, out_features, depth_of_mlp)
        self.mlp2 = MlpBlock(in_features, out_features, depth_of_mlp)
        self.skip = SkipConnection(in_features+out_features, out_features)
    
    def reset_parameters(self):
        self.mlp1.reset_parameters()
        self.mlp2.reset_parameters()
        self.skip.reset_parameters()

    def forward(self, inputs, mask):
        mask = mask.unsqueeze(1).to(inputs.dtype)
        mlp1 = self.mlp1(inputs) * mask
        mlp2 = self.mlp2(inputs) * mask

        mult = torch.matmul(mlp1, mlp2)

        out = self.skip(in1=inputs, in2=mult) * mask
        return out


class MlpBlock(nn.Module):
    """
    Block of MLP layers with activation function after each (1x1 conv layers).
    """
    def __init__(self, in_features, out_features, depth_of_mlp, activation_fn=nn.functional.relu_):
        super().__init__()
        self.activation = activation_fn
        self.convs = nn.ModuleList()
        for _ in range(depth_of_mlp):
            self.convs.append(nn.Conv2d(in_features, out_features, kernel_size=1, padding=0, bias=True))
            in_features = out_features

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()

    def forward(self, inputs):
        out = inputs
        for conv_layer in self.convs:
            out = self.activation(conv_layer(out))

        return out

class SkipConnection(nn.Module):
    """
    Connects the two given inputs with concatenation
    :param in1: earlier input tensor of shape N x d1 x m x m
    :param in2: later input tensor of shape N x d2 x m x m
    :param in_features: d1+d2
    :param out_features: output num of features
    :return: Tensor of shape N x output_depth x m x m
    """
    def __init__(self, in_features, out_features):
        super().__init__()
        self.conv = nn.Conv2d(in_features, out_features, kernel_size=1, padding=0, bias=True)

    def reset_parameters(self):
        self.conv.reset_parameters()

    def forward(self, in1, in2):
        # in1: N x d1 x m x m
        # in2: N x d2 x m x m
        out = torch.cat((in1, in2), dim=1)
        out = self.conv(out)
        return out

import torch_geometric.utils as pyg_utils
def to_dense_batch(x, edge_index, edge_attr, batch, max_num_nodes=None):
    x, mask = pyg_utils.to_dense_batch(x, batch, max_num_nodes=max_num_nodes) # B x N_max x F
    adj = pyg_utils.to_dense_adj(edge_index, batch, edge_attr, max_num_nodes=max_num_nodes)
    # x:  B x N_max x F
    # mask: B x N_max
    # adj: B x N_max x N_max x F
    return x, adj, mask
#
#######################################################################################################