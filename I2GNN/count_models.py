import torch
import torch.nn.functional as F
from torch.nn import Linear, Conv1d
from torch.nn import Sequential, ReLU
from torch_scatter import scatter_mean, scatter_add
from torch_geometric.nn import NNConv, GCNConv, RGCNConv, GatedGraphConv, GINConv
from torch_geometric.nn import GatedGraphConv
from torch_geometric.nn import (
    global_sort_pool, global_add_pool, global_mean_pool, global_max_pool
)
from torch_geometric.utils import dropout_adj, to_dense_adj, to_dense_batch, degree
# from utils import *
from modules.ppgn_modules import *
from modules.ppgn_layers import *
import numpy as np
from kernel.gatedgcn import HeteroGatedGraphConv
# from k_gnn import GraphConv, avg_pool
from kernel.graphconv import GraphConv

class GNN(torch.nn.Module):
    def __init__(self, dataset, num_layers=5, concat=False, y_ndim=1, **kwargs):
        super(GNN, self).__init__()
        self.concat = concat
        self.convs = torch.nn.ModuleList()
        self.node_type_embedding = torch.nn.Embedding(100, 8)
        self.y_ndim = y_ndim
        fc_in = 0

        M_in, M_out = 8, 64
        self.convs.append(GatedGraphConv(M_out, 1))


        for i in range(num_layers-1):
            M_in, M_out = M_out, 64
            self.convs.append(GatedGraphConv(M_out, 1))

        fc_in += M_out
        if self.concat:
            self.fc1 = torch.nn.Linear(32 + 64*(num_layers-1), 32)
        else:
            self.fc1 = torch.nn.Linear(fc_in, 32)
        self.fc2 = torch.nn.Linear(32, 16)

        # if self.use_max_dist:
        #    self.fc3 = torch.nn.Linear(17, 2)
        # else:
        self.fc3 = torch.nn.Linear(16, 1)

    def forward(self, data):
        if len(data.num_nodes) > 1:
            num_nodes = torch.sum(data.num_nodes)
        else:
            num_nodes = data.num_nodes
        x = torch.zeros([num_nodes, 8]).to(data.edge_index.device)
        xs = []
        for conv in self.convs:
            x = conv(x, data.edge_index)
            if self.concat:
                xs.append(x)
        if self.concat:
            x = torch.cat(xs, 1)

        if self.y_ndim == 1:
            x = scatter_add(x, data.batch, dim=0)

        x = F.elu(self.fc1(x))
        x = F.elu(self.fc2(x))

        x = self.fc3(x)
        return x


class I2GNN(torch.nn.Module):
    def __init__(self, dataset, num_layers=5, subgraph_pooling='mean', use_pos=False,
                 edge_attr_dim=5, use_rd=False, RNI=False, y_ndim=1, **kwargs):
        super(I2GNN, self).__init__()
        self.subgraph_pooling = subgraph_pooling
        self.use_pos = use_pos
        self.use_rd = use_rd
        self.RNI = RNI
        self.y_ndim = y_ndim
        if self.use_rd:
            self.rd_projection = torch.nn.Linear(2, 8)

        self.z_embedding_list = torch.nn.ModuleList()
        self.transform_before_conv = torch.nn.ModuleList()
        # self.z_embedding = torch.nn.Embedding(1000, 64)
        # self.node_type_embedding = torch.nn.Embedding(100, 8)

        # integer node label feature
        self.convs = torch.nn.ModuleList()
        # M_in, M_out = 64, 64
        M_in, M_out = 32, 32
        # M_in, M_out = dataset.num_features + 8, 1
        M_in = M_in + 3 if self.use_pos else M_in
        M_in = M_in * 2 if self.RNI else M_in
        # nn = Sequential(Linear(edge_attr_dim, 128), ReLU(), Linear(128, M_in * M_out))
        self.convs.append(GatedGraphConv(M_out, 1))
        # self.z_embedding_list.append(torch.nn.Embedding(100, M_in))
        self.z_embedding_list.append(torch.nn.Embedding(50, M_in))
        self.transform_before_conv.append(Linear(2*M_in, M_in))
        # self.convs.append(GraphConv(M_in, M_out))
        # self.convs.append(GatedRGCNConv(M_in, M_out, edge_attr_dim, aggr='add'))
        self.bns = torch.nn.ModuleList()
        self.bns.append(torch.nn.BatchNorm1d(M_out))

        for i in range(num_layers - 1):
            # M_in, M_out = M_out, 64
            M_in, M_out = M_out, 32
            # M_in, M_out = M_out, 1
            # nn = Sequential(Linear(M_in, 128), ReLU(), Linear(128, M_out))
            # self.convs.append(RGCNConv(M_in, M_out, edge_attr_dim, aggr='add'))
            self.convs.append(GatedGraphConv(M_out, 1))
            # self.z_embedding_list.append(torch.nn.Embedding(100, M_in))
            self.z_embedding_list.append(torch.nn.Embedding(50, M_in))
            self.transform_before_conv.append(Linear(2 * M_in, M_in))
            self.bns.append(torch.nn.BatchNorm1d(M_out))
            # self.convs.append(GINConv(nn))
            # self.convs.append(GraphConv(M_in, M_out))

        # subgraph gnn
        # self.subgraph_convs = torch.nn.ModuleList()
        # for i in range(2):
        #    M_in, M_out = M_out, 64
            # M_in, M_out = M_out, 1
       #     nn = Sequential(Linear(edge_attr_dim, 128), ReLU(), Linear(128, M_in * M_out))
       #     self.subgraph_convs.append(NNConv(M_in, M_out, nn))

        # MLPs for hierarchical pooling
        # self.tuple_pooling_nn = Sequential(Linear(2 * M_out, M_out), ReLU(), Linear(M_out, M_out))
        self.edge_pooling_nn = Sequential(Linear(M_out, M_out), ReLU(), Linear(M_out, M_out))
        self.node_pooling_nn = Sequential(Linear(M_out, M_out), ReLU(), Linear(M_out, M_out))

        # self.fc1 = torch.nn.Linear(M_out, 32)
        self.fc1 = torch.nn.Linear(M_out, 16)
        # self.fc2 = torch.nn.Linear(32, 16)
        self.fc3 = torch.nn.Linear(16, 16)

        self.bn1 = torch.nn.BatchNorm1d(16, affine=False, momentum=0.99)
        # self.bn2 = torch.nn.BatchNorm1d(16, affine=False, momentum=0.99)
        # self.bn3 = torch.nn.BatchNorm1d(16, affine=False, momentum=0.99)
        

    def forward(self, data):

        # node label embedding
        # z_emb = 0
        # if 'z' in data:
            ### computing input node embedding
       #     z_emb = self.z_embedding(data.z)
       #     if z_emb.ndim == 3:
        #        z_emb = z_emb.sum(dim=1)

        # if self.use_rd and 'rd' in data:
        #     rd_proj = self.rd_projection(data.rd)
        #     z_emb += rd_proj

        # integer node type embedding
        # x = z_emb
        x = None

        # concatenate with continuous node features
        # x = torch.cat([x, data.x.view(-1, 1)], -1)

        if self.use_pos:
            x = torch.cat([x, data.pos], 1)

        if self.RNI:
            rand_x = torch.rand(*x.size()).to(x.device) * 2 - 1
            x = torch.cat([x, rand_x], 1)

        for layer, conv in enumerate(self.convs):
            z = self.z_embedding_list[layer](data.z)
            if z.ndim == 3:
                z = z.sum(dim=1)
            if x == None:
                x = z.clone()
            else:
                x = torch.cat([x, z], dim=-1)
                x = self.transform_before_conv[layer](x)
            # x = F.elu(conv(x, data.edge_index, data.edge_attr))
            # x = x + global_add_pool(x, data.node_to_subgraph2)[data.node_to_subgraph2]
            x = conv(x, data.edge_index)
            x = self.bns[layer](x)
            # x = x + center_pool(x, data.node_to_subgraph2)[data.node_to_subgraph2]
            # x = x + global_add_pool(x, data.node_to_subgraph2)[data.node_to_subgraph2]

        # inner-subgraph2 communications
        # x_e = global_add_pool(x, data.node_to_subgraph2) # first learn (i,j)
        # x_e = x_e[data.node_to_subgraph2] # spread to all nodes
        # x = torch.cat([x, x_e], dim=-1)
        # x = torch.cat([x, center_pool(x, data.node_to_subgraph2)[data.node_to_subgraph2]], dim=-1)
        # x = self.tuple_pooling_nn(x)

        # subgraph-level pooling
        if self.subgraph_pooling == 'mean':
            x = global_add_pool(x, data.node_to_subgraph2)
            x = self.edge_pooling_nn(x)
            # x_e = global_add_pool(x, data.node_to_subgraph2)
            # x = torch.cat([x, x_e], dim=-1)
            # x = self.edge_pooling_nn(x)
            # x = global_add_pool(x, data.node_to_subgraph2)
            x = global_add_pool(x, data.subgraph2_to_subgraph)
            x = self.node_pooling_nn(x)
            # x = global_add_pool(x, data.node_to_subgraph_node)
            # x = self.edge_pooling_nn(x)
            # x = global_add_pool(x, data.subgraph_node_to_subgraph)
            # x = self.node_pooling_nn(x)
        elif self.subgraph_pooling == 'center':
            node_to_subgraph2 = data.node_to_subgraph2.cpu().numpy()
            # the first node of each subgraph is its center
            _, center_indices = np.unique(node_to_subgraph2, return_index=True)
            x = x[center_indices]
            x = global_mean_pool(x, data.subgraph2_to_subgraph)

        # pooling to subgraph node
        # x = scatter_mean(x, data.node_to_subgraph_node, dim=0)
        # subgraph gnn
        # x = x
        # for conv in self.subgraph_convs:
        #    x = F.elu(conv(x, data.subgraph_edge_index, data.subgraph_edge_attr))
        # subgraph node to subgraph
        # x = scatter_mean(x, data.subgraph_node_to_subgraph, dim=0)
        # subgraph to graph
        if self.y_ndim == 1:
            x = global_add_pool(x, data.subgraph_to_graph)
        # x = global_add_pool(x, data.subgraph_to_graph)
        # x = global_max_pool(x, data.subgraph_to_graph)

        x = F.elu(self.fc1(x))
        x = self.bn1(x)
        # x = F.elu(self.fc2(x))
        # x = self.bn2(x)
        x = self.fc3(x)
        # x = self.bn3(x)
        # x = F.log_softmax(x, dim=-1)
        return x


class NGNN(torch.nn.Module):
    def __init__(self, dataset, num_layers=5, subgraph_pooling='mean', use_pos=False,
                 edge_attr_dim=5, use_rd=False, RNI=False, y_ndim=1, **kwargs):
        super(NGNN, self).__init__()
        self.subgraph_pooling = subgraph_pooling
        self.use_pos = use_pos
        self.use_rd = use_rd
        self.RNI = RNI
        self.y_ndim = y_ndim
        if self.use_rd:
            self.rd_projection = torch.nn.Linear(1, 8)
        
        # self.z_embedding = torch.nn.Embedding(1000, 8)
        self.z_embedding_list = torch.nn.ModuleList()
        self.transform_before_conv = torch.nn.ModuleList()
        # self.node_type_embedding = torch.nn.Embedding(100, 8)

        # integer node label feature
        self.convs = torch.nn.ModuleList()
        M_in, M_out = 64, 64
        # M_in, M_out = dataset.num_features + 8, 1
        M_in = M_in + 3 if self.use_pos else M_in
        M_in = M_in * 2 if self.RNI else M_in
        # nn = Sequential(Linear(edge_attr_dim, 128), ReLU(), Linear(128, M_in * M_out))
        self.convs.append(GatedGraphConv(M_out, 1))
        self.z_embedding_list.append(torch.nn.Embedding(100, M_in))
        self.transform_before_conv.append(Linear(2 * M_in, M_in))

        for i in range(num_layers-1):
            M_in, M_out = M_out, 64
            # M_in, M_out = M_out, 1
            # nn = Sequential(Linear(edge_attr_dim, 128), ReLU(), Linear(128, M_in * M_out))
            self.convs.append(GatedGraphConv(M_out, 1))
            self.z_embedding_list.append(torch.nn.Embedding(100, M_in))
            self.transform_before_conv.append(Linear(2 * M_in, M_in))

        self.node_pooling_nn = Sequential(Linear(M_out, M_out), ReLU(), Linear(M_out, M_out))

        self.fc1 = torch.nn.Linear(M_out, 32)
        self.fc2 = torch.nn.Linear(32, 16)
        self.fc3 = torch.nn.Linear(16, 1)

    def forward(self, data):

        # node label embedding
        # z_emb = 0
        #if 'z' in data:
            ### computing input node embedding
        #    z_emb = self.z_embedding(data.z)
        #    if z_emb.ndim == 3:
        #        z_emb = z_emb.sum(dim=1)
        
        #if self.use_rd and 'rd' in data:
        #    rd_proj = self.rd_projection(data.rd)
        #    z_emb += rd_proj
            
        # integer node type embedding
        # x = z_emb
        x = None
            
        # concatenate with continuous node features
        # x = torch.cat([x, data.x.view(-1, 1)], -1)
        for layer, conv in enumerate(self.convs):
            # x = F.elu(conv(x, data.edge_index, data.edge_attr))
            z = self.z_embedding_list[layer](data.z)
            if z.ndim == 3:
                z = z.sum(dim=1)
            if x == None:
                x = z
            else:
                x = torch.cat([x, z], dim=-1)
                x = self.transform_before_conv[layer](x)
            x = conv(x, data.edge_index)

        # subgraph-level pooling
        if self.subgraph_pooling == 'mean':
            x = global_add_pool(x, data.node_to_subgraph)
            # x = self.node_pooling_nn(x)
        elif self.subgraph_pooling == 'center':
            node_to_subgraph = data.node_to_subgraph.cpu().numpy()
            # the first node of each subgraph is its center
            _, center_indices = np.unique(node_to_subgraph, return_index=True)
            x = x[center_indices]
        
        # graph-level pooling
        if self.y_ndim == 1:
            x = global_add_pool(x, data.subgraph_to_graph)
        #x = global_add_pool(x, data.subgraph_to_graph)
        #x = global_max_pool(x, data.subgraph_to_graph)

        x = F.elu(self.fc1(x))
        x = F.elu(self.fc2(x))
        x = self.fc3(x)
        # x = F.log_softmax(x, dim=-1)
        return x


class GNNAK(torch.nn.Module):
    def __init__(self, dataset, num_layers=5, subgraph_pooling='mean', use_pos=False,
                 edge_attr_dim=5, use_rd=False, RNI=False, y_ndim=1, **kwargs):
        super(GNNAK, self).__init__()
        self.subgraph_pooling = subgraph_pooling
        self.y_ndim = y_ndim
        # self.z_embedding = torch.nn.Embedding(1000, 8)
        self.z_embedding_list = torch.nn.ModuleList()
        self.transform_before_conv = torch.nn.ModuleList()
        self.pooling_nns = torch.nn.ModuleList()
        self.subgraph_gate = torch.nn.ModuleList()
        self.context_gate = torch.nn.ModuleList()

        self.convs = torch.nn.ModuleList()
        M_in, M_out = 64, 64
        # M_in, M_out = dataset.num_features + 8, 1
        # nn = Sequential(Linear(edge_attr_dim, 128), ReLU(), Linear(128, M_in * M_out))
        self.convs.append(GatedGraphConv(M_out, 1))
        self.z_embedding_list.append(torch.nn.Embedding(100, M_in))
        self.transform_before_conv.append(Linear(2 * M_in, M_in))
        self.pooling_nns.append(Sequential(Linear(3*M_out, M_out), ReLU(), Linear(M_out, M_out)))
        self.subgraph_gate.append(Sequential(Linear(M_in, M_out), torch.nn.Sigmoid()))
        self.context_gate.append(Sequential(Linear(M_in, M_out), torch.nn.Sigmoid()))
        for i in range(num_layers - 1):
            M_in, M_out = M_out, 64
            # M_in, M_out = M_out, 1
            # nn = Sequential(Linear(edge_attr_dim, 128), ReLU(), Linear(128, M_in * M_out))
            self.convs.append(GatedGraphConv(M_out, 1))
            self.z_embedding_list.append(torch.nn.Embedding(100, M_in))
            self.transform_before_conv.append(Linear(2 * M_in, M_in))
            self.pooling_nns.append(Sequential(Linear(3 * M_out, M_out), ReLU(), Linear(M_out, M_out)))
            self.subgraph_gate.append(Sequential(Linear(M_in, M_out), torch.nn.Sigmoid()))
            self.context_gate.append(Sequential(Linear(M_in, M_out), torch.nn.Sigmoid()))

        self.z_embedding_list.append(torch.nn.Embedding(100, M_in))
        self.pooling_nns.append(Sequential(Linear(3 * M_out, M_out), ReLU(), Linear(M_out, M_out)))
        self.subgraph_gate.append(Sequential(Linear(M_in, M_out), torch.nn.Sigmoid()))
        self.context_gate.append(Sequential(Linear(M_in, M_out), torch.nn.Sigmoid()))

        self.fc1 = torch.nn.Linear(M_out, 32)
        self.fc2 = torch.nn.Linear(32, 16)
        self.fc3 = torch.nn.Linear(16, 1)

    def forward(self, data):
        x = None

        for layer, conv in enumerate(self.convs):
            # add distance encoding
            z = self.z_embedding_list[layer](data.z)
            if z.ndim == 3:
                z = z.sum(dim=1)
            if x == None:
                x = z
            else:
                x = torch.cat([x, z], dim=-1)
                x = self.transform_before_conv[layer](x)

            # convolution layer
            x = conv(x, data.edge_index)

            # update node embeddings in subgraphs
            x = self.graph_pooling(x, data, z, layer, node_emb_only=True)[data.node_to_original_node]

        # graph pooling
        z = self.z_embedding_list[-1](data.z)
        if z.ndim == 3:
            z = z.sum(dim=1)
        if self.y_ndim == 1:
            x = self.graph_pooling(x, data, z, -1)
        else:
            x = self.graph_pooling(x, data, z, -1, node_emb_only=True)
        # x = global_add_pool(x, data.subgraph_to_graph)
        # x = global_max_pool(x, data.subgraph_to_graph)

        x = F.elu(self.fc1(x))
        x = F.elu(self.fc2(x))
        x = self.fc3(x)
        # x = F.log_softmax(x, dim=-1)
        return x

    def graph_pooling(self, x, data, z, layer, node_emb_only=False):
        # subgraph, center and context pooling
        x_subgraph = global_add_pool(self.subgraph_gate[layer](z) * x, data.node_to_subgraph)
        x_center = center_pool(x, data.node_to_subgraph)
        x_context = global_add_pool(self.context_gate[layer](z) * x, data.node_to_original_node)
        x = torch.cat([x_subgraph, x_center, x_context], dim=-1)
        x = self.pooling_nns[layer](x)
        if node_emb_only:
            return x
        else:
            return global_add_pool(x, data.subgraph_to_graph)


class IDGNN(torch.nn.Module):
    def __init__(self, dataset, num_layers=5, subgraph_pooling='mean', use_pos=False,
                 edge_attr_dim=5, use_rd=False, RNI=False, y_ndim=1, **kwargs):
        super(IDGNN, self).__init__()
        self.subgraph_pooling = subgraph_pooling
        self.y_ndim = y_ndim
        self.node_embedding = nn.Embedding(1, 64)
        self.convs = torch.nn.ModuleList()
        M_in, M_out = 64, 64
        # M_in, M_out = dataset.num_features + 8, 1
        # nn = Sequential(Linear(edge_attr_dim, 128), ReLU(), Linear(128, M_in * M_out))
        self.convs.append(HeteroGatedGraphConv(M_out, 1))
        for i in range(num_layers - 1):
            M_in, M_out = M_out, 64
            # M_in, M_out = M_out, 1
            # nn = Sequential(Linear(edge_attr_dim, 128), ReLU(), Linear(128, M_in * M_out))
            self.convs.append(HeteroGatedGraphConv(M_out, 1))

        self.fc1 = torch.nn.Linear(M_out, 32)
        self.fc2 = torch.nn.Linear(32, 16)
        self.fc3 = torch.nn.Linear(16, 1)

    def forward(self, data):
        x = self.node_embedding(torch.zeros_like(data.z))
        if x.ndim == 3:
            x = x.sum(dim=1)
        z = data.z
        for layer, conv in enumerate(self.convs):
            # convolution layer
            x = conv(x, data.edge_index, z)

        # graph pooling
        x = global_add_pool(x, data.node_to_subgraph)
        # x = global_max_pool(x, data.subgraph_to_graph)

        if self.y_ndim == 1:
            x = global_add_pool(x, data.subgraph_to_graph)

        x = F.elu(self.fc1(x))
        x = F.elu(self.fc2(x))
        x = self.fc3(x)
        # x = F.log_softmax(x, dim=-1)
        return x



# Provably Powerful Graph Networks
class PPGN(torch.nn.Module):
    # Provably powerful graph networks
    def __init__(self, dataset, emb_dim=64, use_embedding=False, use_spd=False, y_ndim=1,
                 **kwargs):
        super(PPGN, self).__init__()

        self.use_embedding = use_embedding
        self.use_spd = use_spd
        self.y_ndim = y_ndim

        initial_dim = 2

        if self.use_spd:
            initial_dim += 1

        # ppgn modules
        num_blocks = 2
        num_rb_layers = 4
        num_fc_layers = 2

        self.ppgn_rb = torch.nn.ModuleList()
        self.ppgn_rb.append(RegularBlock(num_blocks, initial_dim, emb_dim))
        for i in range(num_rb_layers - 1):
            self.ppgn_rb.append(RegularBlock(num_blocks, emb_dim, emb_dim))

        self.ppgn_fc = torch.nn.ModuleList()
        self.ppgn_fc.append(FullyConnected(emb_dim * 2, emb_dim))
        for i in range(num_fc_layers - 2):
            self.ppgn_fc.append(FullyConnected(emb_dim, emb_dim))
        self.ppgn_fc.append(FullyConnected(emb_dim, 1, activation_fn=None))

    def forward(self, data):
        # prepare dense data
        device = data.edge_index.device
        node_embedding = torch.zeros(data.num_nodes.sum()).to(device)
        dense_edge_data = to_dense_adj(
            data.edge_index, data.batch
        )  # |graphs| * max_nodes * max_nodes * edge_data_dim
        dense_edge_data = torch.unsqueeze(dense_edge_data, -1)
        dense_node_data, mask = to_dense_batch(node_embedding, data.batch) # |graphs| * max_nodes * d
        dense_node_data = torch.unsqueeze(dense_node_data, -1)
        shape = dense_node_data.shape
        shape = (shape[0], shape[1], shape[1], shape[2])
        diag_node_data = torch.empty(*shape).to(device)

        if self.use_spd:
            dense_dist_mat = torch.zeros(shape[0], shape[1], shape[1], 1).to(device)

        for g in range(shape[0]):
            for i in range(shape[-1]):
                diag_node_data[g, :, :, i] = torch.diag(dense_node_data[g, :, i])

        if self.use_spd:
            z = torch.cat([dense_dist_mat, dense_edge_data, diag_node_data], -1)
        else:
            z = torch.cat([dense_edge_data, diag_node_data], -1)
        z = torch.transpose(z, 1, 3)

        # ppng
        for rb in self.ppgn_rb:
            z = rb(z)

        # z = diag_offdiag_maxpool(z)
        if self.y_ndim == 1:
            z = diag_offdiag_meanpool(z, level='graph')
        else:
            z = diag_offdiag_meanpool(z, level='node')

        # reshape
        if z.ndim == 3:
            z = torch.transpose(z, -1, -2)
        for fc in self.ppgn_fc:
            z = fc(z)
        if z.ndim == 3:
            # from dense to sparse
            z = z.view(-1, 1)
            z = z[mask.view(-1)]
        torch.cuda.empty_cache()
        return z


# pooling functions
def center_pool(x, node_to_subgraph):
    node_to_subgraph = node_to_subgraph.cpu().numpy()
    # the first node of each subgraph is its center
    _, center_indices = np.unique(node_to_subgraph, return_index=True)
    # x = x[center_indices]
    return x[center_indices]
