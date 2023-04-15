import torch
import random
import math
import pdb
import torch.nn as nn
import torch.nn.functional as F
import scipy.sparse as ssp
from scipy import linalg
from scipy.linalg import inv, eig, eigh
import numpy as np
from torch_geometric.data import Data
from torch_geometric.utils import to_scipy_sparse_matrix
from torch_scatter import scatter_min
from batch import Batch
from collections import defaultdict
from copy import deepcopy
import networkx as nx


def create_subgraphs3(data, h=1, sample_ratio=1.0, max_nodes_per_hop=None,
                      node_label='hop', use_rd=False, subgraph_pretransform=None):
    # Given a PyG graph data, extract an h-hop rooted subgraph for each of its
    # nodes, and combine these node-subgraphs into a new large disconnected graph
    # If given a list of h, will return multiple subgraphs for each node stored in
    # a dict.

    if type(h) == int:
        h = [h]
    assert (isinstance(data, Data))
    x, edge_index, num_nodes = data.x, data.edge_index, data.num_nodes

    new_data_multi_hop = {}
    for h_ in h:
        subgraphs = []
        for ind in range(num_nodes):
            nodes_, edge_index_, edge_mask_, z_ = k_hop_subgraph(
                ind, h_, edge_index, True, num_nodes, node_label=node_label,
                max_nodes_per_hop=max_nodes_per_hop
            )
            x_ = None
            edge_attr_ = None
            pos_ = None
            if x is not None:
                x_ = x[nodes_]
            else:
                x_ = None

            if 'node_type' in data:
                node_type_ = data.node_type[nodes_]

            if data.edge_attr is not None:
                edge_attr_ = data.edge_attr[edge_mask_]
            if data.pos is not None:
                pos_ = data.pos[nodes_]
            data_ = data.__class__(x_, edge_index_, edge_attr_, None, pos_, z=z_)
            data_.num_nodes = nodes_.shape[0]

            if 'node_type' in data:
                data_.node_type = node_type_

            # pre-process subgraph-2
            # subgraph_edge_index = data_.edge_index
            # subgraph_edge_attr = data_.edge_attr
            # if 'pos' in data_.keys:
            #    subgraph_pos = data_.pos
            # subgraph_node_to_subgraph = torch.zeros([data_.num_nodes])
            data_ = subgraph_to_subgraph3(data_, h_, node_label=node_label, use_rd=use_rd)
            # data_.subgraph_node_to_subgraph = subgraph_node_to_subgraph
            # data_.num_subgraphs = 1
            # data_.subgraph_edge_index = subgraph_edge_index
            # data_.subgraph_edge_attr = subgraph_edge_attr
            # if 'pos' in data_.keys:
            #    data_.subgraph_pos = subgraph_pos

            if subgraph_pretransform is not None:  # for k-gnn
                data_ = subgraph_pretransform(data_)
                if 'assignment_index_2' in data_:
                    data_.batch_2 = torch.zeros(
                        data_.iso_type_2.shape[0], dtype=torch.long
                    )
                if 'assignment_index_3' in data_:
                    data_.batch_3 = torch.zeros(
                        data_.iso_type_3.shape[0], dtype=torch.long
                    )

            subgraphs.append(data_)

        # new_data is treated as a big disconnected graph of the batch of subgraphs
        new_data = Batch.from_data_list(subgraphs)
        new_data.num_nodes = sum(data_.num_nodes for data_ in subgraphs)
        new_data.num_subgraphs = len(subgraphs)

        # save some memories
        # new_data.original_edge_index = edge_index
        # new_data.original_edge_attr = data.edge_attr
        # new_data.original_pos = data.pos


        subgraph_node_to_subgraph = []
        for i, data_ in enumerate(subgraphs):
            subgraph_node_to_subgraph = subgraph_node_to_subgraph + [i for _ in
                                                                     range(torch.max(data_.node_to_subgraph_node) + 1)]
        new_data.subgraph_node_to_subgraph = torch.tensor(subgraph_node_to_subgraph)

        # delete batch, because batch will be used to store node_to_graph assignment
        # new_data.node_to_subgraph = new_data.batch
        del new_data.batch
        if 'batch_2' in new_data:
            new_data.assignment2_to_subgraph = new_data.batch_2
            del new_data.batch_2
        if 'batch_3' in new_data:
            new_data.assignment3_to_subgraph = new_data.batch_3
            del new_data.batch_3

        # create a subgraph_to_graph assignment vector (all zero)
        new_data.subgraph_to_graph = torch.zeros(len(subgraphs), dtype=torch.long)
        # new_data.subgraph2_to_graph = torch.zeros(torch.sum(new_data.num_subgraphs2), dtype=torch.long)
        del new_data.num_subgraphs2

        # copy remaining graph attributes
        for k, v in data:
            if k not in ['x', 'edge_index', 'edge_attr', 'pos', 'num_nodes', 'batch',
                         'z', 'rd', 'node_type']:
                new_data[k] = v

        if len(h) == 1:
            return new_data
        else:
            new_data_multi_hop[h_] = new_data

    return new_data_multi_hop

def create_subgraphs2(data, h=1, sample_ratio=1.0, max_nodes_per_hop=None,
                     node_label='hop', use_rd=False, subgraph_pretransform=None, center_idx=True, self_loop=False):
    # Given a PyG graph data, extract an h-hop rooted subgraph for each of its
    # nodes, and combine these node-subgraphs into a new large disconnected graph
    # If given a list of h, will return multiple subgraphs for each node stored in
    # a dict.

    if type(h) == int:
        h = [h]
    assert (isinstance(data, Data))
    x, edge_index, num_nodes = data.x, data.edge_index, data.num_nodes

    new_data_multi_hop = {}
    relabels = []
    for h_ in h:
        subgraphs = []
        for ind in range(num_nodes):
            nodes_, edge_index_, edge_mask_, z_, relabel = k_hop_subgraph(
                ind, h_, edge_index, True, num_nodes, node_label=node_label,
                max_nodes_per_hop=max_nodes_per_hop
            )
            x_ = None
            edge_attr_ = None
            pos_ = None
            if x is not None:
                x_ = x[nodes_]
            else:
                x_ = None


            if 'node_type' in data:
                node_type_ = data.node_type[nodes_]

            if data.edge_attr is not None:
                edge_attr_ = data.edge_attr[edge_mask_]
            if data.pos is not None:
                pos_ = data.pos[nodes_]
            data_ = data.__class__(x_, edge_index_, edge_attr_, None, pos_, z=z_)
            data_.num_nodes = nodes_.shape[0]

            if 'node_type' in data:
                data_.node_type = node_type_

            # pre-process subgraph-2
            # subgraph_edge_index = data_.edge_index
            # subgraph_edge_attr = data_.edge_attr
            # if 'pos' in data_.keys:
            #    subgraph_pos = data_.pos
            # subgraph_node_to_subgraph = torch.zeros([data_.num_nodes])
            if center_idx:
                data_ = subgraph_to_subgraph2_with_idx(data_, h_, node_label=node_label, use_rd=use_rd, self_loop=self_loop)
            else:
                data_ = subgraph_to_subgraph2(data_, h_, node_label=node_label, use_rd=use_rd)
            # data_.subgraph_node_to_subgraph = subgraph_node_to_subgraph
            # data_.num_subgraphs = 1
            # data_.subgraph_edge_index = subgraph_edge_index
            # data_.subgraph_edge_attr = subgraph_edge_attr
            # if 'pos' in data_.keys:
            #    data_.subgraph_pos = subgraph_pos

            # node_to_original_node
            deg = torch.sum(edge_index_[0] == 0).item()
            if deg == 0:
                deg = 1
            relabels = relabels + list(np.array(relabel)) * deg

            if subgraph_pretransform is not None:  # for k-gnn
                data_ = subgraph_pretransform(data_)
                if 'assignment_index_2' in data_:
                    data_.batch_2 = torch.zeros(
                        data_.iso_type_2.shape[0], dtype=torch.long
                    )
                if 'assignment_index_3' in data_:
                    data_.batch_3 = torch.zeros(
                        data_.iso_type_3.shape[0], dtype=torch.long
                    )

            subgraphs.append(data_)

        # new_data is treated as a big disconnected graph of the batch of subgraphs
        new_data = Batch.from_data_list(subgraphs)
        new_data.num_nodes = sum(data_.num_nodes for data_ in subgraphs)
        new_data.num_subgraphs = len(subgraphs)

        # node_to_original_node
        new_data.node_to_original_node = torch.tensor(relabels)
        new_data.num_original_nodes = data.num_nodes
        # save some memories
        # new_data.original_edge_index = edge_index
        # new_data.original_edge_attr = data.edge_attr
        # new_data.original_pos = data.pos


        # subgraph_node_to_subgraph = []
        # for i, data_ in enumerate(subgraphs):
        #    subgraph_node_to_subgraph = subgraph_node_to_subgraph + [i for _ in
        #                                                             range(torch.max(data_.node_to_subgraph_node) + 1)]
        # new_data.subgraph_node_to_subgraph = torch.tensor(subgraph_node_to_subgraph)

        # delete batch, because batch will be used to store node_to_graph assignment
        # new_data.node_to_subgraph = new_data.batch
        del new_data.batch
        if 'batch_2' in new_data:
            new_data.assignment2_to_subgraph = new_data.batch_2
            del new_data.batch_2
        if 'batch_3' in new_data:
            new_data.assignment3_to_subgraph = new_data.batch_3
            del new_data.batch_3

        # create a subgraph_to_graph assignment vector (all zero)
        new_data.subgraph_to_graph = torch.zeros(len(subgraphs), dtype=torch.long)
        new_data.subgraph2_to_graph = torch.zeros(torch.sum(new_data.num_subgraphs2), dtype=torch.long)
        del new_data.num_subgraphs2

        # copy remaining graph attributes
        for k, v in data:
            if k not in ['x', 'edge_index', 'edge_attr', 'pos', 'num_nodes', 'batch',
                         'z', 'rd', 'node_type']:
                new_data[k] = v

        if len(h) == 1:
            return new_data
        else:
            new_data_multi_hop[h_] = new_data

    return new_data_multi_hop


def create_subgraphs(data, h=1, sample_ratio=1.0, max_nodes_per_hop=None,
                     node_label='hop', use_rd=False, subgraph_pretransform=None, save_relabel=False):
    # Given a PyG graph data, extract an h-hop rooted subgraph for each of its
    # nodes, and combine these node-subgraphs into a new large disconnected graph
    # If given a list of h, will return multiple subgraphs for each node stored in
    # a dict.

    if type(h) == int:
        h = [h]
    assert(isinstance(data, Data))
    x, edge_index, num_nodes = data.x, data.edge_index, data.num_nodes

    new_data_multi_hop = {}
    relabels = []
    for h_ in h:
        subgraphs = []
        for ind in range(num_nodes):
            nodes_, edge_index_, edge_mask_, z_, relabel = k_hop_subgraph(
                ind, h_, edge_index, True, num_nodes, node_label=node_label, 
                max_nodes_per_hop=max_nodes_per_hop
            )
            x_ = None
            edge_attr_ = None
            pos_ = None
            if x is not None:
                x_ = x[nodes_]
            else:
                x_ = None

            if 'node_type' in data:
                node_type_ = data.node_type[nodes_]

            if data.edge_attr is not None:
                edge_attr_ = data.edge_attr[edge_mask_]
            if data.pos is not None:
                pos_ = data.pos[nodes_]
            data_ = data.__class__(x_, edge_index_, edge_attr_, None, pos_, z=z_)
            data_.num_nodes = nodes_.shape[0]
            
            if 'node_type' in data:
                data_.node_type = node_type_

            if use_rd:
                # See "Link prediction in complex networks: A survey".
                adj = to_scipy_sparse_matrix(
                    edge_index_, num_nodes=nodes_.shape[0]
                ).tocsr()
                laplacian = ssp.csgraph.laplacian(adj).toarray()
                try:
                    L_inv = linalg.pinv(laplacian)
                except:
                    laplacian += 0.01 * np.eye(*laplacian.shape)
                lxx = L_inv[0, 0]
                lyy = L_inv[list(range(len(L_inv))), list(range(len(L_inv)))]
                lxy = L_inv[0, :]
                lyx = L_inv[:, 0]
                rd_to_x = torch.FloatTensor((lxx + lyy - lxy - lyx)).unsqueeze(1)
                data_.rd = rd_to_x

            relabels = relabels + list(np.array(relabel))

            if subgraph_pretransform is not None:  # for k-gnn
                data_ = subgraph_pretransform(data_)
                if 'assignment_index_2' in data_:
                    data_.batch_2 = torch.zeros(
                        data_.iso_type_2.shape[0], dtype=torch.long
                    )
                if 'assignment_index_3' in data_:
                    data_.batch_3 = torch.zeros(
                        data_.iso_type_3.shape[0], dtype=torch.long
                    )

            subgraphs.append(data_)

        # new_data is treated as a big disconnected graph of the batch of subgraphs
        new_data = Batch.from_data_list(subgraphs)
        new_data.num_nodes = sum(data_.num_nodes for data_ in subgraphs)
        new_data.num_subgraphs = len(subgraphs)
        if save_relabel:
            new_data.node_to_original_node = torch.tensor(relabels)
            new_data.num_original_nodes = data.num_nodes
        new_data.original_edge_index = edge_index
        new_data.original_edge_attr = data.edge_attr
        new_data.original_pos = data.pos
            
        # rename batch, because batch will be used to store node_to_graph assignment
        new_data.node_to_subgraph = new_data.batch
        del new_data.batch
        if 'batch_2' in new_data:
            new_data.assignment2_to_subgraph = new_data.batch_2
            del new_data.batch_2
        if 'batch_3' in new_data:
            new_data.assignment3_to_subgraph = new_data.batch_3
            del new_data.batch_3

        # create a subgraph_to_graph assignment vector (all zero)
        new_data.subgraph_to_graph = torch.zeros(len(subgraphs), dtype=torch.long)
        
        # copy remaining graph attributes
        for k, v in data:
            if k not in ['x', 'edge_index', 'edge_attr', 'pos', 'num_nodes', 'batch',
                         'z', 'rd', 'node_type']:
                new_data[k] = v

        if len(h) == 1:
            return new_data
        else:
            new_data_multi_hop[h_] = new_data

    return new_data_multi_hop


def k_hop_subgraph(node_idx, num_hops, edge_index, relabel_nodes=False,
                   num_nodes=None, flow='source_to_target', node_label='hop', 
                   max_nodes_per_hop=None):

    num_nodes = maybe_num_nodes(edge_index, num_nodes)

    assert flow in ['source_to_target', 'target_to_source']
    if flow == 'target_to_source':
        row, col = edge_index
    else:
        col, row = edge_index

    node_mask = row.new_empty(num_nodes, dtype=torch.bool)
    edge_mask = row.new_empty(row.size(0), dtype=torch.bool)

    subsets = [torch.tensor([node_idx], device=row.device).flatten()]
    visited = set(subsets[-1].tolist())
    label = defaultdict(list)
    for node in subsets[-1].tolist():
        label[node].append(1)
    if node_label == 'hop':
        hops = [torch.LongTensor([0], device=row.device).flatten()]
    for h in range(num_hops):
        node_mask.fill_(False)
        node_mask[subsets[-1]] = True
        torch.index_select(node_mask, 0, row, out=edge_mask)
        new_nodes = col[edge_mask]
        tmp = []
        for node in new_nodes.tolist():
            if node in visited:
                continue
            tmp.append(node)
            label[node].append(h+2)
        if len(tmp) == 0:
            break
        if max_nodes_per_hop is not None:
            if max_nodes_per_hop < len(tmp):
                tmp = random.sample(tmp, max_nodes_per_hop)
        new_nodes = set(tmp)
        visited = visited.union(new_nodes)
        new_nodes = torch.tensor(list(new_nodes), device=row.device)
        subsets.append(new_nodes)
        if node_label == 'hop':
            hops.append(torch.LongTensor([h+1] * len(new_nodes), device=row.device))
    subset = torch.cat(subsets)
    inverse_map = torch.tensor(range(subset.shape[0]))
    if node_label == 'hop':
        hop = torch.cat(hops)
    # Add `node_idx` to the beginning of `subset`.
    subset = subset[subset != node_idx]
    subset = torch.cat([torch.tensor([node_idx], device=row.device), subset])

    z = None
    if node_label == 'hop':
        hop = hop[hop != 0]
        hop = torch.cat([torch.LongTensor([0], device=row.device), hop])
        z = hop.unsqueeze(1)
        z = torch.zeros_like(z)
        z[0, 0] = 1.
    elif node_label.startswith('spd') or node_label == 'drnl' or node_label.startswith('dspd'):
        if node_label.startswith('spd'):
            # keep top k shortest-path distances
            num_spd = int(node_label[3:]) if len(node_label) > 3 else 2
            z = torch.zeros(
                [subset.size(0), num_spd], dtype=torch.long, device=row.device
            )
        elif node_label.startswith('dspd'):
            # keep top k shortest-path distances
            num_spd = int(node_label[4:]) if len(node_label) > 4 else 2
            z = torch.zeros(
                [subset.size(0), num_spd], dtype=torch.long, device=row.device
            )
        elif node_label == 'drnl':
            # see "Link Prediction Based on Graph Neural Networks", a special
            # case of spd2
            num_spd = 2
            z = torch.zeros([subset.size(0), 1], dtype=torch.long, device=row.device)

        for i, node in enumerate(subset.tolist()):
            dists = label[node][:num_spd]  # keep top num_spd distances
            if node_label == 'spd' or node_label == 'dspd':
                z[i][:min(num_spd, len(dists))] = torch.tensor(dists)
            elif node_label == 'drnl':
                dist1 = dists[0]
                dist2 = dists[1] if len(dists) == 2 else 0
                if dist2 == 0:
                    dist = dist1
                else:
                    dist = dist1 * (num_hops + 1) + dist2
                z[i][0] = dist
        
    node_mask.fill_(False)
    node_mask[subset] = True
    edge_mask = node_mask[row] & node_mask[col]

    edge_index = edge_index[:, edge_mask]

    if relabel_nodes:
        node_idx = row.new_full((num_nodes, ), -1)
        node_idx[subset] = torch.arange(subset.size(0), device=row.device)
        edge_index = node_idx[edge_index]

    return subset, edge_index, edge_mask, z, subset

def find_all_spd(node_idx, edge_index, node_label='spd', num_nodes=None, flow='source_to_target', delete_node=None):
    num_nodes = maybe_num_nodes(edge_index, num_nodes)

    assert flow in ['source_to_target', 'target_to_source']
    if flow == 'target_to_source':
        row, col = edge_index
    else:
        col, row = edge_index

    node_mask = row.new_empty(num_nodes, dtype=torch.bool)
    edge_mask = row.new_empty(row.size(0), dtype=torch.bool)

    subsets = [torch.tensor([node_idx], device=row.device).flatten()]
    visited = set(subsets[-1].tolist())
    label = defaultdict(list)
    for node in subsets[-1].tolist():
        label[node].append(1)

    # delete marked node
    if delete_node is not None:
        # subsets.append(torch.tensor([delete_node]))
        visited = visited.union([delete_node])
        # label[delete_node].append(0)

    for h in range(999):
        if len(subsets) == num_nodes:
            break
        node_mask.fill_(False)
        node_mask[subsets[-1]] = True
        torch.index_select(node_mask, 0, row, out=edge_mask)
        new_nodes = col[edge_mask]
        tmp = []
        for node in new_nodes.tolist():
            if node in visited:
                continue
            tmp.append(node)
            label[node].append(h + 2)
        if len(tmp) == 0:
            break
        new_nodes = set(tmp)
        visited = visited.union(new_nodes)
        new_nodes = torch.tensor(list(new_nodes), device=row.device)
        subsets.append(new_nodes)
    subset = torch.cat(subsets)
    inverse_map = torch.tensor(range(subset.shape[0]))
    # Add `node_idx` to the beginning of `subset`.
    subset = subset[subset != node_idx]
    subset = torch.cat([torch.tensor([node_idx], device=row.device), subset])

    z = None
    if node_label.startswith('spd') or node_label == 'drnl' or node_label.startswith('dspd'):
        if node_label.startswith('spd'):
            # keep top k shortest-path distances
            num_spd = int(node_label[3:]) if len(node_label) > 3 else 2
            z = torch.zeros(
                [subset.size(0), num_spd], dtype=torch.long, device=row.device
            )
        elif node_label.startswith('dspd'):
            # keep top k shortest-path distances
            # subset may be smaller than num_nodes, we use num_nodes instead
            num_spd = int(node_label[4:]) if len(node_label) > 4 else 2
            z = torch.zeros(
                [num_nodes, num_spd], dtype=torch.long, device=row.device
            )
        elif node_label == 'drnl':
            # see "Link Prediction Based on Graph Neural Networks", a special
            # case of spd2
            num_spd = 2
            z = torch.zeros([subset.size(0), 1], dtype=torch.long, device=row.device)

        # assert torch.max(edge_index) == subset.size(0) - 1 # cover the whole subgraph
        for i, node in enumerate(subset.tolist()):
            dists = label[node][:num_spd]  # keep top num_spd distances
            if node_label == 'spd' or node_label == 'dspd':
                z[node][:min(num_spd, len(dists))] = torch.tensor(dists) # do not re-label
            elif node_label == 'drnl':
                dist1 = dists[0]
                dist2 = dists[1] if len(dists) == 2 else 0
                if dist2 == 0:
                    dist = dist1
                else:
                    dist = dist1 * (num_hops + 1) + dist2
                z[node][0] = dist
    else:
        print('Use find_all_spd for non-path node labeling')
        exit(1)
    return z

def maybe_num_nodes(index, num_nodes=None):
	return index.max().item() + 1 if num_nodes is None else num_nodes


def neighbors(fringe, A):
    # Find all 1-hop neighbors of nodes in fringe from A
    res = set()
    for node in fringe:
        _, out_nei, _ = ssp.find(A[node, :])
        in_nei, _, _ = ssp.find(A[:, node])
        nei = set(out_nei).union(set(in_nei))
        res = res.union(nei)
    return res


class return_prob(object):
    def __init__(self, steps=50):
        self.steps = steps

    def __call__(self, data):
        adj = to_scipy_sparse_matrix(data.edge_index, num_nodes=data.num_nodes).tocsr()
        adj += ssp.identity(data.num_nodes, dtype='int', format='csr')
        rp = np.empty([data.num_nodes, self.steps])
        inv_deg = ssp.lil_matrix((data.num_nodes, data.num_nodes))
        inv_deg.setdiag(1 / adj.sum(1))
        P = inv_deg * adj
        if self.steps < 5:
            Pi = P
            for i in range(self.steps):
                rp[:, i] = Pi.diagonal()
                Pi = Pi * P
        else:
            inv_sqrt_deg = ssp.lil_matrix((data.num_nodes, data.num_nodes))
            inv_sqrt_deg.setdiag(1 / (np.array(adj.sum(1)) ** 0.5))
            B = inv_sqrt_deg * adj * inv_sqrt_deg
            L, U = eigh(B.todense())
            W = U * U
            Li = L
            for i in range(self.steps):
                rp[:, i] = W.dot(Li)
                Li = Li * L

        data.rp = torch.FloatTensor(rp)

        return data




def data2graph(d):
    import networkx as nx
    edge_index = d.edge_index.cpu().detach().numpy()
    G = nx.Graph()
    G.add_nodes_from([i for i in range(d.x.size(0))])
    G.add_edges_from([tuple(edge_index[:, i]) for i in range(edge_index.shape[1])])
    return G

def compute_rd(edge_index, num_nodes):
    adj = to_scipy_sparse_matrix(
        edge_index, num_nodes=num_nodes
    ).tocsr()
    laplacian = ssp.csgraph.laplacian(adj).toarray()
    try:
        L_inv = linalg.pinv(laplacian)
    except:
        laplacian += 0.01 * np.eye(*laplacian.shape)
    l_diag = L_inv[list(range(len(L_inv))), list(range(len(L_inv)))]
    l_i = np.tile(np.expand_dims(l_diag, axis=1), [1, len(L_inv)])
    l_j = np.tile(np.expand_dims(l_diag, axis=0), [len(L_inv), 1])
    rd = torch.FloatTensor(l_i + l_j - L_inv - L_inv.T)
    return rd



def subgraph_to_subgraph4(d, num_hops, center=0, node_label='spd', use_rd=False):
    # transform a pyg-data into multiple subgraphs by labeling neighbors
    neighbors = d.edge_index[1][torch.where(d.edge_index[0] == center)[0]] # find neighbors of ind node
    num_neighbors = len(neighbors)
    if use_rd:
        rd_to_x = torch.zeros([d.num_nodes * num_neighbors, 2])
        rd = compute_rd(d.edge_index, d.num_nodes)
    if num_neighbors == 0:
        # d.node_to_subgraph2 = torch.zeros([d.num_nodes]).long()
        # d.subgraph2_to_subgraph = torch.zeros([1]).long()
        # d.num_subgraphs2 = 1
        d.node_to_subgraph_node = torch.arange(d.num_nodes)
        if node_label.startswith('spd'):
            d.z = torch.tile(d.z, [1, 2])
        if node_label.startswith('dspd'):
            d.z = torch.tile(d.z, [1, 4])
        if use_rd:
            d.rd = torch.tile(rd[0, :].view([-1, 1]), [1, 2])
        return d

    if d.x != None:
        if d.x.ndim == 2:
            x = torch.tile(d.x, [num_neighbors, 1])
        elif d.x.ndim == 1:
            x = torch.tile(d.x, [num_neighbors])
        else:
            exit(1)
    if d.edge_attr != None:
        if d.edge_attr.ndim == 2:
            edge_attr = torch.tile(d.edge_attr, [num_neighbors, 1])
        elif d.edge_attr.ndim == 1:
            edge_attr = torch.tile(d.edge_attr, [num_neighbors])
        else:
            exit(1)
    if d.pos != None:
        pos = torch.tile(d.pos, [num_neighbors, 1])
    z = torch.zeros_like(d.z)
    if node_label.startswith('spd'):
        z = torch.tile(z, [num_neighbors, 2])
    elif node_label.startswith('dspd'):
        z = torch.tile(z, [num_neighbors, 4])
    else:
        z = torch.tile(z, [num_neighbors, 1])
    edge_index = torch.zeros([2, d.num_edges * num_neighbors]).long()
    node_to_subgraph_node = torch.arange(d.num_nodes)
    node_to_subgraph_node = torch.tile(node_to_subgraph_node, [num_neighbors]) # (i,j,k) to (i,k)
    # node_to_subgraph2 = torch.zeros([d.num_nodes * num_neighbors]).long() # (i,j,k) to (i,j)
    for i, n in enumerate(neighbors):
        # update labeling z
        if node_label.startswith('spd'):
            z_n = find_all_spd(n, edge_index=d.edge_index, node_label=node_label, num_nodes=d.num_nodes)
            z[i * d.num_nodes : (i + 1) * d.num_nodes] = torch.cat([d.z, z_n+num_hops+3], dim=-1) # +num_hops+3 such that it uses different distanc embedding
        elif node_label.startswith('dspd'):
            z_n = find_all_spd(n, edge_index=d.edge_index, node_label=node_label, num_nodes=d.num_nodes)
            z_n_delete_0 = find_all_spd(n, edge_index=d.edge_index, node_label=node_label, num_nodes=d.num_nodes,
                                        delete_node=0)
            z_0_delete_n = find_all_spd(0, edge_index=d.edge_index, node_label=node_label, num_nodes=d.num_nodes,
                                        delete_node=n.item())
            delta = num_hops + 5 # make distance embedding different
            z[i * d.num_nodes: (i + 1) * d.num_nodes] = torch.cat([d.z, z_n + delta, z_n_delete_0 + 2 * delta,
                                                                   z_0_delete_n + 3 * delta], dim=-1)
        else:
            z_n = d.z.clone()
            z_n[n] = 2
            z[i * d.num_nodes : (i + 1) * d.num_nodes] = z_n

        # update rd
        if use_rd:
            rd_to_x[i * d.num_nodes : (i + 1) * d.num_nodes] = torch.cat([rd[0, :].view([-1, 1]),
                                                                          rd[n, :].view([-1, 1])], dim=-1)
        # update edge index
        temp = d.edge_index + i * d.num_nodes
        edge_index[:, i * d.num_edges : (i + 1) * d.num_edges] = temp
        # node_to_subgraph2[i * d.num_nodes : (i + 1) * d.num_nodes] = i * torch.ones([d.num_nodes])
    # combine subgraph-2 into a subgraph data
    subgraphs_2 = Data(edge_index=edge_index, z=z, node_to_subgraph_node=node_to_subgraph_node)
    # subgraphs_2 = Data(edge_index=edge_index, z=z)
    if d.x != None:
        subgraphs_2.x = x
    if d.edge_attr != None:
        subgraphs_2.edge_attr = edge_attr
    if d.pos != None:
        subgraphs_2.pos = pos
    if use_rd:
        subgraphs_2.rd = rd_to_x
    # subgraphs_2 = Data(x=x, edge_index=edge_index, pos=pos, edge_attr=edge_attr, z=z, node_to_subgraph_node=node_to_subgraph_node)
    return subgraphs_2


def subgraph_to_subgraph2_with_idx(d, num_hops, center=0, node_label='spd', use_rd=False, self_loop=False):
    # transform a pyg-data into multiple subgraphs by labeling neighbors
    neighbors = d.edge_index[1][torch.where(d.edge_index[0] == center)[0]] # find neighbors of ind node
    # self-loop
    if self_loop:
        neighbors = torch.cat([neighbors, torch.tensor([0])])
    num_neighbors = len(neighbors)
    if use_rd:
        rd_to_x = torch.zeros([d.num_nodes * num_neighbors, 2])
        rd = compute_rd(d.edge_index, d.num_nodes)
    if num_neighbors == 0:
        d.node_to_subgraph2 = torch.zeros([d.num_nodes]).long()
        d.subgraph2_to_subgraph = torch.zeros([1]).long()
        d.num_subgraphs2 = 1
        if node_label.startswith('spd'):
            d.z = torch.tile(d.z, [1, 2])
        if node_label.startswith('dspd'):
            d.z = torch.tile(d.z, [1, 4])
        if use_rd:
            d.rd = torch.tile(rd[0, :].view([-1, 1]), [1, 2])
        d.center_idx = torch.tensor([[0], [0]]).long().T
        return d

    if d.x != None:
        if d.x.ndim == 2:
            x = torch.tile(d.x, [num_neighbors, 1])
        elif d.x.ndim == 1:
            x = torch.tile(d.x, [num_neighbors])
        else:
            exit(1)
    if d.edge_attr != None:
        if d.edge_attr.ndim == 2:
            edge_attr = torch.tile(d.edge_attr, [num_neighbors, 1])
        elif d.edge_attr.ndim == 1:
            edge_attr = torch.tile(d.edge_attr, [num_neighbors])
        else:
            exit(1)
    if d.pos != None:
        pos = torch.tile(d.pos, [num_neighbors, 1])
    z = torch.zeros_like(d.z)
    if node_label.startswith('spd'):
        z = torch.tile(z, [num_neighbors, 2])
    else:
        z = torch.tile(z, [num_neighbors, 1])
    edge_index = torch.zeros([2, d.num_edges * num_neighbors]).long()
    # node_to_subgraph_node = torch.arange(d.num_nodes)
    # node_to_subgraph_node = torch.tile(node_to_subgraph_node, [num_neighbors]) # (i,j,k) to (i,k)
    node_to_subgraph2 = torch.zeros([d.num_nodes * num_neighbors]).long() # (i,j,k) to (i,j)

    center_idx = [[], []]
    for i, n in enumerate(neighbors):
        # update labeling z
        if node_label.startswith('spd'):
            z_n = find_all_spd(n, edge_index=d.edge_index, node_label=node_label, num_nodes=d.num_nodes)
            z[i * d.num_nodes : (i + 1) * d.num_nodes] = torch.cat([d.z, z_n+num_hops+3], dim=-1) # +num_hops+3 such that it uses different distanc embedding
        else:
            z_n = d.z.clone()
            z_n[n] = 2
            z[i * d.num_nodes : (i + 1) * d.num_nodes] = z_n

        # update rd
        if use_rd:
            rd_to_x[i * d.num_nodes : (i + 1) * d.num_nodes] = torch.cat([rd[0, :].view([-1, 1]),
                                                                          rd[n, :].view([-1, 1])], dim=-1)
        # update edge index
        temp = d.edge_index + i * d.num_nodes
        edge_index[:, i * d.num_edges : (i + 1) * d.num_edges] = temp

        node_to_subgraph2[i * d.num_nodes : (i + 1) * d.num_nodes] = i * torch.ones([d.num_nodes])

        center_idx[0].append(0 + d.num_nodes * i)
        center_idx[1].append(n.item() + d.num_nodes * i) # center node id and side node id
    # combine subgraph-2 into a subgraph data
    subgraphs_2 = Data(edge_index=edge_index, z=z, node_to_subgraph2=node_to_subgraph2,
                       subgraph2_to_subgraph=torch.zeros([num_neighbors]).long(), num_subgraphs2=num_neighbors,
                       center_idx=torch.tensor(center_idx).long().T)
    # subgraphs_2 = Data(edge_index=edge_index, z=z)
    if d.x != None:
        subgraphs_2.x = x
    if d.edge_attr != None:
        subgraphs_2.edge_attr = edge_attr
    if d.pos != None:
        subgraphs_2.pos = pos
    if use_rd:
        subgraphs_2.rd = rd_to_x
    # subgraphs_2 = Data(x=x, edge_index=edge_index, pos=pos, edge_attr=edge_attr, z=z, node_to_subgraph_node=node_to_subgraph_node)
    return subgraphs_2


def subgraph_to_subgraph2(d, num_hops, center=0, node_label='spd', use_rd=False):
    # transform a pyg-data into multiple subgraphs by labeling neighbors
    neighbors = d.edge_index[1][torch.where(d.edge_index[0] == center)[0]] # find neighbors of ind node
    num_neighbors = len(neighbors)
    if use_rd:
        rd_to_x = torch.zeros([d.num_nodes * num_neighbors, 2])
        rd = compute_rd(d.edge_index, d.num_nodes)
    if num_neighbors == 0:
        d.node_to_subgraph2 = torch.zeros([d.num_nodes]).long()
        d.subgraph2_to_subgraph = torch.zeros([1]).long()
        d.num_subgraphs2 = 1
        if node_label.startswith('spd'):
            d.z = torch.tile(d.z, [1, 2])
        if node_label.startswith('dspd'):
            d.z = torch.tile(d.z, [1, 4])
        if use_rd:
            d.rd = torch.tile(rd[0, :].view([-1, 1]), [1, 2])
        return d

    if d.x != None:
        if d.x.ndim == 2:
            x = torch.tile(d.x, [num_neighbors, 1])
        elif d.x.ndim == 1:
            x = torch.tile(d.x, [num_neighbors])
        else:
            exit(1)
    if d.edge_attr != None:
        if d.edge_attr.ndim == 2:
            edge_attr = torch.tile(d.edge_attr, [num_neighbors, 1])
        elif d.edge_attr.ndim == 1:
            edge_attr = torch.tile(d.edge_attr, [num_neighbors])
        else:
            exit(1)
    if d.pos != None:
        pos = torch.tile(d.pos, [num_neighbors, 1])
    z = torch.zeros_like(d.z)
    if node_label.startswith('spd'):
        z = torch.tile(z, [num_neighbors, 2])
    elif node_label.startswith('dspd'):
        z = torch.tile(z, [num_neighbors, 4])
    else:
        z = torch.tile(z, [num_neighbors, 1])
    edge_index = torch.zeros([2, d.num_edges * num_neighbors]).long()
    # node_to_subgraph_node = torch.arange(d.num_nodes)
    # node_to_subgraph_node = torch.tile(node_to_subgraph_node, [num_neighbors]) # (i,j,k) to (i,k)
    node_to_subgraph2 = torch.zeros([d.num_nodes * num_neighbors]).long() # (i,j,k) to (i,j)
    for i, n in enumerate(neighbors):
        # update labeling z
        if node_label.startswith('spd'):
            z_n = find_all_spd(n, edge_index=d.edge_index, node_label=node_label, num_nodes=d.num_nodes)
            z[i * d.num_nodes : (i + 1) * d.num_nodes] = torch.cat([d.z, z_n+num_hops+3], dim=-1) # +num_hops+3 such that it uses different distanc embedding
        elif node_label.startswith('dspd'):
            z_n = find_all_spd(n, edge_index=d.edge_index, node_label=node_label, num_nodes=d.num_nodes)
            z_n_delete_0 = find_all_spd(n, edge_index=d.edge_index, node_label=node_label, num_nodes=d.num_nodes,
                                        delete_node=0)
            z_0_delete_n = find_all_spd(0, edge_index=d.edge_index, node_label=node_label, num_nodes=d.num_nodes,
                                        delete_node=n.item())
            delta = num_hops + 5 # make distance embedding different
            z[i * d.num_nodes: (i + 1) * d.num_nodes] = torch.cat([d.z, z_n + delta, z_n_delete_0 + 2 * delta,
                                                                   z_0_delete_n + 3 * delta], dim=-1)
        else:
            z_n = d.z.clone()
            z_n[n] = 2
            z[i * d.num_nodes : (i + 1) * d.num_nodes] = z_n

        # update rd
        if use_rd:
            rd_to_x[i * d.num_nodes : (i + 1) * d.num_nodes] = torch.cat([rd[0, :].view([-1, 1]),
                                                                          rd[n, :].view([-1, 1])], dim=-1)
        # update edge index
        temp = d.edge_index + i * d.num_nodes
        edge_index[:, i * d.num_edges : (i + 1) * d.num_edges] = temp

        node_to_subgraph2[i * d.num_nodes : (i + 1) * d.num_nodes] = i * torch.ones([d.num_nodes])
    # combine subgraph-2 into a subgraph data
    subgraphs_2 = Data(edge_index=edge_index, z=z, node_to_subgraph2=node_to_subgraph2,
                       subgraph2_to_subgraph=torch.zeros([num_neighbors]).long(), num_subgraphs2=num_neighbors)
    # subgraphs_2 = Data(edge_index=edge_index, z=z)
    if d.x != None:
        subgraphs_2.x = x
    if d.edge_attr != None:
        subgraphs_2.edge_attr = edge_attr
    if d.pos != None:
        subgraphs_2.pos = pos
    if use_rd:
        subgraphs_2.rd = rd_to_x
    # subgraphs_2 = Data(x=x, edge_index=edge_index, pos=pos, edge_attr=edge_attr, z=z, node_to_subgraph_node=node_to_subgraph_node)
    return subgraphs_2


def colored(r, g, b, text):
    return "\033[38;2;{};{};{}m{} \033[38;2;255;255;255m".format(r, g, b, text)


def pyg2nx(data):
    # torch_geometric data to networkx
    G = nx.Graph()
    G.add_nodes_from([i for i in range(data.num_nodes)])
    edges = [(int(edge[0]), int(edge[1])) for edge in data.edge_index.T]
    G.add_edges_from(edges)
    return G

from networkx.algorithms import isomorphism
def count_graphlet(G, target):
    # G is a networkx graph
    # target: 0 for tailed triangle, 1 for chordal cycle, 2 for 4-clique, 3 for 4-path and 4 for triangle-rectangle
    y = torch.zeros(len(G))
    H = nx.Graph()
    if target == 0:
        H.add_edges_from([('*', 1), (1, 2), (1, 3), (2, 3)])
        factor = 1 / 2
    elif target == 1:
        H.add_edges_from([('*', 1), ('*', 2), (1, 2), (1, 3), (2, 3)])
        factor = 1 / 2
    elif target == 2:
        H.add_edges_from([('*', 1), ('*', 2), ('*', 3), (1, 2), (1, 3), (2, 3)])
        factor = 1 / 3
    elif target == 3:
        H.add_edges_from([('*', 1), (1, 2), (2, 3), (3, 4)])
        factor = 1
    elif target == 4:
        H.add_edges_from([('*', 1), ('*', 2), (1, 2), (2, 3), (3, 4), (1, 4)])
        factor = 1 / 2

    GM = isomorphism.GraphMatcher(G, H)
    for map in GM.subgraph_monomorphisms_iter():  # subgraph counting
        node_idx = list(map.keys())[list(map.values()).index('*')]
        y[node_idx] += 1
    return y * factor # remove repeated self-isomorphism


def check_graphlet(dataset, target):
    for i, data in enumerate(dataset):
        if i % 500 == 0:
            print('checking ground truth labels...(%d/%d)'%(i, len(dataset)))
        G = pyg2nx(data)
        y = count_graphlet(G, target)
        assert torch.sum(torch.abs(y - data.y)) < 1e-9





