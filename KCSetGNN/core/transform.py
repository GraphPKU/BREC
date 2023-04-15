import torch
from torch_geometric.transforms import BaseTransform
from torch_geometric.data import Data
from torch_sparse import SparseTensor

import re
class SubgraphsData(Data):
    r""" A data object describing a homogeneous graph together with each node's rooted subgraph. 
    It contains several additional propreties that hold the information of all nodes' rooted subgraphs.
    Assume the data represents a graph with :math:'N' nodes and :math:'M' edges, also each node 
    :math:'i\in \[N\]' has a rooted subgraph with :math:'N_i' nodes and :math:'M_i' edges.
    
    Additional Properties:
        subgraphs_nodes_mapper (LongTensor): map each node in rooted subgraphs to a node in the original graph.
            Size: :math:'\sum_{i=1}^{N}N_i x 1'
        subgraphs_edges_mapper (LongTensor): map each edge in rooted subgraphs to a edge in the original graph.
            Size: :math:'\sum_{i=1}^{N}M_i x 1'
        subgraphs_batch: map each node in rooted subgraphs to its corresponding rooted subgraph index. 
            Size: :math:'\sum_{i=1}^{N}N_i x 1'
        combined_rooted_subgraphs: edge_index of a giant graph which represents a stacking of all rooted subgraphs. 
            Size: :math:'2 x \sum_{i=1}^{N}M_i'
        
    The class works as a wrapper for the data with these properties, and automatically handles mini batching for
    them. 
    
    """
    # TODO: revise later
    def __inc__(self, key, value, *args, **kwargs):
        num_nodes = self.num_nodes
        num_edges = self.edge_index.size(-1)
        if 'combined_subgraphs'in key:
            return getattr(
                self, key[:-len('combined_subgraphs')] +
                'subgraphs_nodes_mapper').size(0)
        elif 'subgraphs_batch' in key:
            # should use number of subgraphs or number of supernodes.
            return 1 + getattr(self, key)[-1]
        elif bool(re.search('(nodes_mapper)|(selected_supernodes)', key)):
            return num_nodes
        elif 'edges_mapper' in key:
            return num_edges
        elif 'k_to_kplus1' in key:
            return getattr(self, key).max(dim=1,  keepdim=True)[0] + 1
        elif 'bipartite' in key:
            idx = int(key[-1])
            inc = torch.cat([self.num_ks[idx:idx+2], torch.tensor([num_nodes])]).unsqueeze(1)
            return inc 
        elif 'components_graph' in key:
            return self.num_components.size(0)
        else:
            return super().__inc__(key, value, *args, **kwargs)

    def __cat_dim__(self, key, value, *args, **kwargs):
        if bool(re.search('(combined_subgraphs)|(bipartite)|(components_graph)', key)):
            return -1
        else:
            return super().__cat_dim__(key, value, *args, **kwargs)


def to_sparse(node_mask, edge_mask):
    subgraphs_nodes = node_mask.nonzero().T
    subgraphs_edges = edge_mask.nonzero().T
    return subgraphs_nodes, subgraphs_edges

def combine_subgraphs(edge_index, subgraphs_nodes, subgraphs_edges,
                      num_selected=None, num_nodes=None):
    if num_selected is None:
        num_selected = subgraphs_nodes[0][-1] + 1
    if num_nodes is None:
        num_nodes = subgraphs_nodes[1].max() + 1

    combined_subgraphs = edge_index[:, subgraphs_edges[1]]
    node_label_mapper = edge_index.new_full((num_selected, num_nodes), -1)
    node_label_mapper[subgraphs_nodes[0], subgraphs_nodes[1]] = torch.arange(
        len(subgraphs_nodes[1]))
    node_label_mapper = node_label_mapper.reshape(-1)

    inc = torch.arange(num_selected) * num_nodes
    combined_subgraphs += inc[subgraphs_edges[0]]
    combined_subgraphs = node_label_mapper[combined_subgraphs]
    return combined_subgraphs

class Subgraphs(BaseTransform):
    r"""
    Base class for subgraphs.
    The object transforms a Data object to RootedSubgraphsData object. 
    """
    def __init__(self):
        super().__init__()

    def extract_subgraphs(self, data: Data):
        r""" For a input graph with N nodes, extract S subgraphs
        Return:
            subgraphs_nodes_mask: S x N dense mask matrix, each row indicates a subgraph 
            subgraphs_nodes:      S x k matrix, each row indicates a subgraph with k nodes 
        """
        raise NotImplementedError

    def __call__(self, data: Data) -> Data:
        subgraphs_nodes_mask, bipartite_graph = self.extract_subgraphs(data) # current asasume only one bipartite graph
        # extract edges that involves these nodes 
        subgraphs_edges_mask = subgraphs_nodes_mask[:, data.edge_index[0]] & \
                               subgraphs_nodes_mask[:, data.edge_index[1]]  # S x E dense mask matrix

        subgraphs_nodes, subgraphs_edges = to_sparse(subgraphs_nodes_mask, subgraphs_edges_mask)
        combined_subgraphs = combine_subgraphs(data.edge_index,
                                               subgraphs_nodes,
                                               subgraphs_edges,
                                               num_nodes=data.num_nodes)

        data = SubgraphsData(**{k: v for k, v in data})
        data.subgraphs_batch = subgraphs_nodes[0]
        data.subgraphs_nodes_mapper = subgraphs_nodes[1]
        data.subgraphs_edges_mapper = subgraphs_edges[1]
        data.combined_subgraphs = combined_subgraphs
        data.__num_nodes__ = data.num_nodes
        # data.num_subgraphs = data.subgraphs_batch[-1] + 1

        if isinstance(bipartite_graph, tuple):
            if len(bipartite_graph) == 3:
                counts, bipartites, num_components = bipartite_graph
            else:
                # record components graph: every set with multiple components has connections 
                # for a set with multiple components (nc), it will connects to nc different 
                # sets, and each this kind of set has only one component. 
                counts, bipartites, num_components, components_graph = bipartite_graph
                data.components_graph = components_graph
                
            data.num_components = num_components.long()  # use integer to represent it 

            for i, bipartite in enumerate(bipartites):
                setattr(data, f'bipartite_{i}', bipartite)
            # data.ks = torch.repeat_interleave(torch.arange(len(counts)), counts)
            data.num_ks = counts
        else:
            data.k_to_kplus1 = bipartite_graph

        return data


class KCSetWLSubgraphs(Subgraphs):
    def __init__(self, k_max=5, stack=True, k_min=0, num_components=1, zero_init=False):
        super().__init__()
        self.k_max = k_max
        self.stack = stack
        self.k_min = k_min 
        self.max_components = num_components
        self.zero_init=zero_init

    def extract_subgraphs(self, data):
        # assert self.k_max < data.num_nodes
        # k = max(2, min(self.k_max, data.num_nodes-1)) # deal with k_max (can later set it as a fraction of number of nodes)
        k_sets, bipartite_graph = extract_k_sets(data.edge_index, data.num_nodes, self.k_max, self.k_min, self.stack, self.max_components, self.zero_init)
        assert len(k_sets) > 0
        # print(bipartite_graph.shape)
        node_mask = data.edge_index.new_empty((len(k_sets), data.num_nodes), dtype=torch.bool)
        node_mask.fill_(False)
        row_idx = torch.arange(len(k_sets), device=data.edge_index.device)
        row_idx = row_idx.unsqueeze(1).repeat(1, k_sets.size(1)).reshape(-1)
        node_mask[row_idx, k_sets.reshape(-1)] = True
        return node_mask, bipartite_graph

def extract_k_sets(edge_index, num_nodes, k, k_min, return_previous=False, max_components=1, zero_init=False):
    # assert k >=2
    sparse_adj = SparseTensor(row=edge_index[0], col=edge_index[1], sparse_sizes=(num_nodes, num_nodes))
    k_sets = torch.arange(num_nodes).unsqueeze(0)
    num_components_k = torch.ones(num_nodes)
    kminus1_to_k = torch.cat([k_sets, k_sets, k_sets]) # self connection
    all_bipartites = []
    backtracks = []
    all_num_components = [] # TODO: currently only support kmin = 0 when considering multiple components!
    for i in range(k-1):
        if i == k_min:
            counts = [k_sets.size(1)]
            all = k_sets
            all_bipartite = kminus1_to_k
            all_num_components = [num_components_k]

        kplus1_sets, k_to_kplus1, num_components_k, backtrack_components_inc = extend_to_kplus1(k_sets, num_components_k, sparse_adj, max_components) 
        ##### deal with boundary cases 
        if  kplus1_sets is None:
            kplus1_sets = k_sets
            k_to_kplus1 = kminus1_to_k
            break 
        ##############################
        k_sets, kminus1_to_k = kplus1_sets, k_to_kplus1

        if i >= k_min: # >=3 sets
            all = torch.cat([torch.cat([all,all[0].unsqueeze(0)],dim=0), kplus1_sets], dim=-1)
            update = torch.clone(k_to_kplus1)
            update[:2] += 1 + all_bipartite[:2].max(dim=1, keepdim=True)[0]
            # update = all_bipartite.max(dim=1, keepdim=True)[0] + k_to_kplus1 + 1
            all_bipartite = torch.cat([all_bipartite, update], dim=1) # this directly stack together
            counts.append(kplus1_sets.size(1))
            all_bipartites.append(k_to_kplus1)
            backtracks.append(backtrack_components_inc)
            all_num_components.append(num_components_k)

    if return_previous:
        # now add another part to deal with sets with num_components > 1
        if k_min == 0 and max_components>1 and not zero_init:
            if len(backtracks) > 0:
                components_graph = build_components_graph_parallel(backtracks, all_bipartites, all_num_components, max_components)
                components_graph = components_graph.long()
            else:
                components_graph = torch.empty(size=(2,0), dtype=torch.long)

        all_num_components = torch.cat(all_num_components)
        m = k - 1 - len(all_bipartites) 
        all_bipartites = all_bipartites + [torch.empty(size=(3,0), dtype=torch.long)]*m
        counts = torch.tensor(counts + [0]*m, dtype=torch.long)
 
        if k_min == 0 and max_components>1 and not zero_init:
            return all.T[all_num_components==1], (counts, all_bipartites, all_num_components, components_graph)
        else:
            return all.T[all_num_components==1], (counts, all_bipartites, all_num_components)
            # return all.T, (counts, all_bipartites, all_num_components)
        # return all.T, all_bipartite
    else:
        # only take the last bipartite
        counts = torch.tensor(counts[-2:], dtype=torch.long)
        all_num_components = torch.cat(all_num_components[-2:])
        all_bipartites = all_bipartites[-1:]

        return all.T[-counts.sum():], (counts, all_bipartites, all_num_components)


def extend_to_kplus1(k_sets, num_components_k, sparse_adj, max_components=1):
    # k_sets: k x num_sets 
    # sparse_adj: n x n
    k, num_sets = k_sets.shape
    n = sparse_adj.size(0)
    col_idx = torch.arange(num_sets).unsqueeze(0).repeat(k,1) 

    # propagate
    k_sets_mask = torch.zeros(n, num_sets) # n x num_sets
    k_sets_mask[k_sets.reshape(-1), col_idx.reshape(-1)] = 1
    # 0:disconnected nodes, 1:direct neighbors, 2:used nodes
    next_node_distance = (sparse_adj.matmul(k_sets_mask) > 0).float()
    next_node_distance[k_sets.reshape(-1), col_idx.reshape(-1)] = 2 
    
    # get direct neigbors: number of components will keep the same or reduce
    next_node, set_idx = (next_node_distance==1).nonzero().T
    component_inc = torch.zeros(len(next_node)) # 0 means not increase or reduce

    if k >= max_components:
        # considering these disconnected nodes, but with number of components less than max_components
        # first thing to do is to mask out these sets with max_components part. 
        mask = (num_components_k >= max_components)
        next_node_distance[:, mask] = 3

    # disconnected noddes neighbors: number of components +1
    next_disconnected_node, dis_set_idx = (next_node_distance==0).nonzero().T
    if len(next_disconnected_node) > 0:
        next_node = torch.cat([next_node, next_disconnected_node])
        set_idx = torch.cat([set_idx, dis_set_idx])
        component_inc = torch.cat([component_inc, torch.ones(len(next_disconnected_node))])

    num_components_k = num_components_k[set_idx]

    #### remark here: when k>= max_components, inc can only be 0!! However this is not correct. 
    # So we need to add these kind of connections back from k >= max_components case. 
    
    if len(next_node) == 0: ## happens if no expanding can be find 
        return None, None, None, None 

    k_plus_1_sets = torch.cat([k_sets[:,set_idx], next_node.unsqueeze(0)], dim=0)
    # print(k_plus_1_sets.shape)
    # order sets 
    k_plus_1_sets, _ = torch.sort(k_plus_1_sets, dim=0) # order along k dim
    # remove duplicated ones
    k_plus_1_sets, inverse = torch.unique(k_plus_1_sets, dim=1, return_inverse=True)    
    # print(k_plus_1_sets.shape)
    # construct connections between k sets and k+1 sets
    next_set_idx = torch.arange(k_plus_1_sets.size(1))[inverse]
    k_to_kplus1_bipartite_graph = torch.stack([set_idx, next_set_idx, next_node]) # many to many bipartite

    # for every k+1 set (with >1 #components), backward to construct a mapping from some (any) k1 set to the k+1 set
    # record (idx of the k+1 set, idx of the k set, )
    temp = torch.cat([k_to_kplus1_bipartite_graph, num_components_k.unsqueeze(0), component_inc.unsqueeze(0)]).T
    backtrack_components_inc = []

    #TODO: think how to remove the for loop
    for i in range(k_plus_1_sets.size(1)):
        m = temp[inverse==i] 
        # first check whether exist inc = 1, choose this kind of first
        # if k < max_components:
        idx = m[:,-1].argmax()
        if m[idx,-1] > 0:
            backtrack_components_inc.append(m[idx])
        else:
            # if not then choose min num_components_k
            # this avoid to consider num_components reduced case
            backtrack_components_inc.append(m[m[:,-2].argmin()])

    backtrack_components_inc = torch.stack(backtrack_components_inc)
    # compute num_components k+1
    backtrack_components_inc[:, -2] += backtrack_components_inc[:, -1]
    num_components_kplus1 = backtrack_components_inc[:, -2]

    return k_plus_1_sets, k_to_kplus1_bipartite_graph, num_components_kplus1, backtrack_components_inc[num_components_kplus1>1]   # k+1 x num_sets

def build_components_graph_parallel(backtracks, all_bipartities, all_num_components, max_components):
    # get number of sets at each level
    all_num_sets = [len(x) for x in all_num_components]
    num_nodes = all_num_sets[0] # number of nodes = number of 1-sets
    # construct mapper 
    all_mappers = torch.zeros(len(all_bipartities), max(all_num_sets), num_nodes, dtype=torch.long) - 1 # TODO: change this to sparse matrix [to further improve speed]
    for i, bipartite in enumerate(all_bipartities):
        # step 1: filter bipartite to keep only component=1 sets 
        single_component_idx = (all_num_components[i]==1)[bipartite[0]] & (all_num_components[i+1]==1)[bipartite[1]]
        # construct the mapper
        bipartite = bipartite[:, single_component_idx]
        all_mappers[i, bipartite[0], bipartite[2]] = bipartite[1]

    all_components_info = []
    # construct memory saver for the output
    k_components_info = torch.zeros(num_nodes, max_components, 2, dtype=torch.long) - 1 # 2: one for which k, one for idx of that k
    k_components_info[:, 0, 0] = 0
    k_components_info[:, 0, 1] = torch.arange(num_nodes)

    for i, k_to_kplus1 in enumerate(backtracks):
        k_to_kplus1 = k_to_kplus1.long()
        # init memory for next info
        kplus1_components_info = torch.zeros(all_num_sets[i+1], max_components, 2, dtype=torch.long) - 1
        single_comp_idx = (all_num_components[i+1] == 1).nonzero().squeeze()
        kplus1_components_info[single_comp_idx, 0, 0] = i+1
        kplus1_components_info[single_comp_idx, 0, 1] = single_comp_idx

        # need_to_infer_idx = (all_num_components[i+1] > 1).nonzero().squeeze()
        # assert len(need_to_infer_idx)==len(k_to_kplus1)
   
        k_idx, kplus1_idx, next_node, kplus1_nc, nc_inc = k_to_kplus1.T
        # step 1: copy directly from previous 
        kplus1_components_info[kplus1_idx,:,:] = k_components_info[k_idx,:,:]
        # step 2: update nc_inc = 1 case 
        tmp_idx = (nc_inc==1)
        if len(tmp_idx)>0:
            kplus1_components_info[kplus1_idx[tmp_idx], kplus1_nc[tmp_idx]-1, :] = torch.stack([torch.zeros(tmp_idx.sum()).long() , next_node[tmp_idx]]).T

        # step 3: update nc_inc != 1 case 
        # check every second dim 
        left_part = k_to_kplus1[nc_inc==0]
        if len(left_part)>0:
            for ii in range(left_part[:,3].max()):
                # k_idx, kplus1_idx, next_node, kplus1_nc, nc_inc = k_to_kplus1[left_idx].T
                active_idx = (kplus1_components_info[left_part[:,1], ii, 0] != -1)
                left_part = left_part[active_idx]
                if len(left_part)==0:
                    break

                a, b = kplus1_components_info[left_part[:,1], ii].T
                target = all_mappers[a, b, left_part[:,2]]
                tmp_idx = left_part[:,1][target >= 0]

                if len(tmp_idx)>0:
                    kplus1_components_info[tmp_idx, ii, 0] += 1
                    kplus1_components_info[tmp_idx, ii, 1] = target[target >= 0]

                # every row can only be updated one time
                left_part = left_part[target<0]
                if len(left_part)==0:
                    break

        all_components_info.append(kplus1_components_info)
        k_components_info = kplus1_components_info

    # transform to edges
    offsets = torch.cumsum(torch.tensor(all_num_sets), dim=0)
    offsets = torch.cat([torch.zeros(1), offsets, torch.zeros(1)])
    edges = torch.cat([offsets[x[:,:, 0]] + x[:,:, 1] for x in all_components_info]) # num_sets x max_c

    # filter out num_components = 1 case 
    nc = (edges!=-1).sum(-1)
    assert nc.min() == 1
    mask = (edges != -1)
    mask[nc==1] = 0

    right, c =  mask.nonzero().T
    left = edges[right, c]
    right += num_nodes 
    # filter out num_components = 1 case 
    return torch.stack([left, right])

class RootedEgoNets(Subgraphs):
    """ Record rooted k-hop Egonet for each node in the graph.
    From the `"From Stars to Subgraphs: Uplifting Any GNN with Local Structure Awareness"
    <https://arxiv.org/pdf/2110.03753.pdf>`_ paper
    Args:
        hops (int): k for k-hop Egonet. 
    """
    def __init__(self, hops: int, stack=False):
        super().__init__()
        self.num_hops = hops
        self.stack = stack

    def extract_subgraphs(self, data: Data):
        # return k-hop subgraphs for all nodes in the graph
        row, col = data.edge_index
        sparse_adj = SparseTensor(
            row=row, col=col, sparse_sizes=(data.num_nodes, data.num_nodes))
        hop_mask = sparse_adj.to_dense() > 0
        hop_indicator = torch.eye(data.num_nodes, dtype=torch.long, device=data.edge_index.device) - 1

        all_masks = []
        for i in range(self.num_hops):
            hop_indicator[(hop_indicator == -1) & hop_mask] = i + 1
            hop_mask = sparse_adj.matmul(hop_mask.float()) > 0
            all_masks.append((hop_indicator >= 0).T) 

        hop_indicator = hop_indicator.T  # N x N
        if self.stack:
            node_mask = torch.cat(all_masks, dim=0)
        else:
            node_mask = all_masks[-1]  # N x N dense mask matrix

        return node_mask, None

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}({self.hops})'
