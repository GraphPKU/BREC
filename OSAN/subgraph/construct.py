from typing import Tuple, List, Optional

from torch import Tensor
from torch_geometric.data import Data, Batch
from torch_geometric.utils import subgraph

from data import SubgraphSetBatch
from data.const import MAX_NUM_NODE_DICT
from subgraph.grad_utils import *


def ordered_subgraph_construction(dataset: str,
                                  graphs: List[Data],
                                  masks: Tensor,
                                  sorted_indices: List[Tensor],
                                  add_full_graph: bool,
                                  remove_node: bool,
                                  grad: bool):
    """

    @param dataset:
    @param graphs:
    @param masks:
    @param sorted_indices:
    @param add_full_graph:
    @param remove_node:
    @param grad:
    @return:
    """
    num_subgraphs = masks.shape[1]
    device = masks.device

    for g in graphs:
        g.extra_feature = torch.zeros(g.num_nodes, MAX_NUM_NODE_DICT[dataset], device=device, dtype=torch.float)

    ret_graphs = graphs * (num_subgraphs + int(add_full_graph))

    for k in range(num_subgraphs):
        for i, (graph, sorted_index) in enumerate(zip(graphs, sorted_indices)):
            idx = sorted_index[:, k]
            ret_graphs[k * len(graphs) + i].extra_feature[idx, :idx.numel()] = graph.adj[idx, :][:, idx]

    batch = Batch.from_data_list(ret_graphs, None, None)
    original_graph_mask = torch.repeat_interleave(torch.arange(len(graphs)),
                                                  torch.tensor([g.num_nodes for g in graphs]), dim=0).to(device)
    original_node_mask = torch.arange(sum([g.num_nodes for g in graphs]), device=device)\
        .repeat(num_subgraphs + int(add_full_graph))

    if grad:
        if remove_node:
            selected_node_masks = IdentityMapping.apply(masks)
        else:
            selected_node_masks = CustomedIdentityMapping.apply(masks)
    else:
        if remove_node:
            selected_node_masks = masks
        else:
            selected_node_masks = torch.ones_like(masks, device=masks.device, dtype=masks.dtype)

    if add_full_graph:
        selected_node_masks = torch.cat([selected_node_masks,
                                         torch.ones(selected_node_masks.shape[0], 1,
                                                    dtype=selected_node_masks.dtype,
                                                    device=device)], dim=-1)
    selected_node_masks = selected_node_masks.T.reshape(-1)

    return SubgraphSetBatch(x=batch.x,
                            extra_feature=batch.extra_feature,
                            edge_index=batch.edge_index,
                            edge_attr=batch.edge_attr,
                            selected_node_masks=selected_node_masks,
                            y=batch.y[:len(graphs)],
                            inter_graph_idx=original_graph_mask,
                            original_node_mask=original_node_mask,
                            num_graphs=batch.num_graphs)


def nodesubset_to_subgraph(graph: Data, subset: Tensor, relabel=False) -> Data:
    edge_index, edge_attr = subgraph(subset, graph.edge_index, graph.edge_attr,
                                     relabel_nodes=relabel, num_nodes=graph.num_nodes)

    x = graph.x[subset] if relabel else graph.x
    if relabel:
        if subset.dtype in [torch.bool, torch.uint8]:
            num_nodes = subset.sum()
        else:
            num_nodes = subset.numel()
    else:
        num_nodes = graph.num_nodes

    return Data(x=x,
                edge_index=edge_index,
                edge_attr=edge_attr,
                num_nodes=num_nodes,
                y=graph.y)


def edgemasked_graphs_from_nodemask(graphs: List[Data] = None,
                                    edge_index: Tensor = None,
                                    masks: Tensor = None,
                                    grad: bool = True,
                                    remove_node: bool = False,
                                    add_full_graph: bool = False) \
        -> Tuple[List[Data], Tensor, Optional[Tensor]]:
    """

    :param graphs:
    :param edge_index:
    :param masks:
    :param grad:
    :param remove_node:
    :param add_full_graph:
    :return:
    """
    num_nodes, num_subgraphs = masks.shape
    num_edges = edge_index.shape[1]

    graphs = graphs * num_subgraphs if not add_full_graph else graphs * (num_subgraphs + 1)

    transform_func = Nodemask2Edgemask.apply if grad else nodemask2edgemask
    edge_weights = transform_func(masks.T, edge_index, torch.tensor(num_nodes, device=masks.device))

    edge_weights = edge_weights.reshape(-1)
    if add_full_graph:
        edge_weights = torch.cat((edge_weights,
                                  torch.ones(num_edges, dtype=edge_weights.dtype, device=edge_weights.device)), dim=0)
    if remove_node:
        selected_node_masks = masks.T.reshape(-1)
        if add_full_graph:
            selected_node_masks = torch.cat((selected_node_masks,
                                             torch.ones(num_nodes, dtype=masks.dtype, device=masks.device)),
                                            dim=0)
    else:
        selected_node_masks = None

    return graphs, edge_weights, selected_node_masks


def edgemasked_graphs_from_directed_edgemask(graphs: List[Data] = None,
                                             edge_index: Tensor = None,
                                             masks: Tensor = None,
                                             grad: bool = True,
                                             add_full_graph: bool = False, **kwargs) \
        -> Tuple[List[Data], Tensor, Optional[Tensor]]:
    """
    Given masks of directed edge masks, get undirected ones as well as backprop
    """
    _, num_subgraphs = masks.shape
    num_edges = edge_index.shape[1]

    graphs = graphs * num_subgraphs if not add_full_graph else graphs * (num_subgraphs + 1)

    transform_func = DirectedEdge2UndirectedEdge.apply if grad else directedge2undirectedge
    edge_weights = transform_func(masks, edge_index)

    edge_weights = edge_weights.reshape(-1)
    if add_full_graph:
        edge_weights = torch.cat((edge_weights,
                                  torch.ones(num_edges, dtype=edge_weights.dtype, device=edge_weights.device)), dim=0)

    return graphs, edge_weights, None


def edgemasked_graphs_from_undirected_edgemask(graphs: List[Data] = None,
                                               masks: Tensor = None,
                                               add_full_graph: bool = False,
                                               **kwargs) \
        -> Tuple[List[Data], Tensor, Optional[Tensor]]:
    """
    Create edge_weights which contain the back-propagated gradients

    :param graphs:
    :param masks: shape (n_edge_in_original_graph, n_subgraphs) edge masks
    :param add_full_graph:
    :return:
    """
    num_edges, num_subgraphs = masks.shape

    graphs = graphs * num_subgraphs if not add_full_graph else graphs * (num_subgraphs + 1)
    masks = masks.T.reshape(-1)
    if add_full_graph:
        masks = torch.cat((masks, torch.ones(num_edges, device=masks.device, dtype=masks.dtype)), dim=0)

    return graphs, masks, None


def construct_subgraph_batch(graph_list: List[Data],
                             num_graphs: int,
                             num_subgraphs: int,
                             edge_weights: Tensor,
                             selected_node_masks: Optional[Tensor] = None,
                             device: torch.device = torch.device('cpu')):
    """

    :param graph_list: a list of [subgraph1_1, subgraph2_1, subgraph3_1, subgraph1_2, subgraph2_2, ...]
    :param num_graphs:
    :param num_subgraphs:
    :param edge_weights:
    :param selected_node_masks:
    :param device:
    :return:
    """
    # new batch
    batch = Batch.from_data_list(graph_list, None, None)
    original_graph_mask = torch.arange(num_graphs, device=device).repeat(num_subgraphs)

    # TODO: duplicate labels or aggregate the embeddings for original labels? potential problem: cannot split the
    #  batch because y shape inconsistent:
    #  https://pytorch-geometric.readthedocs.io/en/latest/modules/data.html#torch_geometric.data.Batch.to_data_list.
    #  need to check `batch._slice_dict` and `batch._inc_dict`

    batch_idx = batch.batch
    if selected_node_masks is not None:
        if selected_node_masks.dtype == torch.float:
            pass
        elif selected_node_masks.dtype == torch.bool:
            batch_idx = batch.batch[selected_node_masks]
        else:
            raise ValueError

    return SubgraphSetBatch(x=batch.x,
                            edge_index=batch.edge_index,
                            edge_attr=batch.edge_attr,
                            edge_weight=edge_weights,
                            selected_node_masks=selected_node_masks,
                            y=batch.y[:num_graphs],
                            batch=batch_idx,
                            inter_graph_idx=original_graph_mask,
                            num_graphs=batch.num_graphs)
