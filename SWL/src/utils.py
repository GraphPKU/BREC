from . import *


def subgraph(scheme):
    uv = "uv" in scheme
    vu = "vu" in scheme
    uu = "uu" in scheme
    vv = "vv" in scheme
    uG = "uG" in scheme
    vG = "vG" in scheme
    uL = "uL" in scheme
    vL = "vL" in scheme

    def call(graph):
        node = torch.arange(graph.num_nodes**2).view(
            (graph.num_nodes, graph.num_nodes)
        )

        adj = pyg.utils.to_dense_adj(
            graph.edge_index, max_num_nodes=graph.num_nodes
        ).squeeze(0)
        spd = torch.where(
            ~torch.eye(len(adj), dtype=bool) & (adj == 0),
            torch.full_like(adj, float("inf")),
            adj,
        )
        # Floyd-Warshall
        for k in range(len(spd)):
            spd = torch.minimum(spd, spd[:, [k]] + spd[[k], :])

        return data.Data(
            x=graph.x[None, :, 0].expand(len(node), -1).flatten(end_dim=1),
            d=spd.to(int).flatten(end_dim=1),
            y=graph.y,
            # point
            index_uv=uv and torch.stack((node, node)).flatten(start_dim=1),
            index_vu=vu and torch.stack((node, node.T)).flatten(start_dim=1),
            index_uu=uu
            and torch.stack(
                torch.broadcast_tensors(node, torch.diag(node)[:, None])
            ).flatten(start_dim=1),
            index_vv=vv
            and torch.stack(
                torch.broadcast_tensors(node, torch.diag(node)[None, :])
            ).flatten(start_dim=1),
            # global
            index_uG=uG
            and torch.stack(
                torch.broadcast_tensors(node[:, :, None], node[:, None, :])
            ).flatten(start_dim=1),
            index_vG=vG
            and torch.stack(
                torch.broadcast_tensors(node[None, :, :], node[:, None, :])
            ).flatten(start_dim=1),
            # local
            index_uL=uL
            and (node[None, None, :, 0] + graph.edge_index[:, :, None]).flatten(
                start_dim=1
            ),
            index_vL=vL
            and (
                node[None, None, 0, :] + graph.edge_index[:, :, None] * len(node)
            ).flatten(start_dim=1),
            attrs_uL=graph.edge_attr[:, None].expand(-1, len(node)).flatten(end_dim=1)
            if uL and graph.edge_attr is not None
            else None,
            attrs_vL=graph.edge_attr[:, None].expand(-1, len(node)).flatten(end_dim=1)
            if vL and graph.edge_attr is not None
            else None,
        )

    return call


def aggregate(graph, agg, encode=None):
    dst, src = graph[f"index_{agg}"]

    message = torch.index_select(graph.x, dim=0, index=src)
    if encode is not None:
        message = encode(message, graph[f"attrs_{agg}"])

    return pys.scatter(message, dim=0, index=dst, dim_size=len(graph.x))
