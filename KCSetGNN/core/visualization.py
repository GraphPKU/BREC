import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
from torch_geometric.utils import to_networkx

def plot_nx_graphs(G_list, node_size=20, edge_size=0.2, layout='spring', node_attr='x', edge_attr='edge_attr'):
    N = len(G_list)
    n_rows = int(np.sqrt(N))
    n_cols = int(np.ceil(N/n_rows))
    plt.figure(figsize=(n_rows*4, n_cols*3))
    for i, G in enumerate(G_list):
        if layout=='spring':
            node_pos = nx.spring_layout(G)
        else:
            node_pos = nx.spectral_layout(G) # use spectral layout
        plt.subplot(n_rows, n_cols, i+1)
        node_color = 'blue'
        edge_color = 'k'
        # [edge[-1] for edge in nx_list[0].edges(data='edge_attr')]
        if node_attr is not None:
            node_color =  [node[-1] for node in G.nodes(data=node_attr)]
            # nx.get_node_attributes(G, node_attr)
        if edge_attr is not None:
            edge_color = [edge[-1] for edge in G.edges(data=edge_attr)]

        nx.draw(G, node_size=node_size, node_color=node_color, edge_color=edge_color,
                   linewidths=edge_size, pos=node_pos,  cmap='coolwarm')
    return plt


def plot_pgy_graphs(data_list, node_size=20, edge_size=0.2, layout='spring', node_attr='x', edge_attr='edge_attr'):
    G_list = [to_networkx(data, to_undirected='lower', node_attrs=[node_attr] if node_attr is not None else None, 
                                edge_attrs=[edge_attr] if edge_attr is not None else None) for data in data_list]
    return plot_nx_graphs(G_list, node_size, edge_size, layout, node_attr, edge_attr)