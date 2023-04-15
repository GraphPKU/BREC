import networkx as nx 
import itertools

def create_gadget(k):
    G = nx.Graph()
    # create as and bs 
    for i in range(k):
        G.add_node(("a",i))
        G.add_node(("b",i))

    # create middle nodes
    elements = list(range(k))
    i = 0
    for n in range(0,k+1,2):
        for subset in itertools.combinations(elements, n):
            G.add_node(('m',i))
            for k in subset:
                G.add_edge(('m',i),  ('a',k))
            for k in set(elements) - set(subset):
                G.add_edge(('m',i),  ('b',k))
            i += 1
    return G

def create_cfi(k):
    gadget = create_gadget(k-1)
    
    # create all gadgets
    G = nx.Graph()
    for i in range(k):
        tmp = gadget.copy()
        label_mapping = {node: (i, *node) for node in tmp.nodes}
        G.update(nx.relabel_nodes(tmp, label_mapping))

    # add label 
    colors = {}
    i = 0
    for node in G.nodes():
        # all middle nodes have same color, a and b have same color
        c = (node[0], -1) if node[1] == 'm' else (node[0], node[-1])
        if c not in colors:
            colors[c] = i 
            i += 1
        G.nodes[node]['x'] = colors[c]

    # create connections among all nodes
    index = [0] * k
    base_G = nx.complete_graph(k)
    for edge in base_G.edges():
        left, right = edge
        edge_a = (left, 'a', index[left]), (right, 'a', index[right])
        edge_b = (left, 'b', index[left]), (right, 'b', index[right])
        G.add_edge(*edge_a)
        G.add_edge(*edge_b)
        index[left] += 1 
        index[right] += 1 

    H = G.copy()
    H.remove_edge(*edge_a)
    H.remove_edge(*edge_b)
    H.add_edge(edge_a[0], edge_b[1])
    H.add_edge(edge_b[0], edge_a[1])
    # create correct node labels 
    return G, H

# TODO: create Martin Grohe's version of CFI graph.


def create_grohe_cfi(k):
    return grohe_cfi(k, False), grohe_cfi(k, True)

def grohe_cfi(k, hat=False):
    G = nx.Graph()
    base_G = nx.complete_graph(k)

    edges_hash = {}
    # create edge-based vertex and connections
    for i, edge in enumerate(base_G.edges()):
        edges_hash[edge] = i
        edges_hash[(edge[1], edge[0])] = i
        G.add_node(('e', 0, i))
        G.add_node(('e', 1, i))
        G.add_edge(('e', 0, i),  ('e', 1, i))


    # create node-based vertex, and connection to edge-based vertex
    for i, node in enumerate(base_G.nodes()):
        adjacent_edges = base_G.edges(node)
        encoded_edges = [edges_hash[edge] for edge in adjacent_edges]
        start = 1 if i == 0 and hat else 0    
        for n in range(start,len(encoded_edges)+1,2):
            for subset in itertools.combinations(encoded_edges, n):
                vertex = ('v', subset, node)
                G.add_node(vertex)
                for edge in encoded_edges:
                    if edge in subset:
                        G.add_edge(vertex, ('e', 1, edge))
                    else:
                        G.add_edge(vertex, ('e', 0, edge))

    # create color for all vertex
    colors = {}
    i = 0
    for node in G.nodes():
        # all middle nodes have same color, a and b have same color
        c = (node[0], node[-1])
        if c not in colors:
            colors[c] = i 
            i += 1
        G.nodes[node]['x'] = colors[c]
    
    return G