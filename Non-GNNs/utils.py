from itertools import product
import networkx as nx
import copy
from collections import Counter
import numpy as np
import random
import pynauty
import re


def WL_marking(G, m):
    # k should satisfy that any subgraph with k nodes is distinguishable with 1-WL
    node_list = [x for x in G.nodes]
    n = G.number_of_nodes()
    node_to_id = dict()
    cnt = 0
    for i in range(n):
        node_to_id[node_list[i]] = i
    hash_wl_discret = dict()  # discretization of wl hash value
    node_hash = []  # hash_value for each node
    node_hash_discret = []  # hash_value discretization for each node

    for node in node_list:
        hash_wl = 1
        for (id, m_node) in enumerate(m):
            if node_to_id[node] == m_node:
                hash_wl = id + 2
        if hash_wl not in hash_wl_discret:
            hash_wl_discret[hash_wl] = cnt
            cnt += 1
        node_hash.append(hash_wl)
        node_hash_discret.append(hash_wl_discret[hash_wl])

    epoch = 1
    while epoch:
        epoch += 1
        hash_wl_discret = dict()
        node_hash_nxt = []
        node_hash_discret_nxt = []
        cnt = 0
        for id in range(len(node_list)):
            node = node_list[id]
            hash_neighbor_list = []

            for neighbor in nx.all_neighbors(G, node):
                hash_neighbor_list.append(node_hash[node_to_id[neighbor]])

            counter = Counter(hash_neighbor_list)
            hash_list = sorted(counter.items(), key=lambda x: x[0])
            hash_list.append(node_hash[id])
            hash_l = hash(str(hash_list))
            if hash_l not in hash_wl_discret:
                hash_wl_discret[hash_l] = cnt
                cnt += 1
            node_hash_nxt.append(hash_l)
            node_hash_discret_nxt.append(hash_wl_discret[hash_l])

        if hash(str(node_hash_discret)) == hash(str(node_hash_discret_nxt)):
            counter = Counter(node_hash_nxt)
            return_hash = sorted(counter.items(), key=lambda x: x[0])
            return hash(str(return_hash))

        node_hash = copy.deepcopy(node_hash_nxt)
        node_hash_discret = copy.deepcopy(node_hash_discret_nxt)


def count_sub_3(G):
    node_list = list(G.nodes)
    n = len(node_list)
    node_to_id = dict()
    for i in range(n):
        node_to_id[node_list[i]] = i
    # 0 type is triangle, 1 type is 3-path
    type_list = [[0] * n, [0] * n]
    for (id_1, node_1) in enumerate(node_list):
        for node_2 in G.adj[node_1]:
            id_2 = node_to_id[node_2]
            for node_3 in G.adj[node_2]:
                id_3 = node_to_id[node_3]
                if id_3 == id_1:
                    continue
                if G.has_edge(node_1, node_3):  # triangle
                    type_list[0][id_1] += 1
                    type_list[0][id_2] += 1
                    type_list[0][id_3] += 1
                else:  # 3-path
                    type_list[1][id_1] += 1
                    type_list[1][id_2] += 1
                    type_list[1][id_3] += 1
    return type_list


def count_sub_4(G):
    """
    type 0:     o - o - o - o

    type 1:     o - o - o
                    |
                    o

    type 2:     o - o
                | / |
                o - o

    type 3:     o - o
                | x |
                o - o

    type 4:     o - o
                |   |
                o - o

    type 5:     o - o - o
                    | /
                    o


    """
    node_list = list(G.nodes)
    n = len(node_list)
    node_to_id = dict()
    for i in range(n):
        node_to_id[node_list[i]] = i
    type_list = [[0] * n, [0] * n, [0] * n, [0] * n, [0] * n, [0] * n]
    for (id_1, node_1) in enumerate(node_list):
        for node_2 in G.adj[node_1]:
            id_2 = node_to_id[node_2]
            for node_3 in G.adj[node_2]:
                id_3 = node_to_id[node_3]
                if id_3 == id_1:
                    continue
                for node_4 in G.adj[node_3]:
                    id_4 = node_to_id[node_4]
                    if id_4 == id_1 or id_4 == id_2:
                        continue
                    edge_num = (
                        (node_1 in G[node_3])
                        + (node_1 in G[node_4])
                        + (node_2 in G[node_4])
                    )
                    if edge_num != 1:
                        type_list[edge_num][id_1] += 1
                        type_list[edge_num][id_2] += 1
                        type_list[edge_num][id_3] += 1
                        type_list[edge_num][id_4] += 1
                    elif G.has_edge(node_1, node_4):
                        type_list[4][id_1] += 1
                        type_list[4][id_2] += 1
                        type_list[4][id_3] += 1
                        type_list[4][id_4] += 1
                    else:
                        type_list[5][id_1] += 1
                        type_list[5][id_2] += 1
                        type_list[5][id_3] += 1
                        type_list[5][id_4] += 1
    for (id_1, node_1) in enumerate(node_list):
        for node_2 in G.adj[node_1]:
            id_2 = node_to_id[node_2]
            for node_3 in G.adj[node_1]:
                id_3 = node_to_id[node_3]
                if id_3 <= id_2:
                    continue
                for node_4 in G.adj[node_1]:
                    id_4 = node_to_id[node_4]
                    if id_4 <= id_3:
                        continue
                    edge_num = (
                        (node_2 in G[node_3])
                        + (node_3 in G[node_4])
                        + (node_2 in G[node_4])
                    )
                    if edge_num == 0:
                        type_list[1][id_1] += 1
                        type_list[1][id_2] += 1
                        type_list[1][id_3] += 1
                        type_list[1][id_4] += 1
    return type_list


def tuple_isomorphism_generator(tuple_u, graph_G):
    """
    Given a tuple 'tuple_u', generate tuple isomorphism based on 'graph_G'
    """
    return_type = ""
    for node_u in tuple_u:
        for node_v in tuple_u:
            if graph_G.has_edge(node_u, node_v):
                return_type += "1"
            else:
                return_type += "0"
    return return_type


def WL_hash(G, k, mode=None):
    # k should satisfy that any subgraph with k nodes is distinguishable with 1-WL
    node_list = [x for x in G.nodes]
    # print('node_list', node_list)
    n = G.number_of_nodes()
    node_to_id = dict()
    cnt = 0
    for i in range(n):
        node_to_id[node_list[i]] = i
    hash_wl_discret = dict()  # discretization of wl hash value
    node_vector_hash = []  # hash_value for each vector
    node_vector_hash_discret = []  # hash_value discretization for each vector
    node_vector_list = list(product(node_list, repeat=k))

    for node_vector in node_vector_list:
        sub_G = G.subgraph(node_vector)
        hash_wl = tuple_isomorphism_generator(node_vector, sub_G)
        # hash_wl = nx.weisfeiler_lehman_graph_hash(sub_G)
        # hash_wl = sub_G.number_of_edges()
        if hash_wl not in hash_wl_discret:
            hash_wl_discret[hash_wl] = cnt
            cnt += 1
        node_vector_hash.append(hash_wl)
        node_vector_hash_discret.append(hash_wl_discret[hash_wl])

    epoch = 1
    while epoch:
        epoch += 1
        hash_wl_discret = dict()
        node_vector_hash_nxt = []
        node_vector_hash_discret_nxt = []
        cnt = 0
        for id in range(len(node_vector_list)):
            node_vector = node_vector_list[id]

            # hash_l_list is c^l_v, hash(str(node_vector_hash[id])) is c^(l-1)_v
            hash_l_list = [hash(str(node_vector_hash[id]))]

            for i in range(k):  # iterate pos of neighbor
                base_power = pow(n, k - 1 - i)
                id_remain = id - node_to_id[node_vector[i]] * base_power

                # hash_neighbor_list is {{c^l_(v,i)}}
                hash_neighbor_list = []
                for node in node_list:  # iterate neighbor node
                    id_cur = id_remain + node_to_id[node] * base_power
                    hash_neighbor_list.append(node_vector_hash[id_cur])
                hash_neighbor_list.sort()

                hash_l_list.append(hash(str(hash_neighbor_list)))
            # print('hash_l_list', hash_l_list)
            hash_l = hash(str(hash_l_list))
            if hash_l not in hash_wl_discret:
                hash_wl_discret[hash_l] = cnt
                cnt += 1

            node_vector_hash_nxt.append(hash_l)
            node_vector_hash_discret_nxt.append(hash_wl_discret[hash_l])

        if hash(str(node_vector_hash_discret)) == hash(
            str(node_vector_hash_discret_nxt)
        ):
            counter = Counter(node_vector_hash_nxt)
            return_hash = sorted(counter.items(), key=lambda x: x[0])
            return hash(str(return_hash))

        node_vector_hash = copy.deepcopy(node_vector_hash_nxt)
        node_vector_hash_discret = copy.deepcopy(node_vector_hash_discret_nxt)


def FWL_hash(G, k, mode=None):
    # k should satisfy that any subgraph with k nodes is distinguishable with 1-WL
    node_list = [x for x in G.nodes]
    # print('node_list', node_list)
    n = G.number_of_nodes()
    node_to_id = dict()
    cnt = 0
    for i in range(n):
        node_to_id[node_list[i]] = i
    hash_wl_discret = dict()  # discretization of wl hash value
    node_vector_hash = []  # hash_value for each vector
    node_vector_hash_discret = []  # hash_value discretization for each vector
    node_vector_list = list(product(node_list, repeat=k))

    for node_vector in node_vector_list:
        sub_G = G.subgraph(node_vector)
        # hash_wl = nx.weisfeiler_lehman_graph_hash(sub_G)
        hash_wl = tuple_isomorphism_generator(node_vector, sub_G)
        # hash_wl = sub_G.number_of_edges()
        if hash_wl not in hash_wl_discret:
            hash_wl_discret[hash_wl] = cnt
            cnt += 1
        node_vector_hash.append(hash_wl)
        node_vector_hash_discret.append(hash_wl_discret[hash_wl])

    epoch = 1
    while epoch:
        epoch += 1
        hash_wl_discret = dict()
        node_vector_hash_nxt = []
        node_vector_hash_discret_nxt = []
        cnt = 0

        for id in range(len(node_vector_list)):
            node_vector = node_vector_list[id]

            # hash_l_list is c^l_v, hash(str(node_vector_hash[id])) is c^(l-1)_v
            # hash_l_list = [hash(str(node_vector_hash[id]))]
            hash_l_list = []

            for node in node_list:  # iterate neighbor node
                id_node = node_to_id[node]
                hash_neighbor_list = []

                for i in range(k):  # iterate pos of neighbor
                    base_power = pow(n, k - 1 - i)
                    id_cur = id + (id_node - node_to_id[node_vector[i]]) * base_power
                    hash_neighbor_list.append(node_vector_hash[id_cur])

                hash_l_list.append(hash(str(hash_neighbor_list)))

            counter = Counter(hash_l_list)
            hash_l_list_sorted = sorted(counter.items(), key=lambda x: x[0])
            hash_l_list_sorted.append(node_vector_hash[id])
            hash_l = hash(str(hash_l_list_sorted))
            if hash_l not in hash_wl_discret:
                hash_wl_discret[hash_l] = cnt
                cnt += 1

            node_vector_hash_nxt.append(hash_l)
            node_vector_hash_discret_nxt.append(hash_wl_discret[hash_l])

        if hash(str(node_vector_hash_discret)) == hash(
            str(node_vector_hash_discret_nxt)
        ):
            counter = Counter(node_vector_hash_nxt)
            return_hash = sorted(counter.items(), key=lambda x: x[0])
            return hash(str(return_hash))

        node_vector_hash = copy.deepcopy(node_vector_hash_nxt)
        node_vector_hash_discret = copy.deepcopy(node_vector_hash_discret_nxt)


def WL_1_hash(G, k=1, mode="none"):
    np.random.seed(2022)
    random.seed(2022)
    # k should satisfy that any subgraph with k nodes is distinguishable with 1-WL
    node_list = [x for x in G.nodes]
    # print('node_list', node_list)
    n = G.number_of_nodes()
    node_to_id = dict()
    cnt = 0
    for i in range(n):
        node_to_id[node_list[i]] = i
    hash_wl_discret = dict()  # discretization of wl hash value
    node_hash = []  # hash_value for each node
    node_hash_discret = []  # hash_value discretization for each node

    if mode == "s3":
        type_list = count_sub_3(G)
        for (id, node) in enumerate(node_list):
            hash_wl = hash(f"{type_list[0][id]}_{type_list[1][id]}")
            if hash_wl not in hash_wl_discret:
                hash_wl_discret[hash_wl] = cnt
                cnt += 1
            node_hash.append(hash_wl)
            node_hash_discret.append(hash_wl_discret[hash_wl])
    elif mode == "s4":
        type_list = count_sub_4(G)
        for (id, node) in enumerate(node_list):
            tmp_str = (
                f"{type_list[0][id]}_{type_list[1][id]}_{type_list[2][id]}_"
                + f"{type_list[3][id]}_{type_list[4][id]}_{type_list[5][id]}"
            )
            hash_wl = hash(tmp_str)
            if hash_wl not in hash_wl_discret:
                hash_wl_discret[hash_wl] = cnt
                cnt += 1
            node_hash.append(hash_wl)
            node_hash_discret.append(hash_wl_discret[hash_wl])
    elif mode == "m1":
        hash_return = []
        for i in range(n):
            hash_return.append(WL_marking(G, [i]))
        counter = Counter(hash_return)
        hash_return_sorted = sorted(counter.items(), key=lambda x: x[0])
        hash_return_sorted.append(WL_1_hash(G))
        return_hash = hash(str(hash_return_sorted))
        return return_hash
    elif mode == "m2":
        hash_return = []
        for i in range(n):
            for j in range(i + 1, n):
                hash_return.append(WL_marking(G, [i, j]))
        counter = Counter(hash_return)
        hash_return_sorted = sorted(counter.items(), key=lambda x: x[0])
        hash_return_sorted.append(WL_1_hash(G))
        return_hash = hash(str(hash_return_sorted))
        return return_hash
    elif mode == "none" or mode == "None":
        for node in node_list:
            # hash_wl = 1
            hash_wl = G.degree[node]
            if hash_wl not in hash_wl_discret:
                hash_wl_discret[hash_wl] = cnt
                cnt += 1
            node_hash.append(hash_wl)
            node_hash_discret.append(hash_wl_discret[hash_wl])
    elif mode.startswith("n"):
        radius = int(re.findall(r"\d+", mode)[0])
        # print(radius)
        for node in node_list:
            sub_G = nx.ego_graph(G, node, radius)
            hash_wl = pynauty.certificate(from_nx_to_nauty(sub_G))
            if hash_wl not in hash_wl_discret:
                hash_wl_discret[hash_wl] = cnt
                cnt += 1
            node_hash.append(hash_wl)
            node_hash_discret.append(hash_wl_discret[hash_wl])
    else:
        print(f"{mode} is not supported!")
        exit()
    epoch = 1
    while epoch:
        epoch += 1
        hash_wl_discret = dict()
        node_hash_nxt = []
        node_hash_discret_nxt = []
        cnt = 0
        for id in range(len(node_list)):
            node = node_list[id]
            hash_neighbor_list = []

            for neighbor in nx.all_neighbors(G, node):
                hash_neighbor_list.append(node_hash[node_to_id[neighbor]])

            counter = Counter(hash_neighbor_list)
            hash_list = sorted(counter.items(), key=lambda x: x[0])
            hash_list.append(node_hash[id])
            hash_l = hash(str(hash_list))
            if hash_l not in hash_wl_discret:
                hash_wl_discret[hash_l] = cnt
                cnt += 1
            node_hash_nxt.append(hash_l)
            node_hash_discret_nxt.append(hash_wl_discret[hash_l])

        if hash(str(node_hash_discret)) == hash(str(node_hash_discret_nxt)):
            counter = Counter(node_hash_nxt)
            return_hash = sorted(counter.items(), key=lambda x: x[0])
            return hash(str(return_hash))

        node_hash = copy.deepcopy(node_hash_nxt)
        node_hash_discret = copy.deepcopy(node_hash_discret_nxt)


def from_nx_to_nauty(G):
    n = G.number_of_nodes()
    node_list = list(G.nodes)
    node_to_id = dict()
    for i in range(n):
        node_to_id[node_list[i]] = i
    adjacency_dict = dict()
    for k, v in dict(G.adj).items():
        adjacency_dict[node_to_id[k]] = [node_to_id[v_node] for v_node in v]
    # print(adjacency_dict)
    g = pynauty.Graph(number_of_vertices=n, adjacency_dict=adjacency_dict)
    return g


def Distance_WL_hash(G, k=1, mode="none"):
    distance_matrix = nx.floyd_warshall_numpy(G)
    G_full = nx.Graph()
    G_full.add_edges_from(
        [
            (
                i,
                j,
                {
                    "label": int(distance_matrix[i][j])
                    if distance_matrix[i][j] != np.inf
                    else 0
                },
            )
            for i in range(len(G))
            for j in range(len(G))
            if i != j
        ]
    )
    return nx.weisfeiler_lehman_graph_hash(G_full, edge_attr="label", iterations=10)


def Resistance_distance_WL_hash(G, k=1, mode="none"):
    distance_matrix = nx.floyd_warshall_numpy(G)
    distance_matrix = np.where(distance_matrix == np.inf, 0, distance_matrix)
    resistance_distance_matrix = np.zeros_like(distance_matrix)
    if nx.is_connected(G):
        for i in range(len(G)):
            for j in range(len(G)):
                if i != j:
                    resistance_distance_matrix[i][j] = nx.resistance_distance(G, i, j)
        resistance_distance_matrix = np.where(
            resistance_distance_matrix == np.inf, 0, resistance_distance_matrix
        )
    G_full = nx.Graph()
    G_full.add_edges_from(
        [
            (
                i,
                j,
                {
                    "label": str(int(distance_matrix[i][j]))
                    + "_"
                    + str(int(resistance_distance_matrix[i][j]))
                },
            )
            for i in range(len(G))
            for j in range(len(G))
            if i != j
        ]
    )
    return nx.weisfeiler_lehman_graph_hash(G_full, edge_attr="label", iterations=10)
