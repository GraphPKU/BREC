

#include "AuxiliaryMethods.h"
#include "GenerateThreeSampling.h"


namespace GenerateThreeSampling {
    GenerateThreeSampling::GenerateThreeSampling(const GraphDatabase &graph_database) : m_graph_database(
            graph_database), m_label_to_index(), m_num_labels(0) {}


    GramMatrix
    GenerateThreeSampling::compute_gram_matrix(const uint num_iterations, const bool use_labels,
                                               const uint num_samples, const bool simple) {
        vector<ColorCounter> color_counters;
        color_counters.reserve(m_graph_database.size());

        // Compute labels for each graph in graph database.
        for (Graph &graph: m_graph_database) {
            if (simple) {
                color_counters.push_back(compute_colors_simple(graph, num_iterations, num_samples, use_labels));
            } else {
                color_counters.push_back(compute_colors(graph, num_iterations, num_samples, use_labels));
            }
        }

        size_t num_graphs = m_graph_database.size();
        vector<S> nonzero_compenents;

        ColorCounter c;
        for (Node i = 0; i < num_graphs; ++i) {
            c = color_counters[i];

            for (const auto &j: c) {
                Label key = j.first;
                uint value = j.second;
                uint index = m_label_to_index.find(key)->second;
                nonzero_compenents.push_back(S(i, index, value));
            }
        }

        // Compute Gram matrix.
        GramMatrix feature_vectors(num_graphs, m_num_labels);
        feature_vectors.setFromTriplets(nonzero_compenents.begin(), nonzero_compenents.end());
        feature_vectors = feature_vectors * (1.0 / m_num_labels);
        GramMatrix gram_matrix(num_graphs, num_graphs);
        gram_matrix = feature_vectors * feature_vectors.transpose();

        return gram_matrix;
    }

    ColorCounter
    GenerateThreeSampling::compute_colors(Graph &g, const uint num_iterations, const uint num_samples,
                                          const bool use_labels) {

        Graph tuple_graph(false);
        tuple_graph = generate_local_graph(g, num_iterations, num_samples, use_labels);

        size_t num_nodes = tuple_graph.get_num_nodes();

        Labels coloring;
        Labels coloring_temp;

        coloring.reserve(num_nodes);
        coloring_temp.reserve(num_nodes);
        coloring = tuple_graph.get_labels();
        coloring_temp = coloring;

        EdgeLabels edge_labels = tuple_graph.get_edge_labels();
        EdgeLabels vertex_id = tuple_graph.get_vertex_id();
        EdgeLabels local = tuple_graph.get_local();

        ColorCounter color_map;

        for (Node v = 0; v < num_nodes; ++v) {
            Label new_color = coloring[v];

            ColorCounter::iterator it(color_map.find(new_color));
            if (it == color_map.end()) {
                color_map.insert({{new_color, 1}});
                m_label_to_index.insert({{new_color, m_num_labels}});
                m_num_labels++;
            } else {
                it->second++;
            }
        }

        uint h = 1;
        while (h <= num_iterations) {
            // Iterate over all nodes.
            for (Node v = 0; v < num_nodes; ++v) {
                Labels colors_local;
                Labels colors_global;
                Nodes neighbors(tuple_graph.get_neighbours(v));
                colors_local.reserve(neighbors.size() + 1);
                colors_global.reserve(neighbors.size() + 1);

                // New color of node v.
                Label new_color;

                vector<vector<Label>> set_m_local;
                vector<vector<Label>> set_m_global;
                unordered_map<uint, uint> id_to_position_local;
                unordered_map<uint, uint> id_to_position_global;

                uint dl = 0;
                // Get colors of neighbors.
                for (const Node &n: neighbors) {
                    const auto t = edge_labels.find(make_tuple(v, n));
                    Label l = AuxiliaryMethods::pairing(coloring[n], t->second);

                    const auto type = local.find(make_tuple(v, n));

                    if (type->second == 1) {
                        const auto s = vertex_id.find(make_tuple(v, n));
                        const auto pos(id_to_position_local.find(s->second));
                        if (pos != id_to_position_local.end()) {
                            set_m_local[pos->second].push_back(l);
                        } else {
                            id_to_position_local.insert({{s->second, dl}});
                            set_m_local.push_back(vector<Label>());
                            set_m_local[dl].push_back(l);
                            dl++;

                        }
                    }
                }


                for (auto &m: set_m_local) {
                    if (m.size() != 0) {
                        sort(m.begin(), m.end());
                        new_color = m.back();
                        m.pop_back();
                        for (const Label &c: m) {
                            new_color = AuxiliaryMethods::pairing(new_color, c);
                        }
                        colors_local.push_back(new_color);
                    }
                }
                sort(colors_local.begin(), colors_local.end());
                colors_local.push_back(coloring[v]);

                // Compute new label using composition of pairing function of Matthew Szudzik to map two integers to on integer.
                new_color = colors_local.back();
                colors_local.pop_back();
                for (const Label &c: colors_local) {
                    new_color = AuxiliaryMethods::pairing(new_color, c);
                }
                coloring_temp[v] = new_color;

                // Keep track how often "new_label" occurs.
                auto it(color_map.find(new_color));
                if (it == color_map.end()) {
                    color_map.insert({{new_color, 1}});
                    m_label_to_index.insert({{new_color, m_num_labels}});
                    m_num_labels++;
                } else {
                    it->second++;
                }
            }

            // Assign new colors.
            coloring = coloring_temp;
            h++;
        }

        return color_map;
    }

    ColorCounter
    GenerateThreeSampling::compute_colors_simple(Graph &g, const uint num_iterations, const uint num_samples,
                                                 const bool use_labels) {
        Graph tuple_graph(false);

        tuple_graph = generate_local_graph(g, num_iterations, num_samples, use_labels);



        size_t num_nodes = tuple_graph.get_num_nodes();

        Labels coloring;
        Labels coloring_temp;

        coloring.reserve(num_nodes);
        coloring_temp.reserve(num_nodes);
        coloring = tuple_graph.get_labels();
        coloring_temp = coloring;

        EdgeLabels edge_labels = tuple_graph.get_edge_labels();
        EdgeLabels local = tuple_graph.get_local();

        ColorCounter color_map;
        for (Node v = 0; v < num_nodes; ++v) {
            Label new_color = coloring[v];

            ColorCounter::iterator it(color_map.find(new_color));
            if (it == color_map.end()) {
                color_map.insert({{new_color, 1}});
                m_label_to_index.insert({{new_color, m_num_labels}});
                m_num_labels++;
            } else {
                it->second++;
            }
        }

        uint h = 1;
        while (h <= num_iterations) {
            // Iterate over all nodes.
            for (Node v = 0; v < num_nodes; ++v) {


                Labels colors_local;
                Labels colors_global;
                Nodes neighbors(tuple_graph.get_neighbours(v));
                colors_local.reserve(neighbors.size() + 1);
                colors_global.reserve(neighbors.size() + 1);

                // New color of node v.
                Label new_color;

                vector<vector<Label>> set_m_local;
                vector<vector<Label>> set_m_global;

                set_m_local.push_back(vector<Label>());
                set_m_local.push_back(vector<Label>());
                set_m_local.push_back(vector<Label>());

                set_m_global.push_back(vector<Label>());
                set_m_global.push_back(vector<Label>());
                set_m_global.push_back(vector<Label>());


                // Get colors of neighbors.
                for (const Node &n: neighbors) {
                    const auto type = local.find(make_tuple(v, n));
                    const auto label = edge_labels.find(make_tuple(v, n))->second;

                    // Local neighbor.
                    if (type->second == 1) {
                        if (label == 1) {
                            set_m_local[0].push_back(coloring[n]);
                        }
                        if (label == 2) {
                            set_m_local[1].push_back(coloring[n]);
                        }
                        if (label == 3) {
                            set_m_local[2].push_back(coloring[n]);
                        }
                    }
                }

                for (auto &m: set_m_local) {
                    if (m.size() != 0) {
                        sort(m.begin(), m.end());
                        new_color = m.back();
                        m.pop_back();
                        for (const Label &c: m) {
                            new_color = AuxiliaryMethods::pairing(new_color, c);
                        }
                        colors_local.push_back(new_color);
                    }
                }
                sort(colors_local.begin(), colors_local.end());
                colors_local.push_back(coloring[v]);

                // Compute new label using composition of pairing function of Matthew Szudzik to map two integers to on integer.
                new_color = colors_local.back();
                colors_local.pop_back();
                for (const Label &c: colors_local) {
                    new_color = AuxiliaryMethods::pairing(new_color, c);
                }
                coloring_temp[v] = new_color;

                // Keep track how often "new_label" occurs.
                auto it(color_map.find(new_color));
                if (it == color_map.end()) {
                    color_map.insert({{new_color, 1}});
                    m_label_to_index.insert({{new_color, m_num_labels}});
                    m_num_labels++;
                } else {
                    it->second++;
                }


            }

            // Assign new colors.
            coloring = coloring_temp;
            h++;

        }

        return color_map;
    }


    Graph
    GenerateThreeSampling::generate_local_graph(Graph &g, const uint num_iterations, const uint num_samples,
                                                bool use_labels) {
        // New tuple graph.
        Graph new_graph(false);
        size_t num_nodes = g.get_num_nodes();

        // Random device for sampling node in the graph.
        random_device rand_dev;
        mt19937 mt(rand_dev());
        uniform_int_distribution<Node> uniform_node_sampler(0, num_nodes - 1);



        unordered_map<Edge, uint> edge_type;
        // Manages vertex ids
        unordered_map<Edge, uint> vertex_id;
        // Manage type of neighborhood.
        unordered_map<Edge, uint> local;
        // Map tuples to node ids in new graph.
        unordered_map<ThreeTuple, uint> triple_to_int;
        // Node labels of new graph.
        unordered_map<Node, Label> node_label_map;

        Labels node_labels;
        if (use_labels) {
            node_labels = g.get_labels();
        }

        for (uint cc = 0; cc < num_samples; ++cc) {
            // Sample a triple.
            Node i = uniform_node_sampler(mt);
            Node j = uniform_node_sampler(mt);
            Node k = uniform_node_sampler(mt);

            //cout << i << j << k << endl;

            ThreeTuple triple = make_tuple(i, j, k);

            // Add node to graph representing the triple.
            triple_to_int.insert({{triple, new_graph.get_num_nodes()}});
            Node new_node = new_graph.add_node();

            Label l;

            Label c_i = 1;
            Label c_j = 2;
            Label c_k = 3;

            if (use_labels) {
                c_i = AuxiliaryMethods::pairing(node_labels[i] + 1, c_i);
                c_j = AuxiliaryMethods::pairing(node_labels[j] + 1, c_j);
                c_k = AuxiliaryMethods::pairing(node_labels[k] + 1, c_k);
            }

            Label a, b, c;
            if (g.has_edge(i, j)) {
                a = 1;
            } else if (not g.has_edge(i, j)) {
                a = 2;
            } else {
                a = 3;
            }

            if (g.has_edge(i, k)) {
                b = 1;
            } else if (not g.has_edge(i, k)) {
                b = 2;
            } else {
                b = 3;
            }

            if (g.has_edge(j, k)) {
                c = 1;
            } else if (not g.has_edge(j, k)) {
                c = 2;
            } else {
                c = 3;
            }

            Label new_color_0 = AuxiliaryMethods::pairing(AuxiliaryMethods::pairing(a, b), c);
            Label new_color_1 = AuxiliaryMethods::pairing(AuxiliaryMethods::pairing(c_i, c_j), c_k);
            l = AuxiliaryMethods::pairing(new_color_0, new_color_1);

            node_label_map.insert({{new_node, l}});

            // Get neighborhood around the triple up to depth.
            explore_neighborhood(g, triple, num_iterations, triple_to_int, new_graph, edge_type, vertex_id, local,
                                 node_label_map, use_labels);

        }




        Labels new_node_labels;
        uint n = new_graph.get_num_nodes();
        for (uint i = 0; i < n; ++i) {
            new_node_labels.push_back(node_label_map.find(i)->second);
        }

        new_graph.set_edge_labels(edge_type);
        new_graph.set_vertex_id(vertex_id);
        new_graph.set_local(local);
        new_graph.set_labels(new_node_labels);

        return new_graph;
    }

    void
    GenerateThreeSampling::explore_neighborhood(Graph &g, const ThreeTuple &triple, const uint num_iterations,
                                                unordered_map<ThreeTuple, uint> &triple_to_int, Graph &new_graph,
                                                unordered_map<Edge, uint> &edge_type,
                                                unordered_map<Edge, uint> &vertex_id,
                                                unordered_map<Edge, uint> &local,
                                                unordered_map<Node, Label> &node_label_map, const bool use_labels) {

        // Manage depth of node in k-disk.
        unordered_map<ThreeTuple, uint> depth;

        // Node is in here iff it has been visited.
        unordered_set<ThreeTuple> visited;

        // Queue of nodes for DFS.
        queue<ThreeTuple> queue;

        // Push center tuple to queue.
        queue.push(triple);

        visited.insert(triple);
        depth.insert({{triple, 0}});

        Labels labels;
        if (use_labels) {
            labels = g.get_labels();
        }

        while (!queue.empty()) {
            ThreeTuple q(queue.front());
            queue.pop();

            Node v = get<0>(q);
            Node w = get<1>(q);
            Node u = get<2>(q);
            Node current_node = triple_to_int.find(make_tuple(v, w, u))->second;
            uint current_depth = depth.find(q)->second;

            if (current_depth <= num_iterations) {
                vector<ThreeTuple> neighbours;

                // Exchange first node.
                Nodes v_neighbors = g.get_neighbours(v);
                for (const auto &v_n: v_neighbors) {
                    auto t = triple_to_int.find(make_tuple(v_n, w, u));

                    Node new_node;
                    if (t == triple_to_int.end()) {
                        triple_to_int.insert({{make_tuple(v_n, w, u), new_graph.get_num_nodes()}});
                        new_node = new_graph.add_node();
                    } else {
                        new_node = t->second;
                    }
                    new_graph.add_edge(current_node, new_node);


                    Label new_label;
                    if (use_labels) {
                        new_label = compute_label(g, v_n, w, u, labels[v_n]+1, labels[w]+2, labels[u]+3);
                    } else {
                        new_label = compute_label(g, v_n, w, u, 1, 2, 3);
                    }

                    node_label_map.insert({{new_node, new_label}});
                    edge_type.insert({{make_tuple(current_node, new_node), 1}});
                    edge_type.insert({{make_tuple(new_node, current_node), 1}});
                    vertex_id.insert({{make_tuple(current_node, new_node), v_n}});
                    vertex_id.insert({{make_tuple(new_node, current_node), v_n}});
                    local.insert({{make_tuple(current_node, new_node), 1}});
                    local.insert({{make_tuple(new_node, current_node), 1}});

                    neighbours.push_back(make_tuple(v_n, w, u));
                }

                // Exchange second node.
                Nodes w_neighbors = g.get_neighbours(w);
                for (const auto &w_n: w_neighbors) {
                    auto t = triple_to_int.find(make_tuple(v, w_n, u));

                    Node new_node;
                    if (t == triple_to_int.end()) {
                        triple_to_int.insert({{make_tuple(v, w_n, u), new_graph.get_num_nodes()}});
                        new_node = new_graph.add_node();
                    } else {
                        new_node = t->second;
                    }
                    new_graph.add_edge(current_node, new_node);


                    Label new_label;
                    if (use_labels) {
                        new_label = compute_label(g, v, w_n, u, labels[v]+1, labels[w_n]+2, labels[u]+3);
                    } else {
                        new_label = compute_label(g, v, w_n, u, 1, 2, 3);
                    }


                    node_label_map.insert({{new_node, new_label}});
                    edge_type.insert({{make_tuple(current_node, new_node), 2}});
                    edge_type.insert({{make_tuple(new_node, current_node), 2}});
                    vertex_id.insert({{make_tuple(current_node, new_node), w_n}});
                    vertex_id.insert({{make_tuple(new_node, current_node), w_n}});
                    local.insert({{make_tuple(current_node, new_node), 1}});
                    local.insert({{make_tuple(new_node, current_node), 1}});

                    neighbours.push_back(make_tuple(v, w_n, u));
                }


                // Exchange third node.
                Nodes u_neighbors = g.get_neighbours(u);
                for (const auto &u_n: u_neighbors) {
                    auto t = triple_to_int.find(make_tuple(v, w, u_n));

                    Node new_node;
                    if (t == triple_to_int.end()) {
                        triple_to_int.insert({{make_tuple(v, w, u_n), new_graph.get_num_nodes()}});
                        new_node = new_graph.add_node();
                    } else {
                        new_node = t->second;
                    }
                    new_graph.add_edge(current_node, new_node);

                    Label new_label;
                    if (use_labels) {
                        new_label = compute_label(g, v, w, u_n, labels[v]+1, labels[w]+2, labels[u_n]+3);
                    } else {
                        new_label = compute_label(g, v, w, u_n, 1, 2, 3);
                    }

                    node_label_map.insert({{new_node, new_label}});
                    edge_type.insert({{make_tuple(current_node, new_node), 3}});
                    edge_type.insert({{make_tuple(new_node, current_node), 3}});
                    vertex_id.insert({{make_tuple(current_node, new_node), u_n}});
                    vertex_id.insert({{make_tuple(new_node, current_node), u_n}});
                    local.insert({{make_tuple(current_node, new_node), 1}});
                    local.insert({{make_tuple(new_node, current_node), 1}});

                    neighbours.push_back(make_tuple(v, w, u_n));
                }

                // Preprocessing.
                for (ThreeTuple &n: neighbours) {
                    if (visited.find(n) == visited.end()) {
                        depth.insert({{n, current_depth + 1}});
                        queue.push(n);
                        visited.insert(n);
                    }
                }
            }
        }
    }

    Label GenerateThreeSampling::compute_label(Graph& g, Node i, Node j, Node k, Label c_i, Label c_j, Label c_k) {

        Label a, b, c;
        if (g.has_edge(i,j)) {
            a = 1;
        } else if (not g.has_edge(i, j)) {
            a = 2;
        } else {
            a = 3;
        }

        if (g.has_edge(i, k)) {
            b = 1;
        } else if (not g.has_edge(i, k)) {
            b = 2;
        } else {
            b = 3;
        }

        if (g.has_edge(j, k)) {
            c = 1;
        } else if (not g.has_edge(j, k)) {
            c = 2;
        } else {
            c = 3;
        }

        Label new_color_0 = AuxiliaryMethods::pairing(AuxiliaryMethods::pairing(a, b), c);
        Label new_color_1 = AuxiliaryMethods::pairing(AuxiliaryMethods::pairing(c_i, c_j), c_k);
        Label l = AuxiliaryMethods::pairing(new_color_0, new_color_1);


        return l;
    }


    GenerateThreeSampling::~GenerateThreeSampling() {}
}
