
#include "AuxiliaryMethods.h"
#include "GenerateThree.h"


namespace GenerateThree {
    GenerateThree::GenerateThree(const GraphDatabase &graph_database) : m_graph_database(
            graph_database),
                                                                        m_label_to_index(),
                                                                        m_num_labels(0) {}


    GramMatrix
    GenerateThree::compute_gram_matrix(const uint num_iterations, const bool use_labels, const string algorithm,
                                       const bool simple, const bool compute_gram) {
        vector<ColorCounter> color_counters;
        color_counters.reserve(m_graph_database.size());

        // Compute labels for each graph in graph database.
        for (Graph &graph: m_graph_database) {
            if (simple) {
                color_counters.push_back(compute_colors_simple(graph, num_iterations, use_labels, algorithm));
            } else {
                color_counters.push_back(compute_colors(graph, num_iterations, use_labels, algorithm));
            }
        }

        size_t num_graphs = m_graph_database.size();
        vector<S> nonzero_compenents;

        // Compute feature vectors.
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


        // Compute Gram matrix or feature vectore
        GramMatrix feature_vectors(num_graphs, m_num_labels);
        feature_vectors.setFromTriplets(nonzero_compenents.begin(), nonzero_compenents.end());

        if (not compute_gram) {
            return feature_vectors;
        } else {
            GramMatrix gram_matrix(num_graphs, num_graphs);
            gram_matrix = feature_vectors * feature_vectors.transpose();

            return gram_matrix;
        }

    }


    ColorCounter GenerateThree::compute_colors(const Graph &g, const uint num_iterations, const bool use_labels,
                                               const string algorithm) {

        Graph tuple_graph(false);
        if (algorithm == "local") {
            tuple_graph = generate_local_graph(g, use_labels);
        } else if (algorithm == "wl") {
            tuple_graph = generate_global_graph(g, use_labels);
        } else if (algorithm == "malkin") {
            tuple_graph = generate_global_graph_malkin(g, use_labels);
        }

        size_t num_nodes = tuple_graph.get_num_nodes();

        unordered_map<Node, TwoTuple> node_to_two_tuple;
        if (algorithm == "localp") {
            node_to_two_tuple = tuple_graph.get_node_to_two_tuple();
        }

        Labels coloring;
        ColorCounter color_map_1;
        ColorCounter color_map_2;
        ColorCounter color_map_3;

        Labels coloring_temp;

        coloring.reserve(num_nodes);
        coloring_temp.reserve(num_nodes);
        coloring = tuple_graph.get_labels();
        coloring_temp = coloring;

        EdgeLabels edge_labels = tuple_graph.get_edge_labels();
        EdgeLabels vertex_id = tuple_graph.get_vertex_id();
        EdgeLabels local = tuple_graph.get_local();

        ColorCounter color_map;

        unordered_map<Node, bool> check_1;
        unordered_map<Node, bool> check_2;
        unordered_map<Node, bool> check_3;

        if (algorithm == "localp" and num_iterations == 0) {
            for (Node v = 0; v < num_nodes; ++v) {
                Nodes neighbors(tuple_graph.get_neighbours(v));

                for (const Node &n: neighbors) {
                    const auto t = edge_labels.find(make_tuple(v, n));

                    TwoTuple p = node_to_two_tuple.find(n)->second;
                    Node a = std::get<0>(p);
                    Node b = std::get<1>(p);

                    if (t->second == 1) {
                        Label l = b;
                        l = AuxiliaryMethods::pairing(l, 1);
                        l = AuxiliaryMethods::pairing(l, coloring[n]);

                        Label e = a;
                        e = AuxiliaryMethods::pairing(e, b);
                        e = AuxiliaryMethods::pairing(e, 1);
                        const auto is = check_1.find(e);

                        if (is == check_1.end()) {
                            const auto it = color_map_1.find(l);

                            if (it == color_map_1.end()) {
                                color_map_1.insert({{l, 1}});
                            } else {
                                it->second++;
                            }

                            check_1.insert({{e, true}});
                        }
                    }

                    if (t->second == 2) {
                        Label l = a;

                        l = AuxiliaryMethods::pairing(l, 2);
                        l = AuxiliaryMethods::pairing(l, coloring[n]);

                        Label e = a;
                        e = AuxiliaryMethods::pairing(e, b);
                        e = AuxiliaryMethods::pairing(e, 2);
                        const auto is = check_2.find(e);

                        if (is == check_2.end()) {
                            const auto it = color_map_2.find(l);

                            if (it == color_map_2.end()) {
                                color_map_2.insert({{l, 1}});
                            } else {
                                it->second++;
                            }

                            check_2.insert({{e, true}});
                        }
                    }

                    if (t->second == 3) {
                        Label l = a;

                        l = AuxiliaryMethods::pairing(l, 3);
                        l = AuxiliaryMethods::pairing(l, coloring[n]);

                        Label e = a;
                        e = AuxiliaryMethods::pairing(e, b);
                        e = AuxiliaryMethods::pairing(e, 3);
                        const auto is = check_3.find(e);

                        if (is == check_3.end()) {
                            const auto it = color_map_3.find(l);

                            if (it == color_map_3.end()) {
                                color_map_3.insert({{l, 1}});
                            } else {
                                it->second++;
                            }

                            check_3.insert({{e, true}});
                        }
                    }
                }
            }
        }

        // Iteration.
        for (Node v = 0; v < num_nodes; ++v) {
            Label new_color = coloring[v];
            if (algorithm == "localp" and num_iterations == 0) {
                new_color = AuxiliaryMethods::pairing(coloring[v], color_map_1.find(coloring[v])->second);
                new_color = AuxiliaryMethods::pairing(new_color, color_map_2.find(coloring[v])->second);
                new_color = AuxiliaryMethods::pairing(new_color, color_map_3.find(coloring[v])->second);
            }


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
                uint dg = 0;
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
                    } else {
                        const auto s = vertex_id.find(make_tuple(v, n));
                        const auto pos(id_to_position_global.find(s->second));
                        if (pos != id_to_position_global.end()) {
                            set_m_global[pos->second].push_back(l);
                        } else {
                            id_to_position_global.insert({{s->second, dg}});
                            set_m_global.push_back(vector<Label>());
                            set_m_global[dg].push_back(l);
                            dg++;
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


                for (auto &m: set_m_global) {
                    if (m.size() != 0) {
                        sort(m.begin(), m.end());
                        new_color = m.back();
                        m.pop_back();
                        for (const Label &c: m) {
                            new_color = AuxiliaryMethods::pairing(new_color, c);
                        }
                        colors_global.push_back(new_color);
                    }
                }
                sort(colors_global.begin(), colors_global.end());

                for (auto &c: colors_global) {
                    colors_local.push_back(c);
                }

                Label ll = coloring[v];
                ll = AuxiliaryMethods::pairing(ll, color_map_1.find(coloring[v])->second);
                ll = AuxiliaryMethods::pairing(ll, color_map_2.find(coloring[v])->second);

                if (algorithm == "localp" and num_iterations == h) {
                    colors_local.push_back(ll);
                } else {
                    colors_local.push_back(coloring[v]);
                }

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

            unordered_map<Node, bool> check_1;
            unordered_map<Node, bool> check_2;
            unordered_map<Node, bool> check_3;

            if (algorithm == "localp" and num_iterations == h) {
                for (Node v = 0; v < num_nodes; ++v) {
                    Nodes neighbors(tuple_graph.get_neighbours(v));

                    for (const Node &n: neighbors) {
                        const auto t = edge_labels.find(make_tuple(v, n));

                        TwoTuple p = node_to_two_tuple.find(n)->second;
                        Node a = std::get<0>(p);
                        Node b = std::get<1>(p);

                        if (t->second == 1) {
                            Label l = b;
                            l = AuxiliaryMethods::pairing(l, 1);
                            l = AuxiliaryMethods::pairing(l, coloring[n]);

                            Label e = a;
                            e = AuxiliaryMethods::pairing(e, b);
                            e = AuxiliaryMethods::pairing(e, 1);
                            const auto is = check_1.find(e);

                            if (is == check_1.end()) {
                                const auto it = color_map_1.find(l);

                                if (it == color_map_1.end()) {
                                    color_map_1.insert({{l, 1}});
                                } else {
                                    it->second++;
                                }

                                check_1.insert({{e, true}});
                            }
                        }

                        if (t->second == 2) {
                            Label l = a;

                            l = AuxiliaryMethods::pairing(l, 2);
                            l = AuxiliaryMethods::pairing(l, coloring[n]);

                            Label e = a;
                            e = AuxiliaryMethods::pairing(e, b);
                            e = AuxiliaryMethods::pairing(e, 2);
                            const auto is = check_2.find(e);

                            if (is == check_2.end()) {
                                const auto it = color_map_2.find(l);

                                if (it == color_map_2.end()) {
                                    color_map_2.insert({{l, 1}});
                                } else {
                                    it->second++;
                                }

                                check_2.insert({{e, true}});
                            }
                        }

                        if (t->second == 3) {
                            Label l = a;

                            l = AuxiliaryMethods::pairing(l, 3);
                            l = AuxiliaryMethods::pairing(l, coloring[n]);

                            Label e = a;
                            e = AuxiliaryMethods::pairing(e, b);
                            e = AuxiliaryMethods::pairing(e, 3);
                            const auto is = check_3.find(e);

                            if (is == check_3.end()) {
                                const auto it = color_map_3.find(l);

                                if (it == color_map_3.end()) {
                                    color_map_3.insert({{l, 1}});
                                } else {
                                    it->second++;
                                }

                                check_3.insert({{e, true}});
                            }
                        }
                    }
                }
            }
        }

        return color_map;
    }

    ColorCounter GenerateThree::compute_colors_simple(const Graph &g, const uint num_iterations, const bool use_labels,
                                                      const string algorithm) {

        Graph tuple_graph(false);
        if (algorithm == "local" or algorithm == "localp") {
            tuple_graph = generate_local_graph(g, use_labels);
        } else if (algorithm == "wl") {
            tuple_graph = generate_global_graph(g, use_labels);
        } else if (algorithm == "malkin") {
            tuple_graph = generate_global_graph_malkin(g, use_labels);
        }

        ColorCounter color_map_1;
        ColorCounter color_map_2;
        ColorCounter color_map_3;


        unordered_map<Node, ThreeTuple> node_to_three_tuple;
        if (algorithm == "localp") {
            node_to_three_tuple = tuple_graph.get_node_to_three_tuple();
        }

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
        unordered_map<Label, bool> check_1;
        unordered_map<Label, bool> check_2;
        unordered_map<Label, bool> check_3;

        if (algorithm == "localp" and num_iterations == 0) {
            for (Node v = 0; v < num_nodes; ++v) {
                Nodes neighbors(tuple_graph.get_neighbours(v));

                for (const Node &n: neighbors) {
                    const auto t = edge_labels.find(make_tuple(v, n));

                    ThreeTuple p = node_to_three_tuple.find(n)->second;
                    Node a = std::get<0>(p);
                    Node b = std::get<1>(p);
                    Node c = std::get<2>(p);

                    if (t->second == 1) {
                        Label l = b;
                        l = AuxiliaryMethods::pairing(l, c);
                        l = AuxiliaryMethods::pairing(l, 1);
                        l = AuxiliaryMethods::pairing(l, coloring[n]);

                        Label e = a;
                        e = AuxiliaryMethods::pairing(e, b);
                        e = AuxiliaryMethods::pairing(e, c);
                        e = AuxiliaryMethods::pairing(e, 1);
                        const auto is = check_1.find(e);

                        if (is == check_1.end()) {
                            const auto it = color_map_1.find(l);

                            if (it == color_map_1.end()) {
                                color_map_1.insert({{l, 1}});
                            } else {
                                it->second++;
                            }

                            check_1.insert({{e, true}});
                        }
                    }

                    if (t->second == 2) {
                        Label l = a;
                        l = AuxiliaryMethods::pairing(l, c);
                        l = AuxiliaryMethods::pairing(l, 2);
                        l = AuxiliaryMethods::pairing(l, coloring[n]);

                        Label e = a;
                        e = AuxiliaryMethods::pairing(e, b);
                        e = AuxiliaryMethods::pairing(e, c);
                        e = AuxiliaryMethods::pairing(e, 2);
                        const auto is = check_2.find(e);

                        if (is == check_2.end()) {
                            const auto it = color_map_2.find(l);

                            if (it == color_map_2.end()) {
                                color_map_2.insert({{l, 1}});
                            } else {
                                it->second++;
                            }

                            check_2.insert({{e, true}});
                        }
                    }

                    if (t->second == 3) {
                        Label l = a;
                        l = AuxiliaryMethods::pairing(l, b);
                        l = AuxiliaryMethods::pairing(l, 3);
                        l = AuxiliaryMethods::pairing(l, coloring[n]);

                        Label e = a;
                        e = AuxiliaryMethods::pairing(e, b);
                        e = AuxiliaryMethods::pairing(e, c);
                        e = AuxiliaryMethods::pairing(e, 3);
                        const auto is = check_3.find(e);

                        if (is == check_3.end()) {
                            const auto it = color_map_3.find(l);

                            if (it == color_map_3.end()) {
                                color_map_3.insert({{l, 1}});
                            } else {
                                it->second++;
                            }

                            check_3.insert({{e, true}});
                        }
                    }
                }
            }
        }

        // Iteration 0.
        for (Node v = 0; v < num_nodes; ++v) {
            Label new_color = coloring[v];

            if (algorithm == "localp" and num_iterations == 0) {
                new_color = AuxiliaryMethods::pairing(coloring[v], color_map_1.find(coloring[v])->second);
                new_color = AuxiliaryMethods::pairing(new_color, color_map_2.find(coloring[v])->second);
                new_color = AuxiliaryMethods::pairing(new_color, color_map_3.find(coloring[v])->second);
            }


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
                            if (algorithm == "localp" and num_iterations == h) {
                                set_m_local[0].push_back(
                                        AuxiliaryMethods::pairing(coloring[n], color_map_1.find(coloring[n])->second));
                            } else {
                                set_m_local[0].push_back(coloring[n]);
                            }
                        }
                        if (label == 2) {
                            if (algorithm == "localp" and num_iterations == h) {
                                set_m_local[1].push_back(
                                        AuxiliaryMethods::pairing(coloring[n], color_map_2.find(coloring[n])->second));
                            } else {
                                set_m_local[1].push_back(coloring[n]);
                            }
                        }
                        if (label == 3) {
                            if (algorithm == "localp" and num_iterations == h) {
                                set_m_local[2].push_back(
                                        AuxiliaryMethods::pairing(coloring[n], color_map_3.find(coloring[n])->second));
                            } else {
                                set_m_local[2].push_back(coloring[n]);
                            }
                        }
                    } else {  // Global neighbor.
                        if (label == 1) {
                            set_m_global[0].push_back(coloring[n]);
                        }
                        if (label == 2) {
                            set_m_global[1].push_back(coloring[n]);
                        }
                        if (label == 3) {
                            set_m_global[2].push_back(coloring[n]);
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


                for (auto &m: set_m_global) {
                    if (m.size() != 0) {
                        sort(m.begin(), m.end());
                        new_color = m.back();
                        m.pop_back();
                        for (const Label &c: m) {
                            new_color = AuxiliaryMethods::pairing(new_color, c);
                        }
                        colors_global.push_back(new_color);
                    }
                }
                sort(colors_global.begin(), colors_global.end());

                for (auto &c: colors_global) {
                    colors_local.push_back(c);
                }


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

            unordered_map<Label, bool> check_1;
            unordered_map<Label, bool> check_2;
            unordered_map<Label, bool> check_3;

            if (algorithm == "localp" and num_iterations == h) {
                for (Node v = 0; v < num_nodes; ++v) {
                    Nodes neighbors(tuple_graph.get_neighbours(v));

                    for (const Node &n: neighbors) {
                        const auto t = edge_labels.find(make_tuple(v, n));

                        ThreeTuple p = node_to_three_tuple.find(n)->second;
                        Node a = std::get<0>(p);
                        Node b = std::get<1>(p);
                        Node c = std::get<2>(p);

                        if (t->second == 1) {
                            Label l = b;
                            l = AuxiliaryMethods::pairing(l, c);
                            l = AuxiliaryMethods::pairing(l, 1);
                            l = AuxiliaryMethods::pairing(l, coloring[n]);

                            Label e = a;
                            e = AuxiliaryMethods::pairing(e, b);
                            e = AuxiliaryMethods::pairing(e, c);
                            e = AuxiliaryMethods::pairing(e, 1);
                            const auto is = check_1.find(e);

                            if (is == check_1.end()) {
                                const auto it = color_map_1.find(l);

                                if (it == color_map_1.end()) {
                                    color_map_1.insert({{l, 1}});
                                } else {
                                    it->second++;
                                }

                                check_1.insert({{e, true}});
                            }
                        }

                        if (t->second == 2) {
                            Label l = a;
                            l = AuxiliaryMethods::pairing(l, c);
                            l = AuxiliaryMethods::pairing(l, 2);
                            l = AuxiliaryMethods::pairing(l, coloring[n]);

                            Label e = a;
                            e = AuxiliaryMethods::pairing(e, b);
                            e = AuxiliaryMethods::pairing(e, c);
                            e = AuxiliaryMethods::pairing(e, 2);
                            const auto is = check_2.find(e);

                            if (is == check_2.end()) {
                                const auto it = color_map_2.find(l);

                                if (it == color_map_2.end()) {
                                    color_map_2.insert({{l, 1}});
                                } else {
                                    it->second++;
                                }

                                check_2.insert({{e, true}});
                            }
                        }

                        if (t->second == 3) {
                            Label l = a;
                            l = AuxiliaryMethods::pairing(l, b);
                            l = AuxiliaryMethods::pairing(l, 3);
                            l = AuxiliaryMethods::pairing(l, coloring[n]);

                            Label e = a;
                            e = AuxiliaryMethods::pairing(e, b);
                            e = AuxiliaryMethods::pairing(e, c);
                            e = AuxiliaryMethods::pairing(e, 3);
                            const auto is = check_3.find(e);

                            if (is == check_3.end()) {
                                const auto it = color_map_3.find(l);

                                if (it == color_map_3.end()) {
                                    color_map_3.insert({{l, 1}});
                                } else {
                                    it->second++;
                                }

                                check_3.insert({{e, true}});
                            }
                        }
                    }
                }
            }
        }

        return color_map;
    }


    Graph GenerateThree::generate_local_graph(const Graph &g, const bool use_labels) {
        size_t num_nodes = g.get_num_nodes();
        // New graph to be generated.
        Graph three_tuple_graph(false);

        // Maps node in two set graph to correponding two set.
        unordered_map<Node, ThreeTuple> node_to_three_tuple;
        // Inverse of the above map.
        unordered_map<ThreeTuple, Node> three_tuple_to_node;
        unordered_map<Edge, uint> edge_type;
        // Manages vertex ids
        unordered_map<Edge, uint> vertex_id;
        unordered_map<Edge, uint> local;

        // Create a node for each two set.
        Labels labels;
        Labels tuple_labels;
        if (use_labels) {
            labels = g.get_labels();
        }

        size_t num_three_tuples = 0;
        for (Node i = 0; i < num_nodes; ++i) {
            for (Node j = 0; j < num_nodes; ++j) {
                for (Node k = 0; k < num_nodes; ++k) {
                    three_tuple_graph.add_node();

                    node_to_three_tuple.insert({{num_three_tuples, make_tuple(i, j, k)}});
                    three_tuple_to_node.insert({{make_tuple(i, j, k), num_three_tuples}});
                    num_three_tuples++;

                    Label c_i = 1;
                    Label c_j = 2;
                    Label c_k = 3;

                    if (use_labels) {
                        c_i = AuxiliaryMethods::pairing(labels[i] + 1, c_i);
                        c_j = AuxiliaryMethods::pairing(labels[j] + 1, c_j);
                        c_k = AuxiliaryMethods::pairing(labels[k] + 1, c_k);
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
                    Label new_color = AuxiliaryMethods::pairing(new_color_0, new_color_1);
                    tuple_labels.push_back(new_color);
                }
            }
        }

        for (Node i = 0; i < num_three_tuples; ++i) {
            // Get nodes of original graph corresponding to two tuple i.
            ThreeTuple p = node_to_three_tuple.find(i)->second;
            Node v = std::get<0>(p);
            Node w = std::get<1>(p);
            Node u = std::get<2>(p);

            // Exchange first node.
            Nodes v_neighbors = g.get_neighbours(v);
            for (const auto &v_n: v_neighbors) {
                unordered_map<ThreeTuple, Node>::const_iterator t;
                t = three_tuple_to_node.find(make_tuple(v_n, w, u));

                three_tuple_graph.add_edge(i, t->second);
                edge_type.insert({{make_tuple(i, t->second), 1}});
                vertex_id.insert({{make_tuple(i, t->second), v_n}});
                local.insert({{make_tuple(i, t->second), 1}});
            }

            // Exchange second node.
            Nodes w_neighbors = g.get_neighbours(w);
            for (const auto &w_n: w_neighbors) {
                unordered_map<ThreeTuple, Node>::const_iterator t;
                t = three_tuple_to_node.find(make_tuple(v, w_n, u));

                three_tuple_graph.add_edge(i, t->second);
                edge_type.insert({{make_tuple(i, t->second), 2}});
                vertex_id.insert({{make_tuple(i, t->second), w_n}});
                local.insert({{make_tuple(i, t->second), 1}});
            }

            // Exchange third node.
            Nodes u_neighbors = g.get_neighbours(u);
            for (const auto &u_n: u_neighbors) {
                unordered_map<ThreeTuple, Node>::const_iterator t;
                t = three_tuple_to_node.find(make_tuple(v, w, u_n));

                three_tuple_graph.add_edge(i, t->second);
                edge_type.insert({{make_tuple(i, t->second), 3}});
                vertex_id.insert({{make_tuple(i, t->second), u_n}});
                local.insert({{make_tuple(i, t->second), 1}});
            }
        }

        three_tuple_graph.set_edge_labels(edge_type);
        three_tuple_graph.set_labels(tuple_labels);
        three_tuple_graph.set_vertex_id(vertex_id);
        three_tuple_graph.set_local(local);
        three_tuple_graph.set_node_to_three_tuple(node_to_three_tuple);

        return three_tuple_graph;
    }

    Graph GenerateThree::generate_global_graph(const Graph &g, const bool use_labels) {
        size_t num_nodes = g.get_num_nodes();
        // New graph to be generated.
        Graph three_tuple_graph(false);

        // Maps node in two set graph to correponding two set.
        unordered_map<Node, ThreeTuple> node_to_three_tuple;
        // Inverse of the above map.
        unordered_map<ThreeTuple, Node> three_tuple_to_node;
        unordered_map<Edge, uint> edge_type;
        unordered_map<Edge, uint> vertex_id;
        unordered_map<Edge, uint> local;

        // Create a node for each two set.
        Labels labels;
        Labels tuple_labels;
        if (use_labels) {
            labels = g.get_labels();
        }

        size_t num_three_tuples = 0;
        for (Node i = 0; i < num_nodes; ++i) {
            for (Node j = 0; j < num_nodes; ++j) {
                for (Node k = 0; k < num_nodes; ++k) {
                    three_tuple_graph.add_node();

                    node_to_three_tuple.insert({{num_three_tuples, make_tuple(i, j, k)}});
                    three_tuple_to_node.insert({{make_tuple(i, j, k), num_three_tuples}});
                    num_three_tuples++;

                    Label c_i = 1;
                    Label c_j = 2;
                    Label c_k = 3;

                    if (use_labels) {
                        c_i = AuxiliaryMethods::pairing(labels[i] + 1, c_i);
                        c_j = AuxiliaryMethods::pairing(labels[j] + 1, c_j);
                        c_k = AuxiliaryMethods::pairing(labels[k] + 1, c_k);
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
                    Label new_color = AuxiliaryMethods::pairing(new_color_0, new_color_1);
                    tuple_labels.push_back(new_color);
                }
            }
        }

        for (Node i = 0; i < num_three_tuples; ++i) {
            // Get nodes of original graph corresponding to two tuple i.
            ThreeTuple p = node_to_three_tuple.find(i)->second;
            Node v = std::get<0>(p);
            Node w = std::get<1>(p);
            Node u = std::get<2>(p);

            // Exchange first node.
            // Iterate over nodes.
            for (Node v_i = 0; v_i < num_nodes; ++v_i) {
                unordered_map<ThreeTuple, Node>::const_iterator t;
                t = three_tuple_to_node.find(make_tuple(v_i, w, u));
                three_tuple_graph.add_edge(i, t->second);
                edge_type.insert({{make_tuple(i, t->second), 1}});
                vertex_id.insert({{make_tuple(i, t->second), v_i}});
                local.insert({{make_tuple(i, t->second), 1}});
            }

            // Exchange second node.
            // Iterate over nodes.
            for (Node v_i = 0; v_i < num_nodes; ++v_i) {
                unordered_map<ThreeTuple, Node>::const_iterator t;
                t = three_tuple_to_node.find(make_tuple(v, v_i, u));
                three_tuple_graph.add_edge(i, t->second);
                edge_type.insert({{make_tuple(i, t->second), 2}});
                vertex_id.insert({{make_tuple(i, t->second), v_i}});
                local.insert({{make_tuple(i, t->second), 1}});
            }

            // Exchange second node.
            // Iterate over nodes.
            for (Node v_i = 0; v_i < num_nodes; ++v_i) {
                unordered_map<ThreeTuple, Node>::const_iterator t;
                t = three_tuple_to_node.find(make_tuple(v, w, v_i));
                three_tuple_graph.add_edge(i, t->second);
                edge_type.insert({{make_tuple(i, t->second), 3}});
                vertex_id.insert({{make_tuple(i, t->second), v_i}});
                local.insert({{make_tuple(i, t->second), 1}});
            }
        }

        three_tuple_graph.set_edge_labels(edge_type);
        three_tuple_graph.set_labels(tuple_labels);
        three_tuple_graph.set_vertex_id(vertex_id);
        three_tuple_graph.set_local(local);

        return three_tuple_graph;
    }


    Graph GenerateThree::generate_global_graph_malkin(const Graph &g, const bool use_labels) {
        size_t num_nodes = g.get_num_nodes();
        // New graph to be generated.
        Graph three_tuple_graph(false);

        // Maps node in two set graph to correponding two set.
        unordered_map<Node, ThreeTuple> node_to_three_tuple;
        // Inverse of the above map.
        unordered_map<ThreeTuple, Node> three_tuple_to_node;
        unordered_map<Edge, uint> edge_type;
        // Manages vertex ids
        unordered_map<Edge, uint> vertex_id;
        unordered_map<Edge, uint> local;

        // Create a node for each two set.
        Labels labels;
        Labels tuple_labels;
        if (use_labels) {
            labels = g.get_labels();
        }
        size_t num_three_tuples = 0;
        for (Node i = 0; i < num_nodes; ++i) {
            for (Node j = 0; j < num_nodes; ++j) {
                for (Node k = 0; k < num_nodes; ++k) {
                    three_tuple_graph.add_node();

                    // Map each pair to node in two set graph and also inverse.
                    node_to_three_tuple.insert({{num_three_tuples, make_tuple(i, j, k)}});
                    three_tuple_to_node.insert({{make_tuple(i, j, k), num_three_tuples}});
                    num_three_tuples++;

                    Label c_i = 1;
                    Label c_j = 2;
                    Label c_k = 3;

                    if (use_labels) {
                        c_i = AuxiliaryMethods::pairing(labels[i] + 1, c_i);
                        c_j = AuxiliaryMethods::pairing(labels[j] + 1, c_j);
                        c_k = AuxiliaryMethods::pairing(labels[k] + 1, c_k);
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
                    Label new_color = AuxiliaryMethods::pairing(new_color_0, new_color_1);
                    tuple_labels.push_back(new_color);
                }
            }
        }

        for (Node i = 0; i < num_three_tuples; ++i) {
            ThreeTuple p = node_to_three_tuple.find(i)->second;
            Node v = std::get<0>(p);
            Node w = std::get<1>(p);
            Node u = std::get<2>(p);

            // Exchange first node.
            // Iterate over nodes.
            for (Node v_i = 0; v_i < num_nodes; ++v_i) {
                unordered_map<ThreeTuple, Node>::const_iterator t;
                t = three_tuple_to_node.find(make_tuple(v_i, w, u));
                three_tuple_graph.add_edge(i, t->second);

                // Local vs. global edge.
                if (g.has_edge(v, v_i)) {
                    edge_type.insert({{make_tuple(i, t->second), 1}});
                    vertex_id.insert({{make_tuple(i, t->second), v_i}});
                    local.insert({{make_tuple(i, t->second), 1}});
                } else {
                    edge_type.insert({{make_tuple(i, t->second), 1}});
                    vertex_id.insert({{make_tuple(i, t->second), v_i}});
                    local.insert({{make_tuple(i, t->second), 2}});
                }
            }

            // Exchange second node.
            // Iterate over nodes.
            for (Node w_i = 0; w_i < num_nodes; ++w_i) {
                unordered_map<ThreeTuple, Node>::const_iterator t;
                t = three_tuple_to_node.find(make_tuple(v, w_i, u));
                three_tuple_graph.add_edge(i, t->second);

                // Local vs. global edge.
                if (g.has_edge(w, w_i)) {
                    edge_type.insert({{make_tuple(i, t->second), 2}});
                    vertex_id.insert({{make_tuple(i, t->second), w_i}});
                    local.insert({{make_tuple(i, t->second), 1}});
                } else {
                    edge_type.insert({{make_tuple(i, t->second), 2}});
                    vertex_id.insert({{make_tuple(i, t->second), w_i}});
                    local.insert({{make_tuple(i, t->second), 2}});
                }
            }

            // Exchange three node.
            // Iterate over nodes.
            for (Node u_i = 0; u_i < num_nodes; ++u_i) {
                unordered_map<ThreeTuple, Node>::const_iterator t;
                t = three_tuple_to_node.find(make_tuple(v, w, u_i));
                three_tuple_graph.add_edge(i, t->second);

                // Local vs. global edge.
                if (g.has_edge(u, u_i)) {
                    edge_type.insert({{make_tuple(i, t->second), 3}});
                    vertex_id.insert({{make_tuple(i, t->second), u_i}});
                    local.insert({{make_tuple(i, t->second), 1}});
                } else {
                    edge_type.insert({{make_tuple(i, t->second), 3}});
                    vertex_id.insert({{make_tuple(i, t->second), u_i}});
                    local.insert({{make_tuple(i, t->second), 2}});
                }
            }
        }

        three_tuple_graph.set_edge_labels(edge_type);
        three_tuple_graph.set_labels(tuple_labels);
        three_tuple_graph.set_vertex_id(vertex_id);
        three_tuple_graph.set_local(local);

        return three_tuple_graph;
    }

    GenerateThree::~GenerateThree() {}
}
