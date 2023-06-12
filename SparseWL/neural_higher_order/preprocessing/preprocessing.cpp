
#include <cstdio>
#include <iostream>

#include "src/AuxiliaryMethods.h"
#include "src/Graph.h"

#ifdef __linux__
// #include <pybind11/pybind11.h>
// #include <pybind11/eigen.h>
// #include <pybind11/stl.h>
#include </home/wyb/miniconda3/envs/sparsewl/lib/python3.8/site-packages/pybind11/include/pybind11/eigen.h>
#include </home/wyb/miniconda3/envs/sparsewl/lib/python3.8/site-packages/pybind11/include/pybind11/pybind11.h>
#include </home/wyb/miniconda3/envs/sparsewl/lib/python3.8/site-packages/pybind11/include/pybind11/stl.h>
#else
// MacOS.
#include </usr/local/include/pybind11/eigen.h>
#include </usr/local/include/pybind11/pybind11.h>
#include </usr/local/include/pybind11/stl.h>

#endif

namespace py = pybind11;
using namespace std;
using namespace GraphLibrary;

using namespace std;

tuple<Attributes, Attributes, Attributes> get_attributes(const Graph &g)
{
    size_t num_nodes = g.get_num_nodes();

    // Get continious node and edge information.
    Attributes attributes;
    attributes = g.get_attributes();

    EdgeAttributes edge_attributes;
    edge_attributes = g.get_edge_attributes();

    Attributes first;
    Attributes second;
    Attributes third;

    Node num_two_tuples = 0;
    for (Node i = 0; i < num_nodes; ++i)
    {
        for (Node j = 0; j < num_nodes; ++j)
        {
            // Map each pair to node in two set graph and also inverse.

            Attribute attr_i = attributes[i];
            Attribute attr_j = attributes[j];

            Attribute e_attr_ij;
            if (g.has_edge(i, j))
            {
                e_attr_ij = edge_attributes.find(std::make_pair(i, j))->second;
                // cout << attr_i.size() << " " << e_attr_ij.size() << endl;
            }
            else
            {
                e_attr_ij = vector<float>({{0, 0, 0, 0}});
            }

            first.push_back(attr_i);
            second.push_back(attr_j);
            third.push_back(e_attr_ij);
        }
    }

    return std::make_tuple(first, second, third);
}

pair<vector<vector<uint>>, vector<vector<uint>>>

generate_local_sparse_am(const Graph &g, const bool use_labels,
                         const bool use_edge_labels)
{
    size_t num_nodes = g.get_num_nodes();
    // New graph to be generated.
    Graph two_tuple_graph(false);

    // Maps node in two set graph to corresponding two tuple.
    unordered_map<Node, TwoTuple> node_to_two_tuple;
    // Inverse of the above map.
    unordered_map<TwoTuple, Node> two_tuple_to_node;
    // Manages edges labels.
    unordered_map<Edge, uint> edge_type;
    // Manages vertex ids
    unordered_map<Edge, uint> vertex_id;
    unordered_map<Edge, uint> local;

    // Create a node for each two set.
    Labels labels;
    Labels tuple_labels;
    if (use_labels)
    {
        labels = g.get_labels();
    }

    EdgeLabels edge_labels;
    if (use_edge_labels)
    {
        edge_labels = g.get_edge_labels();
    }

    Node num_two_tuples = 0;
    for (Node i = 0; i < num_nodes; ++i)
    {
        for (Node j = 0; j < num_nodes; ++j)
        {
            two_tuple_graph.add_node();

            // Map each pair to node in two set graph and also inverse.
            node_to_two_tuple.insert({{num_two_tuples, make_tuple(i, j)}});
            two_tuple_to_node.insert({{make_tuple(i, j), num_two_tuples}});
            num_two_tuples++;

            Label c_i = 1;
            Label c_j = 2;
            if (use_labels)
            {
                c_i = AuxiliaryMethods::pairing(labels[i] + 1, c_i);
                c_j = AuxiliaryMethods::pairing(labels[j] + 1, c_j);
            }

            Label c;
            if (g.has_edge(i, j))
            {
                if (use_edge_labels)
                {
                    auto s = edge_labels.find(make_tuple(i, j));
                    c = AuxiliaryMethods::pairing(3, s->second);
                }
                else
                {
                    c = 3;
                }
            }
            else if (i == j)
            {
                c = 1;
            }
            else
            {
                c = 2;
            }

            Label new_color =
                AuxiliaryMethods::pairing(AuxiliaryMethods::pairing(c_i, c_j), c);
            tuple_labels.push_back(new_color);
        }
    }

    vector<vector<uint>> nonzero_compenents_1;
    vector<vector<uint>> nonzero_compenents_2;

    for (Node i = 0; i < num_two_tuples; ++i)
    {
        // Get nodes of original graph corresponding to two tuple i.
        TwoTuple p = node_to_two_tuple.find(i)->second;
        Node v = std::get<0>(p);
        Node w = std::get<1>(p);

        // Exchange first node.
        Nodes v_neighbors = g.get_neighbours(v);
        for (Node v_n : v_neighbors)
        {
            unordered_map<TwoTuple, Node>::const_iterator t =
                two_tuple_to_node.find(make_tuple(v_n, w));
            two_tuple_graph.add_edge(i, t->second);
            edge_type.insert({{make_tuple(i, t->second), 1}});
            vertex_id.insert({{make_tuple(i, t->second), v_n}});
            local.insert({{make_tuple(i, t->second), 1}});

            nonzero_compenents_1.push_back({{i, t->second}});
        }

        // Exchange second node.
        Nodes w_neighbors = g.get_neighbours(w);
        for (Node w_n : w_neighbors)
        {
            unordered_map<TwoTuple, Node>::const_iterator t =
                two_tuple_to_node.find(make_tuple(v, w_n));
            two_tuple_graph.add_edge(i, t->second);
            edge_type.insert({{make_tuple(i, t->second), 2}});
            vertex_id.insert({{make_tuple(i, t->second), w_n}});
            local.insert({{make_tuple(i, t->second), 1}});

            nonzero_compenents_2.push_back({{i, t->second}});
        }
    }

    return std::make_pair(nonzero_compenents_1, nonzero_compenents_2);
}

pair<vector<vector<uint>>, vector<vector<uint>>>

generate_local_sparse_am_connected(const Graph &g, const bool use_labels,
                                   const bool use_edge_labels)
{
    size_t num_nodes = g.get_num_nodes();
    // New graph to be generated.
    Graph two_tuple_graph(false);

    // Maps node in two set graph to corresponding two tuple.
    unordered_map<Node, TwoTuple> node_to_two_tuple;
    // Inverse of the above map.
    unordered_map<TwoTuple, Node> two_tuple_to_node;
    // Manages edges labels.
    unordered_map<Edge, uint> edge_type;
    // Manages vertex ids
    unordered_map<Edge, uint> vertex_id;
    unordered_map<Edge, uint> local;

    // Create a node for each two set.
    Labels labels;
    Labels tuple_labels;
    if (use_labels)
    {
        labels = g.get_labels();
    }

    EdgeLabels edge_labels;
    if (use_edge_labels)
    {
        edge_labels = g.get_edge_labels();
    }

    Node num_two_tuples = 0;
    for (Node i = 0; i < num_nodes; ++i)
    {
        for (Node j = 0; j < num_nodes; ++j)
        {
            if (true)
            {
                // if (g.has_edge(j,i ) or g.has_edge(i,j) or (i == j)) {
                two_tuple_graph.add_node();

                // Map each pair to node in two set graph and also inverse.
                node_to_two_tuple.insert({{num_two_tuples, make_tuple(i, j)}});
                two_tuple_to_node.insert({{make_tuple(i, j), num_two_tuples}});
                num_two_tuples++;

                Label c_i = 1;
                Label c_j = 2;
                if (use_labels)
                {
                    c_i = AuxiliaryMethods::pairing(labels[i] + 1, c_i);
                    c_j = AuxiliaryMethods::pairing(labels[j] + 1, c_j);
                }

                Label c;
                if (g.has_edge(i, j))
                {
                    if (use_edge_labels)
                    {
                        auto s = edge_labels.find(make_tuple(i, j));
                        c = AuxiliaryMethods::pairing(3, s->second);
                    }
                    else
                    {
                        c = 3;
                    }
                }
                else if (i == j)
                {
                    c = 1;
                }
                else
                {
                    c = 2;
                }

                Label new_color =
                    AuxiliaryMethods::pairing(AuxiliaryMethods::pairing(c_i, c_j), c);
                tuple_labels.push_back(new_color);
            }
        }
    }

    vector<vector<uint>> nonzero_compenents_1;
    vector<vector<uint>> nonzero_compenents_2;

    for (Node i = 0; i < num_two_tuples; ++i)
    {
        // Get nodes of original graph corresponding to two tuple i.
        TwoTuple p = node_to_two_tuple.find(i)->second;
        Node v = std::get<0>(p);
        Node w = std::get<1>(p);

        // Exchange first node.
        Nodes v_neighbors = g.get_neighbours(v);
        for (Node v_n : v_neighbors)
        {
            unordered_map<TwoTuple, Node>::const_iterator t =
                two_tuple_to_node.find(make_tuple(v_n, w));
            if (g.has_edge(v_n, w) or g.has_edge(w, v_n) or (w == v_n))
            {
                two_tuple_graph.add_edge(i, t->second);
                edge_type.insert({{make_tuple(i, t->second), 1}});
                vertex_id.insert({{make_tuple(i, t->second), v_n}});
                local.insert({{make_tuple(i, t->second), 1}});
                nonzero_compenents_1.push_back({{i, t->second}});
            }
        }

        // Exchange second node.
        Nodes w_neighbors = g.get_neighbours(w);
        for (Node w_n : w_neighbors)
        {
            unordered_map<TwoTuple, Node>::const_iterator t =
                two_tuple_to_node.find(make_tuple(v, w_n));
            if (g.has_edge(w_n, v) or g.has_edge(v, w_n) or (v == w_n))
            {
                two_tuple_graph.add_edge(i, t->second);
                edge_type.insert({{make_tuple(i, t->second), 2}});
                vertex_id.insert({{make_tuple(i, t->second), w_n}});
                local.insert({{make_tuple(i, t->second), 1}});
                nonzero_compenents_2.push_back({{i, t->second}});
            }
        }
    }

    return std::make_pair(nonzero_compenents_1, nonzero_compenents_2);
}

pair<vector<vector<uint>>, vector<vector<uint>>>

generate_local_sparse_am_con(const Graph &g, const bool use_labels,
                             const bool use_edge_labels)
{
    size_t num_nodes = g.get_num_nodes();
    // New graph to be generated.
    Graph two_tuple_graph(false);

    // Maps node in two set graph to corresponding two tuple.
    unordered_map<Node, TwoTuple> node_to_two_tuple;
    // Inverse of the above map.
    unordered_map<TwoTuple, Node> two_tuple_to_node;
    // Manages edges labels.
    unordered_map<Edge, uint> edge_type;
    // Manages vertex ids
    unordered_map<Edge, uint> vertex_id;
    unordered_map<Edge, uint> local;

    // Create a node for each two set.
    Labels labels;
    Labels tuple_labels;
    if (use_labels)
    {
        labels = g.get_labels();
    }

    EdgeLabels edge_labels;
    if (use_edge_labels)
    {
        edge_labels = g.get_edge_labels();
    }

    Node num_two_tuples = 0;
    for (Node i = 0; i < num_nodes; ++i)
    {
        for (Node j = 0; j < num_nodes; ++j)
        {
            if (g.has_edge(j, i) or g.has_edge(i, j) or (i == j))
            {
                two_tuple_graph.add_node();

                // Map each pair to node in two set graph and also inverse.
                node_to_two_tuple.insert({{num_two_tuples, make_tuple(i, j)}});
                two_tuple_to_node.insert({{make_tuple(i, j), num_two_tuples}});
                num_two_tuples++;

                Label c_i = 1;
                Label c_j = 2;
                if (use_labels)
                {
                    c_i = AuxiliaryMethods::pairing(labels[i] + 1, c_i);
                    c_j = AuxiliaryMethods::pairing(labels[j] + 1, c_j);
                }

                Label c;
                if (g.has_edge(i, j))
                {
                    if (use_edge_labels)
                    {
                        auto s = edge_labels.find(make_tuple(i, j));
                        c = AuxiliaryMethods::pairing(3, s->second);
                    }
                    else
                    {
                        c = 3;
                    }
                }
                else if (i == j)
                {
                    c = 1;
                }
                else
                {
                    c = 2;
                }

                Label new_color =
                    AuxiliaryMethods::pairing(AuxiliaryMethods::pairing(c_i, c_j), c);
                tuple_labels.push_back(new_color);
            }
        }
    }

    vector<vector<uint>> nonzero_compenents_1;
    vector<vector<uint>> nonzero_compenents_2;

    for (Node i = 0; i < num_two_tuples; ++i)
    {
        // Get nodes of original graph corresponding to two tuple i.
        TwoTuple p = node_to_two_tuple.find(i)->second;
        Node v = std::get<0>(p);
        Node w = std::get<1>(p);

        // Exchange first node.
        Nodes v_neighbors = g.get_neighbours(v);
        for (Node v_n : v_neighbors)
        {
            unordered_map<TwoTuple, Node>::const_iterator t =
                two_tuple_to_node.find(make_tuple(v_n, w));
            if (g.has_edge(v_n, w) or g.has_edge(w, v_n) or (w == v_n))
            {
                two_tuple_graph.add_edge(i, t->second);
                edge_type.insert({{make_tuple(i, t->second), 1}});
                vertex_id.insert({{make_tuple(i, t->second), v_n}});
                local.insert({{make_tuple(i, t->second), 1}});
                nonzero_compenents_1.push_back({{i, t->second}});
            }
        }

        // Exchange second node.
        Nodes w_neighbors = g.get_neighbours(w);
        for (Node w_n : w_neighbors)
        {
            unordered_map<TwoTuple, Node>::const_iterator t =
                two_tuple_to_node.find(make_tuple(v, w_n));
            if (g.has_edge(w_n, v) or g.has_edge(v, w_n) or (v == w_n))
            {
                two_tuple_graph.add_edge(i, t->second);
                edge_type.insert({{make_tuple(i, t->second), 2}});
                vertex_id.insert({{make_tuple(i, t->second), w_n}});
                local.insert({{make_tuple(i, t->second), 1}});
                nonzero_compenents_2.push_back({{i, t->second}});
            }
        }
    }

    return std::make_pair(nonzero_compenents_1, nonzero_compenents_2);
}

pair<vector<vector<uint>>, vector<vector<uint>>>

generate_local_sparse_am_unc(const Graph &g, const bool use_labels,
                             const bool use_edge_labels)
{
    size_t num_nodes = g.get_num_nodes();
    // New graph to be generated.
    Graph two_tuple_graph(false);

    // Maps node in two set graph to corresponding two tuple.
    unordered_map<Node, TwoTuple> node_to_two_tuple;
    // Inverse of the above map.
    unordered_map<TwoTuple, Node> two_tuple_to_node;
    // Manages edges labels.
    unordered_map<Edge, uint> edge_type;
    // Manages vertex ids
    unordered_map<Edge, uint> vertex_id;
    unordered_map<Edge, uint> local;

    // Create a node for each two set.
    Labels labels;
    Labels tuple_labels;
    if (use_labels)
    {
        labels = g.get_labels();
    }

    EdgeLabels edge_labels;
    if (use_edge_labels)
    {
        edge_labels = g.get_edge_labels();
    }

    Node num_two_tuples = 0;
    for (Node i = 0; i < num_nodes; ++i)
    {
        for (Node j = 0; j < num_nodes; ++j)
        {
            // if (g.has_edge(j,i ) or g.has_edge(i,j) or (i == j)) {
            two_tuple_graph.add_node();

            // Map each pair to node in two set graph and also inverse.
            node_to_two_tuple.insert({{num_two_tuples, make_tuple(i, j)}});
            two_tuple_to_node.insert({{make_tuple(i, j), num_two_tuples}});
            num_two_tuples++;

            Label c_i = 1;
            Label c_j = 2;
            if (use_labels)
            {
                c_i = AuxiliaryMethods::pairing(labels[i] + 1, c_i);
                c_j = AuxiliaryMethods::pairing(labels[j] + 1, c_j);
            }

            Label c;
            if (g.has_edge(i, j))
            {
                if (use_edge_labels)
                {
                    auto s = edge_labels.find(make_tuple(i, j));
                    c = AuxiliaryMethods::pairing(3, s->second);
                }
                else
                {
                    c = 3;
                }
            }
            else if (i == j)
            {
                c = 1;
            }
            else
            {
                c = 2;
            }

            Label new_color =
                AuxiliaryMethods::pairing(AuxiliaryMethods::pairing(c_i, c_j), c);
            tuple_labels.push_back(new_color);
        }
    }
    //}

    vector<vector<uint>> nonzero_compenents_1;
    vector<vector<uint>> nonzero_compenents_2;

    for (Node i = 0; i < num_two_tuples; ++i)
    {
        // Get nodes of original graph corresponding to two tuple i.
        TwoTuple p = node_to_two_tuple.find(i)->second;
        Node v = std::get<0>(p);
        Node w = std::get<1>(p);

        // Exchange first node.
        Nodes v_neighbors = g.get_neighbours(v);
        for (Node v_n : v_neighbors)
        {
            unordered_map<TwoTuple, Node>::const_iterator t =
                two_tuple_to_node.find(make_tuple(v_n, w));
            if (g.has_edge(v_n, w) or g.has_edge(w, v_n) or (w == v_n))
            {
                two_tuple_graph.add_edge(i, t->second);
                edge_type.insert({{make_tuple(i, t->second), 1}});
                vertex_id.insert({{make_tuple(i, t->second), v_n}});
                local.insert({{make_tuple(i, t->second), 1}});
                nonzero_compenents_1.push_back({{i, t->second}});
            }
        }

        // Exchange second node.
        Nodes w_neighbors = g.get_neighbours(w);
        for (Node w_n : w_neighbors)
        {
            unordered_map<TwoTuple, Node>::const_iterator t =
                two_tuple_to_node.find(make_tuple(v, w_n));
            if (g.has_edge(w_n, v) or g.has_edge(v, w_n) or (v == w_n))
            {
                two_tuple_graph.add_edge(i, t->second);
                edge_type.insert({{make_tuple(i, t->second), 2}});
                vertex_id.insert({{make_tuple(i, t->second), w_n}});
                local.insert({{make_tuple(i, t->second), 1}});
                nonzero_compenents_2.push_back({{i, t->second}});
            }
        }
    }

    return std::make_pair(nonzero_compenents_1, nonzero_compenents_2);
}

vector<vector<uint>> generate_local_sparse_am_1(const Graph &g)
{
    size_t num_nodes = g.get_num_nodes();

    vector<vector<uint>> nonzero_compenents;

    for (Node v = 0; v < num_nodes; ++v)
    {
        // Exchange first node.
        Nodes v_neighbors = g.get_neighbours(v);
        for (Node v_n : v_neighbors)
        {
            nonzero_compenents.push_back({{v, v_n}});
        }
    }

    return nonzero_compenents;
}

pair<vector<vector<uint>>, vector<vector<uint>>>

generate_wl_sparse_am(const Graph &g, const bool use_labels,
                      const bool use_edge_labels)
{
    size_t num_nodes = g.get_num_nodes();
    // New graph to be generated.
    Graph two_tuple_graph(false);

    // Maps node in two set graph to corresponding two tuple.
    unordered_map<Node, TwoTuple> node_to_two_tuple;
    // Inverse of the above map.
    unordered_map<TwoTuple, Node> two_tuple_to_node;
    // Manages edges labels.
    unordered_map<Edge, uint> edge_type;
    // Manages vertex ids
    unordered_map<Edge, uint> vertex_id;
    unordered_map<Edge, uint> local;

    // Create a node for each two set.
    Labels labels;
    Labels tuple_labels;
    if (use_labels)
    {
        labels = g.get_labels();
    }

    EdgeLabels edge_labels;
    if (use_edge_labels)
    {
        edge_labels = g.get_edge_labels();
    }

    Node num_two_tuples = 0;
    for (Node i = 0; i < num_nodes; ++i)
    {
        for (Node j = 0; j < num_nodes; ++j)
        {
            two_tuple_graph.add_node();

            // Map each pair to node in two set graph and also inverse.
            node_to_two_tuple.insert({{num_two_tuples, make_tuple(i, j)}});
            two_tuple_to_node.insert({{make_tuple(i, j), num_two_tuples}});
            num_two_tuples++;

            Label c_i = 1;
            Label c_j = 2;
            if (use_labels)
            {
                c_i = AuxiliaryMethods::pairing(labels[i] + 1, c_i);
                c_j = AuxiliaryMethods::pairing(labels[j] + 1, c_j);
            }

            Label c;
            if (g.has_edge(i, j))
            {
                if (use_edge_labels)
                {
                    auto s = edge_labels.find(make_tuple(i, j));
                    c = AuxiliaryMethods::pairing(3, s->second);
                }
                else
                {
                    c = 3;
                }
            }
            else if (i == j)
            {
                c = 1;
            }
            else
            {
                c = 2;
            }

            Label new_color =
                AuxiliaryMethods::pairing(AuxiliaryMethods::pairing(c_i, c_j), c);
            tuple_labels.push_back(new_color);
        }
    }

    vector<vector<uint>> nonzero_compenents_1;
    vector<vector<uint>> nonzero_compenents_2;

    for (Node i = 0; i < num_two_tuples; ++i)
    {
        // Get nodes of original graph corresponding to two set i.
        TwoTuple p = node_to_two_tuple.find(i)->second;
        Node v = std::get<0>(p);
        Node w = std::get<1>(p);

        // Exchange first node.
        // Iterate over nodes.
        for (Node v_i = 0; v_i < num_nodes; ++v_i)
        {
            unordered_map<TwoTuple, Node>::const_iterator t;
            t = two_tuple_to_node.find(make_tuple(v_i, w));
            two_tuple_graph.add_edge(i, t->second);
            edge_type.insert({{make_tuple(i, t->second), 1}});
            vertex_id.insert({{make_tuple(i, t->second), v_i}});
            local.insert({{make_tuple(i, t->second), 1}});

            nonzero_compenents_1.push_back({{i, t->second}});
        }

        // Exchange second node.
        // Iterate over nodes.
        for (Node v_i = 0; v_i < num_nodes; ++v_i)
        {
            unordered_map<TwoTuple, Node>::const_iterator t;
            t = two_tuple_to_node.find(make_tuple(v, v_i));
            two_tuple_graph.add_edge(i, t->second);
            edge_type.insert({{make_tuple(i, t->second), 2}});
            vertex_id.insert({{make_tuple(i, t->second), v_i}});
            local.insert({{make_tuple(i, t->second), 1}});

            nonzero_compenents_2.push_back({{i, t->second}});
        }
    }

    return std::make_pair(nonzero_compenents_1, nonzero_compenents_2);
}

tuple<vector<vector<uint>>, vector<vector<uint>>, vector<vector<uint>>,
      vector<vector<uint>>>

generate_dwl_sparse_am(const Graph &g, const bool use_labels,
                       const bool use_edge_labels)
{
    size_t num_nodes = g.get_num_nodes();
    // New graph to be generated.
    Graph two_tuple_graph(false);

    // Maps node in two set graph to corresponding two tuple.
    unordered_map<Node, TwoTuple> node_to_two_tuple;
    // Inverse of the above map.
    unordered_map<TwoTuple, Node> two_tuple_to_node;
    // Manages edges labels.
    unordered_map<Edge, uint> edge_type;
    // Manages vertex ids
    unordered_map<Edge, uint> vertex_id;
    unordered_map<Edge, uint> local;

    // Create a node for each two set.
    Labels labels;
    Labels tuple_labels;
    if (use_labels)
    {
        labels = g.get_labels();
    }

    EdgeLabels edge_labels;
    if (use_edge_labels)
    {
        edge_labels = g.get_edge_labels();
    }

    Node num_two_tuples = 0;
    for (Node i = 0; i < num_nodes; ++i)
    {
        for (Node j = 0; j < num_nodes; ++j)
        {
            two_tuple_graph.add_node();

            // Map each pair to node in two set graph and also inverse.
            node_to_two_tuple.insert({{num_two_tuples, make_tuple(i, j)}});
            two_tuple_to_node.insert({{make_tuple(i, j), num_two_tuples}});
            num_two_tuples++;

            Label c_i = 1;
            Label c_j = 2;
            if (use_labels)
            {
                c_i = AuxiliaryMethods::pairing(labels[i] + 1, c_i);
                c_j = AuxiliaryMethods::pairing(labels[j] + 1, c_j);
            }

            Label c;
            if (g.has_edge(i, j))
            {
                if (use_edge_labels)
                {
                    auto s = edge_labels.find(make_tuple(i, j));
                    c = AuxiliaryMethods::pairing(3, s->second);
                }
                else
                {
                    c = 3;
                }
            }
            else if (i == j)
            {
                c = 1;
            }
            else
            {
                c = 2;
            }

            Label new_color =
                AuxiliaryMethods::pairing(AuxiliaryMethods::pairing(c_i, c_j), c);
            tuple_labels.push_back(new_color);
        }
    }

    vector<vector<uint>> nonzero_compenents_1_l;
    vector<vector<uint>> nonzero_compenents_1_g;
    vector<vector<uint>> nonzero_compenents_2_l;
    vector<vector<uint>> nonzero_compenents_2_g;

    for (Node i = 0; i < num_two_tuples; ++i)
    {
        // Get nodes of orginal graph corresponding to two set i.
        TwoTuple p = node_to_two_tuple.find(i)->second;
        Node v = std::get<0>(p);
        Node w = std::get<1>(p);

        // Exchange first node.
        // Iterate over nodes.
        for (Node v_i = 0; v_i < num_nodes; ++v_i)
        {
            unordered_map<TwoTuple, Node>::const_iterator t;
            t = two_tuple_to_node.find(make_tuple(v_i, w));

            // Local vs. global edge.
            if (g.has_edge(v, v_i))
            {
                edge_type.insert({{make_tuple(i, t->second), 1}});
                vertex_id.insert({{make_tuple(i, t->second), v_i}});
                local.insert({{make_tuple(i, t->second), 1}});
                nonzero_compenents_1_l.push_back({{i, t->second}});
            }
            else
            {
                edge_type.insert({{make_tuple(i, t->second), 1}});
                vertex_id.insert({{make_tuple(i, t->second), v_i}});
                local.insert({{make_tuple(i, t->second), 2}});
                nonzero_compenents_1_g.push_back({{i, t->second}});
            }

            two_tuple_graph.add_edge(i, t->second);
        }
        // Exchange second node.
        // Iterate over nodes.
        for (Node v_i = 0; v_i < num_nodes; ++v_i)
        {
            unordered_map<TwoTuple, Node>::const_iterator t;
            t = two_tuple_to_node.find(make_tuple(v, v_i));

            // Local vs. global edge.
            if (g.has_edge(w, v_i))
            {
                edge_type.insert({{make_tuple(i, t->second), 2}});
                vertex_id.insert({{make_tuple(i, t->second), v_i}});
                local.insert({{make_tuple(i, t->second), 1}});
                nonzero_compenents_2_l.push_back({{i, t->second}});
            }
            else
            {
                edge_type.insert({{make_tuple(i, t->second), 2}});
                vertex_id.insert({{make_tuple(i, t->second), v_i}});
                local.insert({{make_tuple(i, t->second), 2}});
                nonzero_compenents_2_g.push_back({{i, t->second}});
            }

            two_tuple_graph.add_edge(i, t->second);
        }
    }
    return std::make_tuple(nonzero_compenents_1_l, nonzero_compenents_1_g,
                           nonzero_compenents_2_l, nonzero_compenents_2_g);
}

tuple<vector<vector<uint>>, vector<vector<uint>>>

generate_dwled_sparse_am(const Graph &g, const bool use_labels,
                         const bool use_edge_labels)
{
    size_t num_nodes = g.get_num_nodes();
    // New graph to be generated.
    Graph two_tuple_graph(false);

    // Maps node in two set graph to corresponding two tuple.
    unordered_map<Node, TwoTuple> node_to_two_tuple;
    // Inverse of the above map.
    unordered_map<TwoTuple, Node> two_tuple_to_node;
    // Manages edges labels.
    unordered_map<Edge, uint> edge_type;
    // Manages vertex ids
    unordered_map<Edge, uint> vertex_id;
    unordered_map<Edge, uint> local;

    // Create a node for each two set.
    Labels labels;
    Labels tuple_labels;
    if (use_labels)
    {
        labels = g.get_labels();
    }

    EdgeLabels edge_labels;
    if (use_edge_labels)
    {
        edge_labels = g.get_edge_labels();
    }

    Node num_two_tuples = 0;
    for (Node i = 0; i < num_nodes; ++i)
    {
        for (Node j = 0; j < num_nodes; ++j)
        {
            two_tuple_graph.add_node();

            // Map each pair to node in two set graph and also inverse.
            node_to_two_tuple.insert({{num_two_tuples, make_tuple(i, j)}});
            two_tuple_to_node.insert({{make_tuple(i, j), num_two_tuples}});
            num_two_tuples++;

            Label c_i = 1;
            Label c_j = 2;
            if (use_labels)
            {
                c_i = AuxiliaryMethods::pairing(labels[i] + 1, c_i);
                c_j = AuxiliaryMethods::pairing(labels[j] + 1, c_j);
            }

            Label c;
            if (g.has_edge(i, j))
            {
                if (use_edge_labels)
                {
                    auto s = edge_labels.find(make_tuple(i, j));
                    c = AuxiliaryMethods::pairing(3, s->second);
                }
                else
                {
                    c = 3;
                }
            }
            else if (i == j)
            {
                c = 1;
            }
            else
            {
                c = 2;
            }

            Label new_color =
                AuxiliaryMethods::pairing(AuxiliaryMethods::pairing(c_i, c_j), c);
            tuple_labels.push_back(new_color);
        }
    }

    vector<vector<uint>> nonzero_compenents_1;
    vector<vector<uint>> nonzero_compenents_2;

    for (Node i = 0; i < num_two_tuples; ++i)
    {
        // Get nodes of orginal graph corresponding to two set i.
        TwoTuple p = node_to_two_tuple.find(i)->second;
        Node v = std::get<0>(p);
        Node w = std::get<1>(p);

        // Exchange first node.
        // Iterate over nodes.
        for (Node v_i = 0; v_i < num_nodes; ++v_i)
        {
            unordered_map<TwoTuple, Node>::const_iterator t;
            t = two_tuple_to_node.find(make_tuple(v_i, w));

            // Local vs. global edge.
            if (g.has_edge(v, v_i))
            {
                edge_type.insert({{make_tuple(i, t->second), 1}});
                vertex_id.insert({{make_tuple(i, t->second), v_i}});
                local.insert({{make_tuple(i, t->second), 1}});
                nonzero_compenents_1.push_back({{i, t->second}});
            }
            else
            {
                edge_type.insert({{make_tuple(i, t->second), 1}});
                vertex_id.insert({{make_tuple(i, t->second), v_i}});
                local.insert({{make_tuple(i, t->second), 2}});
                nonzero_compenents_1.push_back({{i, t->second}});
            }

            two_tuple_graph.add_edge(i, t->second);
        }
        // Exchange second node.
        // Iterate over nodes.
        for (Node v_i = 0; v_i < num_nodes; ++v_i)
        {
            unordered_map<TwoTuple, Node>::const_iterator t;
            t = two_tuple_to_node.find(make_tuple(v, v_i));

            // Local vs. global edge.
            if (g.has_edge(w, v_i))
            {
                edge_type.insert({{make_tuple(i, t->second), 2}});
                vertex_id.insert({{make_tuple(i, t->second), v_i}});
                local.insert({{make_tuple(i, t->second), 1}});
                nonzero_compenents_2.push_back({{i, t->second}});
            }
            else
            {
                edge_type.insert({{make_tuple(i, t->second), 2}});
                vertex_id.insert({{make_tuple(i, t->second), v_i}});
                local.insert({{make_tuple(i, t->second), 2}});
                nonzero_compenents_2.push_back({{i, t->second}});
            }

            two_tuple_graph.add_edge(i, t->second);
        }
    }
    return std::make_tuple(nonzero_compenents_1, nonzero_compenents_2);
}

tuple<vector<vector<uint>>, vector<vector<uint>>, vector<vector<uint>>>

generate_local_sparse_am_3(const Graph &g, const bool use_labels,
                           const bool use_edge_labels)
{
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
    if (use_labels)
    {
        labels = g.get_labels();
    }

    size_t num_three_tuples = 0;
    for (Node i = 0; i < num_nodes; ++i)
    {
        for (Node j = 0; j < num_nodes; ++j)
        {
            for (Node k = 0; k < num_nodes; ++k)
            {
                three_tuple_graph.add_node();

                node_to_three_tuple.insert({{num_three_tuples, make_tuple(i, j, k)}});
                three_tuple_to_node.insert({{make_tuple(i, j, k), num_three_tuples}});
                num_three_tuples++;

                Label c_i = 1;
                Label c_j = 2;
                Label c_k = 3;

                if (use_labels)
                {
                    c_i = AuxiliaryMethods::pairing(labels[i] + 1, c_i);
                    c_j = AuxiliaryMethods::pairing(labels[j] + 1, c_j);
                    c_k = AuxiliaryMethods::pairing(labels[k] + 1, c_k);
                }

                Label a, b, c;
                if (g.has_edge(i, j))
                {
                    a = 1;
                }
                else if (not g.has_edge(i, j))
                {
                    a = 2;
                }
                else
                {
                    a = 3;
                }

                if (g.has_edge(i, k))
                {
                    b = 1;
                }
                else if (not g.has_edge(i, k))
                {
                    b = 2;
                }
                else
                {
                    b = 3;
                }

                if (g.has_edge(j, k))
                {
                    c = 1;
                }
                else if (not g.has_edge(j, k))
                {
                    c = 2;
                }
                else
                {
                    c = 3;
                }

                Label new_color_0 =
                    AuxiliaryMethods::pairing(AuxiliaryMethods::pairing(a, b), c);
                Label new_color_1 =
                    AuxiliaryMethods::pairing(AuxiliaryMethods::pairing(c_i, c_j), c_k);
                Label new_color = AuxiliaryMethods::pairing(new_color_0, new_color_1);
                tuple_labels.push_back(new_color);
            }
        }
    }

    vector<vector<uint>> nonzero_compenents_1;
    vector<vector<uint>> nonzero_compenents_2;
    vector<vector<uint>> nonzero_compenents_3;

    for (Node i = 0; i < num_three_tuples; ++i)
    {
        // Get nodes of original graph corresponding to two tuple i.
        ThreeTuple p = node_to_three_tuple.find(i)->second;
        Node v = std::get<0>(p);
        Node w = std::get<1>(p);
        Node u = std::get<2>(p);

        // Exchange first node.
        Nodes v_neighbors = g.get_neighbours(v);
        for (const auto &v_n : v_neighbors)
        {
            unordered_map<ThreeTuple, Node>::const_iterator t;
            t = three_tuple_to_node.find(make_tuple(v_n, w, u));

            three_tuple_graph.add_edge(i, t->second);
            edge_type.insert({{make_tuple(i, t->second), 1}});
            vertex_id.insert({{make_tuple(i, t->second), v_n}});
            local.insert({{make_tuple(i, t->second), 1}});

            nonzero_compenents_1.push_back({{i, t->second}});
        }

        // Exchange second node.
        Nodes w_neighbors = g.get_neighbours(w);
        for (const auto &w_n : w_neighbors)
        {
            unordered_map<ThreeTuple, Node>::const_iterator t;
            t = three_tuple_to_node.find(make_tuple(v, w_n, u));

            three_tuple_graph.add_edge(i, t->second);
            edge_type.insert({{make_tuple(i, t->second), 2}});
            vertex_id.insert({{make_tuple(i, t->second), w_n}});
            local.insert({{make_tuple(i, t->second), 1}});

            nonzero_compenents_2.push_back({{i, t->second}});
        }

        // Exchange third node.
        Nodes u_neighbors = g.get_neighbours(u);
        for (const auto &u_n : u_neighbors)
        {
            unordered_map<ThreeTuple, Node>::const_iterator t;
            t = three_tuple_to_node.find(make_tuple(v, w, u_n));

            three_tuple_graph.add_edge(i, t->second);
            edge_type.insert({{make_tuple(i, t->second), 3}});
            vertex_id.insert({{make_tuple(i, t->second), u_n}});
            local.insert({{make_tuple(i, t->second), 1}});

            nonzero_compenents_3.push_back({{i, t->second}});
        }
    }

    return std::make_tuple(nonzero_compenents_1, nonzero_compenents_2,
                           nonzero_compenents_3);
}

tuple<vector<vector<uint>>, vector<vector<uint>>, vector<vector<uint>>>

generate_local_sparse_am_3_connected(const Graph &g, const bool use_labels,
                                     const bool use_edge_labels)
{
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
    if (use_labels)
    {
        labels = g.get_labels();
    }

    size_t num_three_tuples = 0;
    for (Node i = 0; i < num_nodes; ++i)
    {
        for (Node j = 0; j < num_nodes; ++j)
        {
            for (Node k = 0; k < num_nodes; ++k)
            {
                if ((g.has_edge(i, j) and g.has_edge(i, k)) or
                    (g.has_edge(j, i) and g.has_edge(j, k)) or
                    (g.has_edge(k, j) and g.has_edge(k, i)) or
                    ((i == j) and (j == k)) or ((i == k) and (g.has_edge(i, j))) or
                    ((i == j) and (g.has_edge(i, k))) or
                    ((j == k) and (g.has_edge(i, j))) or
                    ((j == i) and (g.has_edge(j, k))) or
                    ((k == i) and (g.has_edge(k, j))) or
                    ((k == j) and (g.has_edge(k, i))))
                {
                    three_tuple_graph.add_node();

                    node_to_three_tuple.insert({{num_three_tuples, make_tuple(i, j, k)}});
                    three_tuple_to_node.insert({{make_tuple(i, j, k), num_three_tuples}});
                    num_three_tuples++;

                    Label c_i = 1;
                    Label c_j = 2;
                    Label c_k = 3;

                    if (use_labels)
                    {
                        c_i = AuxiliaryMethods::pairing(labels[i] + 1, c_i);
                        c_j = AuxiliaryMethods::pairing(labels[j] + 1, c_j);
                        c_k = AuxiliaryMethods::pairing(labels[k] + 1, c_k);
                    }

                    Label a, b, c;
                    if (g.has_edge(i, j))
                    {
                        a = 1;
                    }
                    else if (not g.has_edge(i, j))
                    {
                        a = 2;
                    }
                    else
                    {
                        a = 3;
                    }

                    if (g.has_edge(i, k))
                    {
                        b = 1;
                    }
                    else if (not g.has_edge(i, k))
                    {
                        b = 2;
                    }
                    else
                    {
                        b = 3;
                    }

                    if (g.has_edge(j, k))
                    {
                        c = 1;
                    }
                    else if (not g.has_edge(j, k))
                    {
                        c = 2;
                    }
                    else
                    {
                        c = 3;
                    }

                    Label new_color_0 =
                        AuxiliaryMethods::pairing(AuxiliaryMethods::pairing(a, b), c);
                    Label new_color_1 = AuxiliaryMethods::pairing(
                        AuxiliaryMethods::pairing(c_i, c_j), c_k);
                    Label new_color = AuxiliaryMethods::pairing(new_color_0, new_color_1);
                    tuple_labels.push_back(new_color);
                }
            }
        }
    }

    vector<vector<uint>> nonzero_compenents_1;
    vector<vector<uint>> nonzero_compenents_2;
    vector<vector<uint>> nonzero_compenents_3;

    for (Node i = 0; i < num_three_tuples; ++i)
    {
        // Get nodes of original graph corresponding to two tuple i.
        ThreeTuple p = node_to_three_tuple.find(i)->second;
        Node v = std::get<0>(p);
        Node w = std::get<1>(p);
        Node u = std::get<2>(p);

        // Exchange first node.
        Nodes v_neighbors = g.get_neighbours(v);
        for (const auto &v_n : v_neighbors)
        {
            unordered_map<ThreeTuple, Node>::const_iterator t;
            t = three_tuple_to_node.find(make_tuple(v_n, w, u));

            if (t != three_tuple_to_node.end())
            {
                three_tuple_graph.add_edge(i, t->second);
                edge_type.insert({{make_tuple(i, t->second), 1}});
                vertex_id.insert({{make_tuple(i, t->second), v_n}});
                local.insert({{make_tuple(i, t->second), 1}});

                nonzero_compenents_1.push_back({{i, t->second}});
            }
        }

        // Exchange second node.
        Nodes w_neighbors = g.get_neighbours(w);
        for (const auto &w_n : w_neighbors)
        {
            unordered_map<ThreeTuple, Node>::const_iterator t;
            t = three_tuple_to_node.find(make_tuple(v, w_n, u));

            if (t != three_tuple_to_node.end())
            {
                three_tuple_graph.add_edge(i, t->second);
                edge_type.insert({{make_tuple(i, t->second), 2}});
                vertex_id.insert({{make_tuple(i, t->second), w_n}});
                local.insert({{make_tuple(i, t->second), 1}});

                nonzero_compenents_2.push_back({{i, t->second}});
            }
        }

        // Exchange third node.
        Nodes u_neighbors = g.get_neighbours(u);
        for (const auto &u_n : u_neighbors)
        {
            unordered_map<ThreeTuple, Node>::const_iterator t;
            t = three_tuple_to_node.find(make_tuple(v, w, u_n));

            if (t != three_tuple_to_node.end())
            {
                three_tuple_graph.add_edge(i, t->second);
                edge_type.insert({{make_tuple(i, t->second), 3}});
                vertex_id.insert({{make_tuple(i, t->second), u_n}});
                local.insert({{make_tuple(i, t->second), 1}});

                nonzero_compenents_3.push_back({{i, t->second}});
            }
        }
    }

    return std::make_tuple(nonzero_compenents_1, nonzero_compenents_2,
                           nonzero_compenents_3);
}

pair<vector<int>, vector<int>> get_edge_labels(const Graph &g,
                                               const bool use_labels,
                                               const bool use_edge_labels)
{
    size_t num_nodes = g.get_num_nodes();
    // New graph to be generated.
    Graph two_tuple_graph(false);

    // Maps node in two set graph to corresponding two tuple.
    unordered_map<Node, TwoTuple> node_to_two_tuple;
    // Inverse of the above map.
    unordered_map<TwoTuple, Node> two_tuple_to_node;
    // Manages edges labels.
    unordered_map<Edge, uint> edge_type;
    // Manages vertex ids
    unordered_map<Edge, uint> vertex_id;
    unordered_map<Edge, uint> local;

    // Create a node for each two set.
    Labels labels;
    Labels tuple_labels;
    if (use_labels)
    {
        labels = g.get_labels();
    }

    EdgeLabels edge_labels;
    if (use_edge_labels)
    {
        edge_labels = g.get_edge_labels();
    }

    Node num_two_tuples = 0;
    for (Node i = 0; i < num_nodes; ++i)
    {
        for (Node j = 0; j < num_nodes; ++j)
        {
            two_tuple_graph.add_node();

            // Map each pair to node in two set graph and also inverse.
            node_to_two_tuple.insert({{num_two_tuples, make_tuple(i, j)}});
            two_tuple_to_node.insert({{make_tuple(i, j), num_two_tuples}});
            num_two_tuples++;

            Label c_i = 1;
            Label c_j = 2;
            if (use_labels)
            {
                c_i = AuxiliaryMethods::pairing(labels[i] + 1, c_i);
                c_j = AuxiliaryMethods::pairing(labels[j] + 1, c_j);
            }

            Label c;
            if (g.has_edge(i, j))
            {
                if (use_edge_labels)
                {
                    auto s = edge_labels.find(make_tuple(i, j));
                    c = AuxiliaryMethods::pairing(3, s->second);
                }
                else
                {
                    c = 3;
                }
            }
            else if (i == j)
            {
                c = 1;
            }
            else
            {
                c = 2;
            }

            Label new_color =
                AuxiliaryMethods::pairing(AuxiliaryMethods::pairing(c_i, c_j), c);
            tuple_labels.push_back(new_color);
        }
    }

    vector<int> edge_labelsn_1;
    vector<int> edge_labelsn_2;
    for (Node i = 0; i < num_two_tuples; ++i)
    {
        // Get nodes of orginal graph corresponding to two set i.
        TwoTuple p = node_to_two_tuple.find(i)->second;
        Node v = std::get<0>(p);
        Node w = std::get<1>(p);

        // Exchange first node.
        // Iterate over nodes.
        for (Node v_i = 0; v_i < num_nodes; ++v_i)
        {
            unordered_map<TwoTuple, Node>::const_iterator t;
            t = two_tuple_to_node.find(make_tuple(v_i, w));

            // Local vs. global edge.
            if (g.has_edge(v, v_i))
            {
                edge_labelsn_1.push_back(0);
            }
            else
            {
                edge_labelsn_1.push_back(1);
            }

            two_tuple_graph.add_edge(i, t->second);
        }
        // Exchange second node.
        // Iterate over nodes.
        for (Node v_i = 0; v_i < num_nodes; ++v_i)
        {
            unordered_map<TwoTuple, Node>::const_iterator t;
            t = two_tuple_to_node.find(make_tuple(v, v_i));

            // Local vs. global edge.
            if (g.has_edge(w, v_i))
            {
                edge_labelsn_2.push_back(0);
            }
            else
            {
                edge_labelsn_2.push_back(1);
            }

            two_tuple_graph.add_edge(i, t->second);
        }
    }

    return std::make_pair(edge_labelsn_1, edge_labelsn_2);
}

vector<unsigned long> get_node_labels(const Graph &g, const bool use_labels,
                                      const bool use_edge_labels)
{
    size_t num_nodes = g.get_num_nodes();

    // Create a node for each two set.
    Labels labels;
    vector<unsigned long> tuple_labels;
    if (use_labels)
    {
        labels = g.get_labels();
    }

    EdgeLabels edge_labels;
    if (use_edge_labels)
    {
        edge_labels = g.get_edge_labels();
    }

    for (Node i = 0; i < num_nodes; ++i)
    {
        for (Node j = 0; j < num_nodes; ++j)
        {
            Label c_i = 1;
            Label c_j = 2;
            if (use_labels)
            {
                c_i = AuxiliaryMethods::pairing(labels[i] + 1, c_i);
                c_j = AuxiliaryMethods::pairing(labels[j] + 1, c_j);
            }

            Label c;
            if (g.has_edge(i, j))
            {
                if (use_edge_labels)
                {
                    auto s = edge_labels.find(make_tuple(i, j));
                    c = AuxiliaryMethods::pairing(3, s->second);
                }
                else
                {
                    c = 3;
                }
            }
            else if (i == j)
            {
                c = 1;
            }
            else
            {
                c = 2;
            }

            Label new_color =
                AuxiliaryMethods::pairing(AuxiliaryMethods::pairing(c_i, c_j), c);
            tuple_labels.push_back(new_color);
        }
    }

    return tuple_labels;
}

vector<unsigned long> get_node_labels_con(const Graph &g, const bool use_labels,
                                          const bool use_edge_labels)
{
    size_t num_nodes = g.get_num_nodes();

    // Create a node for each two set.
    Labels labels;
    vector<unsigned long> tuple_labels;
    if (use_labels)
    {
        labels = g.get_labels();
    }

    EdgeLabels edge_labels;
    if (use_edge_labels)
    {
        edge_labels = g.get_edge_labels();
    }

    for (Node i = 0; i < num_nodes; ++i)
    {
        for (Node j = 0; j < num_nodes; ++j)
        {
            if (g.has_edge(i, j) or g.has_edge(j, i) or (j == i))
            {
                Label c_i = 1;
                Label c_j = 2;
                if (use_labels)
                {
                    c_i = AuxiliaryMethods::pairing(labels[i] + 1, c_i);
                    c_j = AuxiliaryMethods::pairing(labels[j] + 1, c_j);
                }

                Label c;
                if (g.has_edge(i, j))
                {
                    if (use_edge_labels)
                    {
                        auto s = edge_labels.find(make_tuple(i, j));
                        c = AuxiliaryMethods::pairing(3, s->second);
                    }
                    else
                    {
                        c = 3;
                    }
                }
                else if (i == j)
                {
                    c = 1;
                }
                else
                {
                    c = 2;
                }

                Label new_color =
                    AuxiliaryMethods::pairing(AuxiliaryMethods::pairing(c_i, c_j), c);
                tuple_labels.push_back(new_color);
            }
        }
    }

    return tuple_labels;
}

vector<unsigned long> get_node_labels_unc(const Graph &g, const bool use_labels,
                                          const bool use_edge_labels)
{
    size_t num_nodes = g.get_num_nodes();

    // Create a node for each two set.
    Labels labels;
    vector<unsigned long> tuple_labels;
    if (use_labels)
    {
        labels = g.get_labels();
    }

    EdgeLabels edge_labels;
    if (use_edge_labels)
    {
        edge_labels = g.get_edge_labels();
    }

    for (Node i = 0; i < num_nodes; ++i)
    {
        for (Node j = 0; j < num_nodes; ++j)
        {
            if (not g.has_edge(i, j))
            {
                Label c_i = 1;
                Label c_j = 2;
                if (use_labels)
                {
                    c_i = AuxiliaryMethods::pairing(labels[i] + 1, c_i);
                    c_j = AuxiliaryMethods::pairing(labels[j] + 1, c_j);
                }

                Label c;
                if (g.has_edge(i, j))
                {
                    if (use_edge_labels)
                    {
                        auto s = edge_labels.find(make_tuple(i, j));
                        c = AuxiliaryMethods::pairing(3, s->second);
                    }
                    else
                    {
                        c = 3;
                    }
                }
                else if (i == j)
                {
                    c = 1;
                }
                else
                {
                    c = 2;
                }

                Label new_color =
                    AuxiliaryMethods::pairing(AuxiliaryMethods::pairing(c_i, c_j), c);
                tuple_labels.push_back(new_color);
            }
        }
    }

    return tuple_labels;
}

vector<unsigned long> get_node_labels_connected(const Graph &g,
                                                const bool use_labels,
                                                const bool use_edge_labels)
{
    size_t num_nodes = g.get_num_nodes();

    // Create a node for each two set.
    Labels labels;
    vector<unsigned long> tuple_labels;
    if (use_labels)
    {
        labels = g.get_labels();
    }

    EdgeLabels edge_labels;
    if (use_edge_labels)
    {
        edge_labels = g.get_edge_labels();
    }

    for (Node i = 0; i < num_nodes; ++i)
    {
        for (Node j = 0; j < num_nodes; ++j)
        {
            if (g.has_edge(i, j) or g.has_edge(i, j) or (i == j))
            {
                Label c_i = 1;
                Label c_j = 2;
                if (use_labels)
                {
                    c_i = AuxiliaryMethods::pairing(labels[i] + 1, c_i);
                    c_j = AuxiliaryMethods::pairing(labels[j] + 1, c_j);
                }

                Label c;
                if (g.has_edge(i, j))
                {
                    if (use_edge_labels)
                    {
                        auto s = edge_labels.find(make_tuple(i, j));
                        c = AuxiliaryMethods::pairing(3, s->second);
                    }
                    else
                    {
                        c = 3;
                    }
                }
                else if (i == j)
                {
                    c = 1;
                }
                else
                {
                    c = 2;
                }

                Label new_color =
                    AuxiliaryMethods::pairing(AuxiliaryMethods::pairing(c_i, c_j), c);
                tuple_labels.push_back(new_color);
            }
        }
    }

    return tuple_labels;
}

vector<unsigned long> get_node_labels_1(const Graph &g, const bool use_labels)
{
    size_t num_nodes = g.get_num_nodes();

    // Create a node for each two set.
    Labels labels;
    vector<unsigned long> tuple_labels;
    if (use_labels)
    {
        tuple_labels = g.get_labels();
    }

    return tuple_labels;
}

vector<int> get_edge_labels_1(const Graph &g)
{
    size_t num_nodes = g.get_num_nodes();
    EdgeLabels edge_labels = g.get_edge_labels();

    vector<int> labels;

    for (Node v = 0; v < num_nodes; ++v)
    {
        // Exchange first node.
        Nodes v_neighbors = g.get_neighbours(v);
        for (Node v_n : v_neighbors)
        {
            labels.push_back(edge_labels.find(std::make_tuple(v, v_n))->second);
        }
    }

    return labels;
}

vector<unsigned long> get_node_labels_3(const Graph &g, const bool use_labels,
                                        const bool use_edge_labels)
{
    size_t num_nodes = g.get_num_nodes();

    // Create a node for each two set.
    Labels labels;
    vector<unsigned long> tuple_labels;
    if (use_labels)
    {
        labels = g.get_labels();
    }

    size_t num_three_tuples = 0;
    for (Node i = 0; i < num_nodes; ++i)
    {
        for (Node j = 0; j < num_nodes; ++j)
        {
            for (Node k = 0; k < num_nodes; ++k)
            {
                if ((g.has_edge(i, j) and g.has_edge(i, k)) or
                    (g.has_edge(j, i) and g.has_edge(j, k)) or
                    (g.has_edge(k, j) and g.has_edge(k, i)) or
                    ((i == j) and (j == k)) or ((i == k) and (g.has_edge(i, j))) or
                    ((i == j) and (g.has_edge(i, k))) or
                    ((j == k) and (g.has_edge(i, j))) or
                    ((j == i) and (g.has_edge(j, k))) or
                    ((k == i) and (g.has_edge(k, j))) or
                    ((k == j) and (g.has_edge(k, i))))
                {
                    Label c_i = 1;
                    Label c_j = 2;
                    Label c_k = 3;

                    if (use_labels)
                    {
                        c_i = AuxiliaryMethods::pairing(labels[i] + 1, c_i);
                        c_j = AuxiliaryMethods::pairing(labels[j] + 1, c_j);
                        c_k = AuxiliaryMethods::pairing(labels[k] + 1, c_k);
                    }

                    Label a, b, c;
                    if (g.has_edge(i, j))
                    {
                        a = 1;
                    }
                    else if (not g.has_edge(i, j))
                    {
                        a = 2;
                    }
                    else
                    {
                        a = 3;
                    }

                    if (g.has_edge(i, k))
                    {
                        b = 1;
                    }
                    else if (not g.has_edge(i, k))
                    {
                        b = 2;
                    }
                    else
                    {
                        b = 3;
                    }

                    if (g.has_edge(j, k))
                    {
                        c = 1;
                    }
                    else if (not g.has_edge(j, k))
                    {
                        c = 2;
                    }
                    else
                    {
                        c = 3;
                    }

                    Label new_color_0 =
                        AuxiliaryMethods::pairing(AuxiliaryMethods::pairing(a, b), c);
                    Label new_color_1 = AuxiliaryMethods::pairing(
                        AuxiliaryMethods::pairing(c_i, c_j), c_k);
                    Label new_color = AuxiliaryMethods::pairing(new_color_0, new_color_1);
                    tuple_labels.push_back(new_color);
                }
            }
        }
    }
    return tuple_labels;
}

vector<pair<vector<vector<uint>>, vector<vector<uint>>>> get_all_matrices_brec(
    string name)
{
    GraphDatabase gdb = AuxiliaryMethods::read_graph_txt_file(name);
    gdb.erase(gdb.

              begin()

              + 0);

    GraphDatabase gdb_new;
    for (auto data : gdb)
    {
        gdb_new.push_back(data);
    }

    vector<pair<vector<vector<uint>>, vector<vector<uint>>>> matrices;

    for (auto &g : gdb_new)
    {
        matrices.push_back(generate_local_sparse_am(g, false, false));
    }

    return matrices;
}

vector<pair<vector<vector<uint>>, vector<vector<uint>>>> get_all_matrices(
    string name, const std::vector<int> &indices)
{
    GraphDatabase gdb = AuxiliaryMethods::read_graph_txt_file(name);
    gdb.erase(gdb.

              begin()

              + 0);

    GraphDatabase gdb_new;
    for (auto i : indices)
    {
        gdb_new.push_back(gdb[int(i)]);
    }

    vector<pair<vector<vector<uint>>, vector<vector<uint>>>> matrices;

    for (auto &g : gdb_new)
    {
        matrices.push_back(generate_local_sparse_am(g, false, false));
    }

    return matrices;
}

vector<pair<vector<vector<uint>>, vector<vector<uint>>>>
get_all_matrices_connected(string name, const std::vector<int> &indices)
{
    GraphDatabase gdb = AuxiliaryMethods::read_graph_txt_file(name);
    gdb.erase(gdb.

              begin()

              + 0);

    GraphDatabase gdb_new;
    for (auto i : indices)
    {
        gdb_new.push_back(gdb[int(i)]);
    }

    vector<pair<vector<vector<uint>>, vector<vector<uint>>>> matrices;

    for (auto &g : gdb_new)
    {
        matrices.push_back(generate_local_sparse_am_connected(g, false, false));
    }

    return matrices;
}

vector<pair<vector<vector<uint>>, vector<vector<uint>>>> get_all_matrices_con(
    string name, const std::vector<int> &indices)
{
    GraphDatabase gdb = AuxiliaryMethods::read_graph_txt_file(name);
    gdb.erase(gdb.

              begin()

              + 0);

    GraphDatabase gdb_new;
    for (auto i : indices)
    {
        gdb_new.push_back(gdb[int(i)]);
    }

    vector<pair<vector<vector<uint>>, vector<vector<uint>>>> matrices;

    for (auto &g : gdb_new)
    {
        matrices.push_back(generate_local_sparse_am_con(g, false, false));
    }

    return matrices;
}

vector<pair<vector<vector<uint>>, vector<vector<uint>>>> get_all_matrices_unc(
    string name, const std::vector<int> &indices)
{
    GraphDatabase gdb = AuxiliaryMethods::read_graph_txt_file(name);
    gdb.erase(gdb.

              begin()

              + 0);

    GraphDatabase gdb_new;
    for (auto i : indices)
    {
        gdb_new.push_back(gdb[int(i)]);
    }

    vector<pair<vector<vector<uint>>, vector<vector<uint>>>> matrices;

    for (auto &g : gdb_new)
    {
        matrices.push_back(generate_local_sparse_am_unc(g, false, false));
    }

    return matrices;
}

vector<vector<vector<uint>>> get_all_matrices_1(
    string name, const std::vector<int> &indices)
{
    GraphDatabase gdb = AuxiliaryMethods::read_graph_txt_file(name);
    gdb.erase(gdb.

              begin()

              + 0);

    GraphDatabase gdb_new;
    for (auto i : indices)
    {
        gdb_new.push_back(gdb[int(i)]);
    }

    vector<vector<vector<uint>>> matrices;

    for (auto &g : gdb_new)
    {
        matrices.push_back(generate_local_sparse_am_1(g));
    }

    return matrices;
}

vector<pair<vector<vector<uint>>, vector<vector<uint>>>> get_all_matrices_wl(
    string name, const std::vector<int> &indices)
{
    GraphDatabase gdb = AuxiliaryMethods::read_graph_txt_file(name);
    gdb.erase(gdb.

              begin()

              + 0);
    cout << "@@@" << endl;

    vector<pair<vector<vector<uint>>, vector<vector<uint>>>> matrices;

    for (auto i : indices)
    {
        matrices.push_back(generate_wl_sparse_am(gdb[int(i)], false, false));
    }

    return matrices;
}

vector<tuple<vector<vector<uint>>, vector<vector<uint>>>> get_all_matrices_dwle(
    string name, const std::vector<int> &indices)
{
    GraphDatabase gdb = AuxiliaryMethods::read_graph_txt_file(name);
    gdb.erase(gdb.

              begin()

              + 0);

    GraphDatabase gdb_new;
    for (auto i : indices)
    {
        gdb_new.push_back(gdb[int(i)]);
    }

    vector<tuple<vector<vector<uint>>, vector<vector<uint>>>> matrices;

    for (auto &g : gdb_new)
    {
        matrices.push_back(generate_dwled_sparse_am(g, true, false));
    }

    return matrices;
}

vector<tuple<vector<vector<uint>>, vector<vector<uint>>, vector<vector<uint>>,
             vector<vector<uint>>>>
get_all_matrices_dwl(string name, const std::vector<int> &indices)
{
    GraphDatabase gdb = AuxiliaryMethods::read_graph_txt_file(name);
    gdb.erase(gdb.

              begin()

              + 0);

    GraphDatabase gdb_new;
    for (auto i : indices)
    {
        gdb_new.push_back(gdb[int(i)]);
    }

    vector<tuple<vector<vector<uint>>, vector<vector<uint>>, vector<vector<uint>>,
                 vector<vector<uint>>>>
        matrices;

    for (auto &g : gdb_new)
    {
        matrices.push_back(generate_dwl_sparse_am(g, true, false));
    }

    return matrices;
}

vector<tuple<vector<vector<uint>>, vector<vector<uint>>, vector<vector<uint>>>>
get_all_matrices_3(string name, const std::vector<int> &indices)
{
    GraphDatabase gdb = AuxiliaryMethods::read_graph_txt_file(name);
    gdb.erase(gdb.

              begin()

              + 0);

    GraphDatabase gdb_new;
    for (auto i : indices)
    {
        gdb_new.push_back(gdb[int(i)]);
    }

    vector<
        tuple<vector<vector<uint>>, vector<vector<uint>>, vector<vector<uint>>>>
        matrices;

    uint i = 0;
    for (auto &g : gdb_new)
    {
        matrices.push_back(generate_local_sparse_am_3(g, true, false));
        i++;
    }

    return matrices;
}

vector<tuple<vector<vector<uint>>, vector<vector<uint>>, vector<vector<uint>>>>
get_all_matrices_3_connected(string name, const std::vector<int> &indices)
{
    GraphDatabase gdb = AuxiliaryMethods::read_graph_txt_file(name);
    gdb.erase(gdb.

              begin()

              + 0);

    GraphDatabase gdb_new;
    for (auto i : indices)
    {
        gdb_new.push_back(gdb[int(i)]);
    }

    vector<
        tuple<vector<vector<uint>>, vector<vector<uint>>, vector<vector<uint>>>>
        matrices;

    uint i = 0;
    for (auto &g : gdb_new)
    {
        matrices.push_back(generate_local_sparse_am_3_connected(g, true, false));
        i++;
    }

    return matrices;
}

vector<vector<int>> get_all_edge_labels_1(string name)
{
    GraphDatabase gdb = AuxiliaryMethods::read_graph_txt_file(name);
    gdb.erase(gdb.

              begin()

              + 0);

    vector<vector<int>> edge_labels;

    uint m_num_labels = 0;
    unordered_map<int, int> m_label_to_index;

    for (auto &g : gdb)
    {
        vector<int> colors = get_edge_labels_1(g);
        vector<int> new_color;

        for (auto &c : colors)
        {
            const auto it(m_label_to_index.find(c));
            if (it == m_label_to_index.

                      end()

            )
            {
                m_label_to_index.insert({{c, m_num_labels}});
                new_color.push_back(m_num_labels);

                m_num_labels++;
            }
            else
            {
                new_color.push_back(it->second);
            }
        }

        edge_labels.push_back(new_color);
    }

    cout << m_num_labels << endl;

    return edge_labels;
}

vector<vector<unsigned long>> get_all_node_labels(string name,
                                                  const bool use_node_labels,
                                                  const bool use_edge_labels)
{
    GraphDatabase gdb = AuxiliaryMethods::read_graph_txt_file(name);
    gdb.erase(gdb.

              begin()

              + 0);

    vector<vector<unsigned long>> node_labels;

    uint m_num_labels = 0;
    unordered_map<int, int> m_label_to_index;

    for (auto &g : gdb)
    {
        vector<unsigned long> colors =
            get_node_labels(g, use_node_labels, use_edge_labels);
        vector<unsigned long> new_color;

        for (auto &c : colors)
        {
            const auto it(m_label_to_index.find(c));
            if (it == m_label_to_index.

                      end()

            )
            {
                m_label_to_index.insert({{c, m_num_labels}});
                new_color.push_back(m_num_labels);

                m_num_labels++;
            }
            else
            {
                new_color.push_back(it->second);
            }
        }

        node_labels.push_back(new_color);
    }

    cout << m_num_labels << endl;
    return node_labels;
}

vector<tuple<Attributes, Attributes, Attributes>> get_all_attributes(
    string name)
{
    GraphDatabase gdb = AuxiliaryMethods::read_graph_txt_file(name);
    gdb.erase(gdb.

              begin()

              + 0);

    vector<tuple<Attributes, Attributes, Attributes>> attributes;

    uint i = 1;
    for (auto &g : gdb)
    {
        attributes.push_back(get_attributes(g));
        cout << i << endl;
        i++;
    }

    return attributes;
}

vector<vector<unsigned long>> get_all_node_labels_1(
    string name, const bool use_node_labels)
{
    GraphDatabase gdb = AuxiliaryMethods::read_graph_txt_file(name);
    gdb.erase(gdb.

              begin()

              + 0);

    vector<vector<unsigned long>> node_labels;

    uint m_num_labels = 0;
    unordered_map<int, int> m_label_to_index;

    for (auto &g : gdb)
    {
        vector<unsigned long> colors = get_node_labels_1(g, use_node_labels);
        vector<unsigned long> new_color;

        for (auto &c : colors)
        {
            const auto it(m_label_to_index.find(c));
            if (it == m_label_to_index.

                      end()

            )
            {
                m_label_to_index.insert({{c, m_num_labels}});
                new_color.push_back(m_num_labels);

                m_num_labels++;
            }
            else
            {
                new_color.push_back(it->second);
            }
        }

        node_labels.push_back(new_color);
    }

    cout << m_num_labels << endl;
    return node_labels;
}

vector<vector<unsigned long>> get_all_node_labels_3(
    string name, const bool use_node_labels, const bool use_edge_labels)
{
    GraphDatabase gdb = AuxiliaryMethods::read_graph_txt_file(name);
    gdb.erase(gdb.

              begin()

              + 0);

    vector<vector<unsigned long>> node_labels;

    uint m_num_labels = 0;
    unordered_map<unsigned long, int> m_label_to_index;

    for (auto &g : gdb)
    {
        vector<unsigned long> colors =
            get_node_labels_3(g, use_node_labels, use_edge_labels);
        vector<unsigned long> new_color;

        for (auto &c : colors)
        {
            const auto it(m_label_to_index.find(c));
            if (it == m_label_to_index.

                      end()

            )
            {
                m_label_to_index.insert({{c, m_num_labels}});
                new_color.push_back(m_num_labels);

                m_num_labels++;
            }
            else
            {
                new_color.push_back(it->second);
            }
        }

        node_labels.push_back(new_color);
    }

    cout << m_num_labels << endl;
    return node_labels;
}

vector<vector<unsigned long>> get_all_node_labels_ZINC(
    const bool use_node_labels, const bool use_edge_labels,
    const std::vector<int> &indices_train, const std::vector<int> &indices_val,
    const std::vector<int> &indices_test)
{
    GraphDatabase gdb_1 = AuxiliaryMethods::read_graph_txt_file("ZINC_train");
    gdb_1.erase(gdb_1.begin() + 0);

    GraphDatabase gdb_new_1;
    for (auto i : indices_train)
    {
        gdb_new_1.push_back(gdb_1[int(i)]);
    }
    cout << gdb_new_1.size() << endl;
    cout << "$$$" << endl;

    GraphDatabase gdb_2 = AuxiliaryMethods::read_graph_txt_file("ZINC_val");
    gdb_2.erase(gdb_2.begin() + 0);

    GraphDatabase gdb_new_2;
    for (auto i : indices_val)
    {
        gdb_new_2.push_back(gdb_2[int(i)]);
    }
    cout << gdb_new_2.size() << endl;
    cout << "$$$" << endl;

    GraphDatabase gdb_3 = AuxiliaryMethods::read_graph_txt_file("ZINC_test");
    gdb_3.erase(gdb_3.begin() + 0);
    GraphDatabase gdb_new_3;
    for (auto i : indices_test)
    {
        gdb_new_3.push_back(gdb_3[int(i)]);
    }
    cout << gdb_new_3.size() << endl;
    cout << "$$$" << endl;

    vector<vector<unsigned long>> node_labels;

    uint m_num_labels = 0;
    unordered_map<int, int> m_label_to_index;

    for (auto &g : gdb_new_1)
    {
        vector<unsigned long> colors =
            get_node_labels(g, use_node_labels, use_edge_labels);
        vector<unsigned long> new_color;

        for (auto &c : colors)
        {
            const auto it(m_label_to_index.find(c));
            if (it == m_label_to_index.end())
            {
                m_label_to_index.insert({{c, m_num_labels}});
                new_color.push_back(m_num_labels);

                m_num_labels++;
            }
            else
            {
                new_color.push_back(it->second);
            }
        }

        node_labels.push_back(new_color);
    }

    for (auto &g : gdb_new_2)
    {
        vector<unsigned long> colors =
            get_node_labels(g, use_node_labels, use_edge_labels);
        vector<unsigned long> new_color;

        for (auto &c : colors)
        {
            const auto it(m_label_to_index.find(c));
            if (it == m_label_to_index.end())
            {
                m_label_to_index.insert({{c, m_num_labels}});
                new_color.push_back(m_num_labels);

                m_num_labels++;
            }
            else
            {
                new_color.push_back(it->second);
            }
        }

        node_labels.push_back(new_color);
    }

    for (auto &g : gdb_new_3)
    {
        vector<unsigned long> colors =
            get_node_labels(g, use_node_labels, use_edge_labels);
        vector<unsigned long> new_color;

        for (auto &c : colors)
        {
            const auto it(m_label_to_index.find(c));
            if (it == m_label_to_index.end())
            {
                m_label_to_index.insert({{c, m_num_labels}});
                new_color.push_back(m_num_labels);

                m_num_labels++;
            }
            else
            {
                new_color.push_back(it->second);
            }
        }

        node_labels.push_back(new_color);
    }

    cout << m_num_labels << endl;
    return node_labels;
}

vector<vector<unsigned long>> get_all_node_labels_ZINC_con(
    const bool use_node_labels, const bool use_edge_labels,
    const std::vector<int> &indices_train, const std::vector<int> &indices_val,
    const std::vector<int> &indices_test)
{
    GraphDatabase gdb_1 = AuxiliaryMethods::read_graph_txt_file("ZINC_train");
    gdb_1.erase(gdb_1.begin() + 0);

    GraphDatabase gdb_new_1;
    for (auto i : indices_train)
    {
        gdb_new_1.push_back(gdb_1[int(i)]);
    }
    cout << gdb_new_1.size() << endl;
    cout << "$$$" << endl;

    GraphDatabase gdb_2 = AuxiliaryMethods::read_graph_txt_file("ZINC_val");
    gdb_2.erase(gdb_2.begin() + 0);

    GraphDatabase gdb_new_2;
    for (auto i : indices_val)
    {
        gdb_new_2.push_back(gdb_2[int(i)]);
    }
    cout << gdb_new_2.size() << endl;
    cout << "$$$" << endl;

    GraphDatabase gdb_3 = AuxiliaryMethods::read_graph_txt_file("ZINC_test");
    gdb_3.erase(gdb_3.begin() + 0);
    GraphDatabase gdb_new_3;
    for (auto i : indices_test)
    {
        gdb_new_3.push_back(gdb_3[int(i)]);
    }
    cout << gdb_new_3.size() << endl;
    cout << "$$$" << endl;

    vector<vector<unsigned long>> node_labels;

    uint m_num_labels = 0;
    unordered_map<int, int> m_label_to_index;

    for (auto &g : gdb_new_1)
    {
        vector<unsigned long> colors =
            get_node_labels_con(g, use_node_labels, use_edge_labels);

        vector<unsigned long> new_color;

        for (auto &c : colors)
        {
            const auto it(m_label_to_index.find(c));
            if (it == m_label_to_index.end())
            {
                m_label_to_index.insert({{c, m_num_labels}});
                new_color.push_back(m_num_labels);

                m_num_labels++;
            }
            else
            {
                new_color.push_back(it->second);
            }
        }

        node_labels.push_back(new_color);
    }

    for (auto &g : gdb_new_2)
    {
        vector<unsigned long> colors =
            get_node_labels_con(g, use_node_labels, use_edge_labels);
        vector<unsigned long> new_color;

        for (auto &c : colors)
        {
            const auto it(m_label_to_index.find(c));
            if (it == m_label_to_index.end())
            {
                m_label_to_index.insert({{c, m_num_labels}});
                new_color.push_back(m_num_labels);

                m_num_labels++;
            }
            else
            {
                new_color.push_back(it->second);
            }
        }

        node_labels.push_back(new_color);
    }

    for (auto &g : gdb_new_3)
    {
        vector<unsigned long> colors =
            get_node_labels_con(g, use_node_labels, use_edge_labels);
        vector<unsigned long> new_color;

        for (auto &c : colors)
        {
            const auto it(m_label_to_index.find(c));
            if (it == m_label_to_index.end())
            {
                m_label_to_index.insert({{c, m_num_labels}});
                new_color.push_back(m_num_labels);

                m_num_labels++;
            }
            else
            {
                new_color.push_back(it->second);
            }
        }

        node_labels.push_back(new_color);
    }

    cout << m_num_labels << endl;
    return node_labels;
}

vector<vector<unsigned long>> get_all_node_labels_ZINC_connected(
    const bool use_node_labels, const bool use_edge_labels,
    const std::vector<int> &indices_train, const std::vector<int> &indices_val,
    const std::vector<int> &indices_test)
{
    GraphDatabase gdb_1 = AuxiliaryMethods::read_graph_txt_file("ZINC_train");
    gdb_1.erase(gdb_1.begin() + 0);

    GraphDatabase gdb_new_1;
    for (auto i : indices_train)
    {
        gdb_new_1.push_back(gdb_1[int(i)]);
    }
    cout << gdb_new_1.size() << endl;
    cout << "$$$" << endl;

    GraphDatabase gdb_2 = AuxiliaryMethods::read_graph_txt_file("ZINC_val");
    gdb_2.erase(gdb_2.begin() + 0);

    GraphDatabase gdb_new_2;
    for (auto i : indices_val)
    {
        gdb_new_2.push_back(gdb_2[int(i)]);
    }
    cout << gdb_new_2.size() << endl;
    cout << "$$$" << endl;

    GraphDatabase gdb_3 = AuxiliaryMethods::read_graph_txt_file("ZINC_test");
    gdb_3.erase(gdb_3.begin() + 0);
    GraphDatabase gdb_new_3;
    for (auto i : indices_test)
    {
        gdb_new_3.push_back(gdb_3[int(i)]);
    }
    cout << gdb_new_3.size() << endl;
    cout << "$$$" << endl;

    vector<vector<unsigned long>> node_labels;

    uint m_num_labels = 0;
    unordered_map<int, int> m_label_to_index;

    for (auto &g : gdb_new_1)
    {
        vector<unsigned long> colors =
            get_node_labels_connected(g, use_node_labels, use_edge_labels);
        vector<unsigned long> new_color;

        for (auto &c : colors)
        {
            const auto it(m_label_to_index.find(c));
            if (it == m_label_to_index.end())
            {
                m_label_to_index.insert({{c, m_num_labels}});
                new_color.push_back(m_num_labels);

                m_num_labels++;
            }
            else
            {
                new_color.push_back(it->second);
            }
        }

        node_labels.push_back(new_color);
    }

    for (auto &g : gdb_new_2)
    {
        vector<unsigned long> colors =
            get_node_labels_connected(g, use_node_labels, use_edge_labels);
        vector<unsigned long> new_color;

        for (auto &c : colors)
        {
            const auto it(m_label_to_index.find(c));
            if (it == m_label_to_index.end())
            {
                m_label_to_index.insert({{c, m_num_labels}});
                new_color.push_back(m_num_labels);

                m_num_labels++;
            }
            else
            {
                new_color.push_back(it->second);
            }
        }

        node_labels.push_back(new_color);
    }

    for (auto &g : gdb_new_3)
    {
        vector<unsigned long> colors =
            get_node_labels_connected(g, use_node_labels, use_edge_labels);
        vector<unsigned long> new_color;

        for (auto &c : colors)
        {
            const auto it(m_label_to_index.find(c));
            if (it == m_label_to_index.end())
            {
                m_label_to_index.insert({{c, m_num_labels}});
                new_color.push_back(m_num_labels);

                m_num_labels++;
            }
            else
            {
                new_color.push_back(it->second);
            }
        }

        node_labels.push_back(new_color);
    }

    cout << m_num_labels << endl;
    return node_labels;
}

vector<vector<unsigned long>> get_all_node_labels_allchem(
    const bool use_node_labels, const bool use_edge_labels,
    const std::vector<int> &indices_train, const std::vector<int> &indices_val,
    const std::vector<int> &indices_test)
{
    GraphDatabase gdb_1 = AuxiliaryMethods::read_graph_txt_file("alchemy_full");
    gdb_1.erase(gdb_1.begin() + 0);

    GraphDatabase gdb_new_1;
    for (auto i : indices_train)
    {
        gdb_new_1.push_back(gdb_1[int(i)]);
    }
    cout << gdb_new_1.size() << endl;
    cout << "$$$" << endl;

    for (auto i : indices_val)
    {
        gdb_new_1.push_back(gdb_1[int(i)]);
    }
    cout << gdb_new_1.size() << endl;
    cout << "$$$" << endl;

    for (auto i : indices_test)
    {
        gdb_new_1.push_back(gdb_1[int(i)]);
    }
    cout << gdb_new_1.size() << endl;
    cout << "$$$" << endl;

    vector<vector<unsigned long>> node_labels;

    uint m_num_labels = 0;
    unordered_map<int, int> m_label_to_index;

    for (auto &g : gdb_new_1)
    {
        vector<unsigned long> colors =
            get_node_labels(g, use_node_labels, use_edge_labels);
        vector<unsigned long> new_color;

        for (auto &c : colors)
        {
            const auto it(m_label_to_index.find(c));
            if (it == m_label_to_index.end())
            {
                m_label_to_index.insert({{c, m_num_labels}});
                new_color.push_back(m_num_labels);

                m_num_labels++;
            }
            else
            {
                new_color.push_back(it->second);
            }
        }

        node_labels.push_back(new_color);
    }

    cout << m_num_labels << endl;
    return node_labels;
}

vector<vector<unsigned long>> get_all_node_labels_brec(
    const bool use_node_labels, const bool use_edge_labels)
{
    GraphDatabase gdb_1 = AuxiliaryMethods::read_graph_txt_file("brec_v3_3wl");
    gdb_1.erase(gdb_1.begin() + 0);

    GraphDatabase gdb_new_1;
    for (auto data : gdb_1)
    {
        gdb_new_1.push_back(data);
    }
    // for (auto i : indices_train)
    // {
    //     gdb_new_1.push_back(gdb_1[int(i)]);
    // }
    // cout << gdb_new_1.size() << endl;
    // cout << "$$$" << endl;

    // for (auto i : indices_val)
    // {
    //     gdb_new_1.push_back(gdb_1[int(i)]);
    // }
    // cout << gdb_new_1.size() << endl;
    // cout << "$$$" << endl;

    // for (auto i : indices_test)
    // {
    //     gdb_new_1.push_back(gdb_1[int(i)]);
    // }
    // cout << gdb_new_1.size() << endl;
    // cout << "$$$" << endl;

    vector<vector<unsigned long>> node_labels;

    uint m_num_labels = 0;
    unordered_map<int, int> m_label_to_index;

    for (auto &g : gdb_new_1)
    {
        vector<unsigned long> colors =
            get_node_labels(g, use_node_labels, use_edge_labels);
        vector<unsigned long> new_color;

        for (auto &c : colors)
        {
            const auto it(m_label_to_index.find(c));
            if (it == m_label_to_index.end())
            {
                m_label_to_index.insert({{c, m_num_labels}});
                new_color.push_back(m_num_labels);

                m_num_labels++;
            }
            else
            {
                new_color.push_back(it->second);
            }
        }

        node_labels.push_back(new_color);
    }

    cout << m_num_labels << endl;
    return node_labels;
}

vector<vector<unsigned long>> get_all_node_labels_allchem_con(
    const bool use_node_labels, const bool use_edge_labels,
    const std::vector<int> &indices_train, const std::vector<int> &indices_val,
    const std::vector<int> &indices_test)
{
    GraphDatabase gdb_1 = AuxiliaryMethods::read_graph_txt_file("alchemy_full");
    gdb_1.erase(gdb_1.begin() + 0);

    GraphDatabase gdb_new_1;
    for (auto i : indices_train)
    {
        gdb_new_1.push_back(gdb_1[int(i)]);
    }
    cout << gdb_new_1.size() << endl;
    cout << "$$$" << endl;

    for (auto i : indices_val)
    {
        gdb_new_1.push_back(gdb_1[int(i)]);
    }
    cout << gdb_new_1.size() << endl;
    cout << "$$$" << endl;

    for (auto i : indices_test)
    {
        gdb_new_1.push_back(gdb_1[int(i)]);
    }
    cout << gdb_new_1.size() << endl;
    cout << "$$$" << endl;

    vector<vector<unsigned long>> node_labels;

    uint m_num_labels = 0;
    unordered_map<int, int> m_label_to_index;

    for (auto &g : gdb_new_1)
    {
        vector<unsigned long> colors =
            get_node_labels_con(g, use_node_labels, use_edge_labels);
        vector<unsigned long> new_color;

        for (auto &c : colors)
        {
            const auto it(m_label_to_index.find(c));
            if (it == m_label_to_index.end())
            {
                m_label_to_index.insert({{c, m_num_labels}});
                new_color.push_back(m_num_labels);

                m_num_labels++;
            }
            else
            {
                new_color.push_back(it->second);
            }
        }

        node_labels.push_back(new_color);
    }

    cout << m_num_labels << endl;
    return node_labels;
}

vector<vector<unsigned long>> get_all_node_labels_allchem_unc(
    const bool use_node_labels, const bool use_edge_labels,
    const std::vector<int> &indices_train, const std::vector<int> &indices_val,
    const std::vector<int> &indices_test)
{
    GraphDatabase gdb_1 = AuxiliaryMethods::read_graph_txt_file("alchemy_full");
    gdb_1.erase(gdb_1.begin() + 0);

    GraphDatabase gdb_new_1;
    for (auto i : indices_train)
    {
        gdb_new_1.push_back(gdb_1[int(i)]);
    }
    cout << gdb_new_1.size() << endl;
    cout << "$$$" << endl;

    for (auto i : indices_val)
    {
        gdb_new_1.push_back(gdb_1[int(i)]);
    }
    cout << gdb_new_1.size() << endl;
    cout << "$$$" << endl;

    for (auto i : indices_test)
    {
        gdb_new_1.push_back(gdb_1[int(i)]);
    }
    cout << gdb_new_1.size() << endl;
    cout << "$$$" << endl;

    vector<vector<unsigned long>> node_labels;

    uint m_num_labels = 0;
    unordered_map<int, int> m_label_to_index;

    for (auto &g : gdb_new_1)
    {
        vector<unsigned long> colors =
            get_node_labels_unc(g, use_node_labels, use_edge_labels);
        vector<unsigned long> new_color;

        for (auto &c : colors)
        {
            const auto it(m_label_to_index.find(c));
            if (it == m_label_to_index.end())
            {
                m_label_to_index.insert({{c, m_num_labels}});
                new_color.push_back(m_num_labels);

                m_num_labels++;
            }
            else
            {
                new_color.push_back(it->second);
            }
        }

        node_labels.push_back(new_color);
    }

    cout << m_num_labels << endl;
    return node_labels;
}

vector<vector<unsigned long>> get_all_node_labels_ZINC_3(
    const bool use_node_labels, const bool use_edge_labels,
    const std::vector<int> &indices_train, const std::vector<int> &indices_val,
    const std::vector<int> &indices_test)
{
    GraphDatabase gdb_1 = AuxiliaryMethods::read_graph_txt_file("ZINC_train");
    gdb_1.erase(gdb_1.begin() + 0);

    GraphDatabase gdb_new_1;
    for (auto i : indices_train)
    {
        gdb_new_1.push_back(gdb_1[int(i)]);
    }
    cout << gdb_new_1.size() << endl;
    cout << "$$$" << endl;

    GraphDatabase gdb_2 = AuxiliaryMethods::read_graph_txt_file("ZINC_val");
    gdb_2.erase(gdb_2.begin() + 0);

    GraphDatabase gdb_new_2;
    for (auto i : indices_val)
    {
        gdb_new_2.push_back(gdb_2[int(i)]);
    }
    cout << gdb_new_2.size() << endl;
    cout << "$$$" << endl;

    GraphDatabase gdb_3 = AuxiliaryMethods::read_graph_txt_file("ZINC_test");
    gdb_3.erase(gdb_3.begin() + 0);
    GraphDatabase gdb_new_3;
    for (auto i : indices_test)
    {
        gdb_new_3.push_back(gdb_3[int(i)]);
    }
    cout << gdb_new_3.size() << endl;
    cout << "$$$" << endl;

    vector<vector<unsigned long>> node_labels;

    uint m_num_labels = 0;
    unordered_map<int, int> m_label_to_index;

    for (auto &g : gdb_new_1)
    {
        vector<unsigned long> colors =
            get_node_labels_3(g, use_node_labels, use_edge_labels);
        vector<unsigned long> new_color;

        for (auto &c : colors)
        {
            const auto it(m_label_to_index.find(c));
            if (it == m_label_to_index.end())
            {
                m_label_to_index.insert({{c, m_num_labels}});
                new_color.push_back(m_num_labels);

                m_num_labels++;
            }
            else
            {
                new_color.push_back(it->second);
            }
        }

        node_labels.push_back(new_color);
    }

    for (auto &g : gdb_new_2)
    {
        vector<unsigned long> colors =
            get_node_labels_3(g, use_node_labels, use_edge_labels);
        vector<unsigned long> new_color;

        for (auto &c : colors)
        {
            const auto it(m_label_to_index.find(c));
            if (it == m_label_to_index.end())
            {
                m_label_to_index.insert({{c, m_num_labels}});
                new_color.push_back(m_num_labels);

                m_num_labels++;
            }
            else
            {
                new_color.push_back(it->second);
            }
        }

        node_labels.push_back(new_color);
    }

    for (auto &g : gdb_new_3)
    {
        vector<unsigned long> colors =
            get_node_labels_3(g, use_node_labels, use_edge_labels);
        vector<unsigned long> new_color;

        for (auto &c : colors)
        {
            const auto it(m_label_to_index.find(c));
            if (it == m_label_to_index.end())
            {
                m_label_to_index.insert({{c, m_num_labels}});
                new_color.push_back(m_num_labels);

                m_num_labels++;
            }
            else
            {
                new_color.push_back(it->second);
            }
        }

        node_labels.push_back(new_color);
    }

    cout << m_num_labels << endl;
    return node_labels;
}

vector<vector<unsigned long>> get_all_node_labels_allchem_3(
    const bool use_node_labels, const bool use_edge_labels,
    const std::vector<int> &indices_train, const std::vector<int> &indices_val,
    const std::vector<int> &indices_test)
{
    GraphDatabase gdb_1 = AuxiliaryMethods::read_graph_txt_file("alchemy_full");
    gdb_1.erase(gdb_1.begin() + 0);

    GraphDatabase gdb_new_1;
    for (auto i : indices_train)
    {
        gdb_new_1.push_back(gdb_1[int(i)]);
    }
    cout << gdb_new_1.size() << endl;
    cout << "$$$" << endl;

    for (auto i : indices_val)
    {
        gdb_new_1.push_back(gdb_1[int(i)]);
    }

    for (auto i : indices_test)
    {
        gdb_new_1.push_back(gdb_1[int(i)]);
    }

    vector<vector<unsigned long>> node_labels;

    uint m_num_labels = 0;
    unordered_map<int, int> m_label_to_index;

    for (auto &g : gdb_new_1)
    {
        vector<unsigned long> colors =
            get_node_labels_3(g, use_node_labels, use_edge_labels);
        vector<unsigned long> new_color;

        for (auto &c : colors)
        {
            const auto it(m_label_to_index.find(c));
            if (it == m_label_to_index.end())
            {
                m_label_to_index.insert({{c, m_num_labels}});
                new_color.push_back(m_num_labels);

                m_num_labels++;
            }
            else
            {
                new_color.push_back(it->second);
            }
        }

        node_labels.push_back(new_color);
    }

    cout << m_num_labels << endl;
    return node_labels;
}

vector<vector<unsigned long>> get_all_node_labels_ZINC_1(
    const bool use_node_labels, const bool use_edge_labels,
    const std::vector<int> &indices_train, const std::vector<int> &indices_val,
    const std::vector<int> &indices_test)
{
    GraphDatabase gdb_1 = AuxiliaryMethods::read_graph_txt_file("ZINC_train");
    gdb_1.erase(gdb_1.begin() + 0);

    GraphDatabase gdb_new_1;
    for (auto i : indices_train)
    {
        gdb_new_1.push_back(gdb_1[int(i)]);
    }
    cout << gdb_new_1.size() << endl;
    cout << "$$$" << endl;

    GraphDatabase gdb_2 = AuxiliaryMethods::read_graph_txt_file("ZINC_val");
    gdb_2.erase(gdb_2.begin() + 0);

    GraphDatabase gdb_new_2;
    for (auto i : indices_val)
    {
        gdb_new_2.push_back(gdb_2[int(i)]);
    }
    cout << gdb_new_2.size() << endl;
    cout << "$$$" << endl;

    GraphDatabase gdb_3 = AuxiliaryMethods::read_graph_txt_file("ZINC_test");
    gdb_3.erase(gdb_3.begin() + 0);
    GraphDatabase gdb_new_3;
    for (auto i : indices_test)
    {
        gdb_new_3.push_back(gdb_3[int(i)]);
    }
    cout << gdb_new_3.size() << endl;
    cout << "$$$" << endl;

    vector<vector<unsigned long>> node_labels;

    uint m_num_labels = 0;
    unordered_map<int, int> m_label_to_index;

    for (auto &g : gdb_new_1)
    {
        vector<unsigned long> colors = get_node_labels_1(g, use_node_labels);
        vector<unsigned long> new_color;

        for (auto &c : colors)
        {
            const auto it(m_label_to_index.find(c));
            if (it == m_label_to_index.end())
            {
                m_label_to_index.insert({{c, m_num_labels}});
                new_color.push_back(m_num_labels);

                m_num_labels++;
            }
            else
            {
                new_color.push_back(it->second);
            }
        }

        node_labels.push_back(new_color);
    }

    for (auto &g : gdb_new_2)
    {
        vector<unsigned long> colors = get_node_labels_1(g, use_node_labels);
        vector<unsigned long> new_color;

        for (auto &c : colors)
        {
            const auto it(m_label_to_index.find(c));
            if (it == m_label_to_index.end())
            {
                m_label_to_index.insert({{c, m_num_labels}});
                new_color.push_back(m_num_labels);

                m_num_labels++;
            }
            else
            {
                new_color.push_back(it->second);
            }
        }

        node_labels.push_back(new_color);
    }

    for (auto &g : gdb_new_3)
    {
        vector<unsigned long> colors = get_node_labels_1(g, use_node_labels);
        vector<unsigned long> new_color;

        for (auto &c : colors)
        {
            const auto it(m_label_to_index.find(c));
            if (it == m_label_to_index.end())
            {
                m_label_to_index.insert({{c, m_num_labels}});
                new_color.push_back(m_num_labels);

                m_num_labels++;
            }
            else
            {
                new_color.push_back(it->second);
            }
        }

        node_labels.push_back(new_color);
    }

    cout << m_num_labels << endl;
    return node_labels;
}

vector<vector<unsigned long>> get_all_node_labels_alchem_1(
    const bool use_node_labels, const bool use_edge_labels,
    const std::vector<int> &indices_train, const std::vector<int> &indices_val,
    const std::vector<int> &indices_test)
{
    GraphDatabase gdb_1 = AuxiliaryMethods::read_graph_txt_file("alchemy_full");
    gdb_1.erase(gdb_1.begin() + 0);

    GraphDatabase gdb_new_1;
    for (auto i : indices_train)
    {
        gdb_new_1.push_back(gdb_1[int(i)]);
    }
    cout << gdb_new_1.size() << endl;
    cout << "$$$" << endl;

    for (auto i : indices_val)
    {
        gdb_new_1.push_back(gdb_1[int(i)]);
    }
    cout << gdb_new_1.size() << endl;
    cout << "$$$" << endl;

    for (auto i : indices_test)
    {
        gdb_new_1.push_back(gdb_1[int(i)]);
    }
    cout << gdb_new_1.size() << endl;
    cout << "$$$" << endl;

    vector<vector<unsigned long>> node_labels;

    uint m_num_labels = 0;
    unordered_map<int, int> m_label_to_index;

    for (auto &g : gdb_new_1)
    {
        vector<unsigned long> colors = get_node_labels_1(g, use_node_labels);
        vector<unsigned long> new_color;

        for (auto &c : colors)
        {
            const auto it(m_label_to_index.find(c));
            if (it == m_label_to_index.end())
            {
                m_label_to_index.insert({{c, m_num_labels}});
                new_color.push_back(m_num_labels);

                m_num_labels++;
            }
            else
            {
                new_color.push_back(it->second);
            }
        }

        node_labels.push_back(new_color);
    }

    cout << m_num_labels << endl;
    return node_labels;
}

vector<vector<int>> get_all_edge_labels_ZINC_1(
    const bool use_node_labels, const bool use_edge_labels,
    const std::vector<int> &indices_train, const std::vector<int> &indices_val,
    const std::vector<int> &indices_test)
{
    GraphDatabase gdb_1 = AuxiliaryMethods::read_graph_txt_file("ZINC_train");
    gdb_1.erase(gdb_1.begin() + 0);

    GraphDatabase gdb_new_1;
    for (auto i : indices_train)
    {
        gdb_new_1.push_back(gdb_1[int(i)]);
    }
    cout << gdb_new_1.size() << endl;
    cout << "$$$" << endl;

    GraphDatabase gdb_2 = AuxiliaryMethods::read_graph_txt_file("ZINC_val");
    gdb_2.erase(gdb_2.begin() + 0);

    GraphDatabase gdb_new_2;
    for (auto i : indices_val)
    {
        gdb_new_2.push_back(gdb_2[int(i)]);
    }
    cout << gdb_new_2.size() << endl;
    cout << "$$$" << endl;

    GraphDatabase gdb_3 = AuxiliaryMethods::read_graph_txt_file("ZINC_test");
    gdb_3.erase(gdb_3.begin() + 0);
    GraphDatabase gdb_new_3;
    for (auto i : indices_test)
    {
        gdb_new_3.push_back(gdb_3[int(i)]);
    }
    cout << gdb_new_3.size() << endl;
    cout << "$$$" << endl;

    vector<vector<int>> edge_labels;

    uint m_num_labels = 0;
    unordered_map<int, int> m_label_to_index;

    for (auto &g : gdb_new_1)
    {
        vector<int> colors = get_edge_labels_1(g);
        vector<int> new_color;

        for (auto &c : colors)
        {
            const auto it(m_label_to_index.find(c));
            if (it == m_label_to_index.end())
            {
                m_label_to_index.insert({{c, m_num_labels}});
                new_color.push_back(m_num_labels);

                m_num_labels++;
            }
            else
            {
                new_color.push_back(it->second);
            }
        }

        edge_labels.push_back(new_color);
    }

    for (auto &g : gdb_new_2)
    {
        vector<int> colors = get_edge_labels_1(g);
        vector<int> new_color;

        for (auto &c : colors)
        {
            const auto it(m_label_to_index.find(c));
            if (it == m_label_to_index.end())
            {
                m_label_to_index.insert({{c, m_num_labels}});
                new_color.push_back(m_num_labels);

                m_num_labels++;
            }
            else
            {
                new_color.push_back(it->second);
            }
        }

        edge_labels.push_back(new_color);
    }

    for (auto &g : gdb_new_3)
    {
        vector<int> colors = get_edge_labels_1(g);
        vector<int> new_color;

        for (auto &c : colors)
        {
            const auto it(m_label_to_index.find(c));
            if (it == m_label_to_index.end())
            {
                m_label_to_index.insert({{c, m_num_labels}});
                new_color.push_back(m_num_labels);

                m_num_labels++;
            }
            else
            {
                new_color.push_back(it->second);
            }
        }

        edge_labels.push_back(new_color);
    }

    cout << m_num_labels << endl;
    return edge_labels;
}

vector<vector<int>> get_all_edge_labels_alchem_1(
    const bool use_node_labels, const bool use_edge_labels,
    const std::vector<int> &indices_train, const std::vector<int> &indices_val,
    const std::vector<int> &indices_test)
{
    GraphDatabase gdb_1 = AuxiliaryMethods::read_graph_txt_file("alchemy_full");
    gdb_1.erase(gdb_1.begin() + 0);

    GraphDatabase gdb_new_1;
    for (auto i : indices_train)
    {
        gdb_new_1.push_back(gdb_1[int(i)]);
    }
    cout << gdb_new_1.size() << endl;
    cout << "$$$" << endl;

    for (auto i : indices_val)
    {
        gdb_new_1.push_back(gdb_1[int(i)]);
    }
    cout << gdb_new_1.size() << endl;
    cout << "$$$" << endl;

    for (auto i : indices_test)
    {
        gdb_new_1.push_back(gdb_1[int(i)]);
    }
    cout << gdb_new_1.size() << endl;
    cout << "$$$" << endl;

    vector<vector<int>> edge_labels;

    uint m_num_labels = 0;
    unordered_map<int, int> m_label_to_index;

    for (auto &g : gdb_new_1)
    {
        vector<int> colors = get_edge_labels_1(g);
        vector<int> new_color;

        for (auto &c : colors)
        {
            const auto it(m_label_to_index.find(c));
            if (it == m_label_to_index.end())
            {
                m_label_to_index.insert({{c, m_num_labels}});
                new_color.push_back(m_num_labels);

                m_num_labels++;
            }
            else
            {
                new_color.push_back(it->second);
            }
        }

        edge_labels.push_back(new_color);
    }

    cout << m_num_labels << endl;
    return edge_labels;
}

vector<pair<vector<int>, vector<int>>>

get_all_edge_labelslg_ZINC_1(const bool use_node_labels,
                             const bool use_edge_labels,
                             const std::vector<int> &indices_train,
                             const std::vector<int> &indices_val,
                             const std::vector<int> &indices_test)
{
    GraphDatabase gdb_1 = AuxiliaryMethods::read_graph_txt_file("ZINC_train");
    gdb_1.erase(gdb_1.begin() + 0);

    GraphDatabase gdb_new_1;
    for (auto i : indices_train)
    {
        gdb_new_1.push_back(gdb_1[int(i)]);
    }
    cout << gdb_new_1.size() << endl;
    cout << "$$$" << endl;

    GraphDatabase gdb_2 = AuxiliaryMethods::read_graph_txt_file("ZINC_val");
    gdb_2.erase(gdb_2.begin() + 0);

    GraphDatabase gdb_new_2;
    for (auto i : indices_val)
    {
        gdb_new_2.push_back(gdb_2[int(i)]);
    }
    cout << gdb_new_2.size() << endl;
    cout << "$$$" << endl;

    GraphDatabase gdb_3 = AuxiliaryMethods::read_graph_txt_file("ZINC_test");
    gdb_3.erase(gdb_3.begin() + 0);
    GraphDatabase gdb_new_3;
    for (auto i : indices_test)
    {
        gdb_new_3.push_back(gdb_3[int(i)]);
    }
    cout << gdb_new_3.size() << endl;
    cout << "$$$" << endl;

    vector<pair<vector<int>, vector<int>>> edge_labels;

    for (auto &g : gdb_new_1)
    {
        pair<vector<int>, vector<int>> colors = get_edge_labels(g, true, true);

        edge_labels.push_back(colors);
    }

    for (auto &g : gdb_new_2)
    {
        pair<vector<int>, vector<int>> colors = get_edge_labels(g, true, true);

        edge_labels.push_back(colors);
    }

    for (auto &g : gdb_new_3)
    {
        pair<vector<int>, vector<int>> colors = get_edge_labels(g, true, true);

        edge_labels.push_back(colors);
    }

    return edge_labels;
}

vector<int> read_classes(string data_set_name)
{
    return AuxiliaryMethods::read_classes(data_set_name);
}

vector<float> read_targets(string data_set_name,
                           const std::vector<int> &indices)
{
    vector<float> targets = AuxiliaryMethods::read_targets(data_set_name);

    vector<float> new_targets;
    for (auto i : indices)
    {
        new_targets.push_back(targets[i]);
    }

    return new_targets;
}

PYBIND11_MODULE(preprocessing, m)
{
    m.def("get_all_matrices", &get_all_matrices);
    m.def("get_all_matrices_brec", &get_all_matrices_brec);
    m.def("get_all_matrices_connected", &get_all_matrices_connected);
    m.def("get_all_matrices_con", &get_all_matrices_con);
    m.def("get_all_matrices_unc", &get_all_matrices_unc);

    m.def("get_all_matrices_wl", &get_all_matrices_wl);
    m.def("get_all_matrices_dwl", &get_all_matrices_dwl);
    m.def("get_all_matrices_dwle", &get_all_matrices_dwle);
    m.def("get_all_matrices_1", &get_all_matrices_1);
    m.def("get_all_matrices_3", &get_all_matrices_3);
    m.def("get_all_matrices_3_connected", &get_all_matrices_3_connected);
    m.def("get_all_node_labels", &get_all_node_labels);
    m.def("get_all_node_labels_1", &get_all_node_labels_1);
    m.def("get_all_edge_labels_1", &get_all_edge_labels_1);
    m.def("get_all_matrices_3", &get_all_matrices_3);
    m.def("get_all_node_labels_3", &get_all_node_labels_3);
    m.def("get_all_node_labels_ZINC", &get_all_node_labels_ZINC);
    m.def("get_all_node_labels_ZINC_connected",
          &get_all_node_labels_ZINC_connected);
    m.def("get_all_node_labels_ZINC_1", &get_all_node_labels_ZINC_1);

    m.def("get_all_node_labels_allchem", &get_all_node_labels_allchem);
    m.def("get_all_node_labels_brec", &get_all_node_labels_brec);

    m.def("get_all_node_labels_allchem_con", &get_all_node_labels_allchem_con);
    m.def("get_all_node_labels_allchem_unc", &get_all_node_labels_allchem_unc);

    m.def("get_all_node_labels_ZINC_con", &get_all_node_labels_ZINC_con);

    m.def("get_all_node_labels_ZINC_3", &get_all_node_labels_ZINC_3);
    m.def("get_all_node_labels_allchem_3", &get_all_node_labels_allchem_3);
    m.def("get_all_edge_labels_ZINC_1", &get_all_edge_labels_ZINC_1);

    m.def("get_all_node_labels_alchem_1", &get_all_node_labels_alchem_1);
    m.def("get_all_edge_labels_alchem_1", &get_all_edge_labels_alchem_1);
    m.def("get_all_edge_labelslg_ZINC_1", &get_all_edge_labels_ZINC_1);
    m.def("get_all_attributes", &get_all_attributes);
    m.def("read_classes", &read_classes);
    m.def("read_targets", &read_targets);
}