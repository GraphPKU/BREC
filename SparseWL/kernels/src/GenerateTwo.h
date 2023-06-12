

#ifndef WLFAST_GENERATETWO_H
#define WLFAST_GENERATETWO_H

#include <cmath>
#include <unordered_map>
#include <queue>

#include "Graph.h"

using TwoTuple = tuple<Node, Node>;

using namespace GraphLibrary;

namespace GenerateTwo {
    class GenerateTwo {
    public:
        GenerateTwo(const GraphDatabase &graph_database);

        GramMatrix compute_gram_matrix(const uint num_iterations, const bool use_labels, const bool use_edge_labels, const string algorithm,
                                       const bool simple, const bool compute_gram);

        Graph generate_local_graph(const Graph &g, const bool use_labels, const bool use_edge_labels);
        Graph generate_local_graph_connected(const Graph &g, const bool use_labels, const bool use_edge_labels);

        GramMatrix generate_local_sparse_am(const Graph &g, const bool use_labels, const bool use_edge_labels);
        vector<int> get_edge_labels(const Graph &g, const bool use_labels, const bool use_edge_labels);
        Labels get_node_labels(const Graph &g, const bool use_labels, const bool use_edge_labels);

        Graph generate_global_graph(const Graph &g, const bool use_labels, const bool use_edge_labels);

        Graph generate_global_graph_malkin(const Graph &g, const bool use_labels, const bool use_edge_labels);

        ~GenerateTwo();

    private:
        GraphDatabase m_graph_database;

        // Computes labels for vertices of graph.
        ColorCounter
        compute_colors(const Graph &g, const uint num_iterations, const bool use_labels, const bool use_edge_labels, const string algorithm);

        ColorCounter
        compute_colors_simple(const Graph &g, const uint num_iterations, const bool use_labels, const bool use_edge_labels, const string algorithm);

        // Manage indices of of labels in feature vectors.
        ColorCounter m_label_to_index;

        // Counts number of distinct labels over all graphs.
        uint m_num_labels;
    };
}

#endif //WLFAST_GENERATETWO_H
