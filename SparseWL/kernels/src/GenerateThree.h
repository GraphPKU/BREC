

#ifndef WLFAST_GENERATETHREE_H
#define WLFAST_GENERATETHREE_H

#include <cmath>
#include <unordered_map>
#include <queue>

#include "Graph.h"

using ThreeTuple = tuple<Node, Node, Node>;

using namespace GraphLibrary;

namespace GenerateThree {
    class GenerateThree {
    public:
        GenerateThree(const GraphDatabase &graph_database);

        GramMatrix
        compute_gram_matrix(const uint num_iterations, const bool use_labels, const string algorithm,
                            const bool simple, const bool compute_gram);

        Graph generate_local_graph(const Graph &g, const bool use_labels);

        Graph generate_global_graph(const Graph &g, const bool use_labels);

        Graph generate_global_graph_malkin(const Graph &g, const bool use_labels);

        ~GenerateThree();

    private:
        GraphDatabase m_graph_database;

        // Computes labels for vertices of graph.
        ColorCounter
        compute_colors(const Graph &g, const uint num_iterations, const bool use_labels, const string algorithm);

        ColorCounter
        compute_colors_simple(const Graph &g, const uint num_iterations, const bool use_labels, const string algorithm);

        // Manage indices of of labels in feature vectors.
        ColorCounter m_label_to_index;

        // Counts number of distinct labels over all graphs.
        unsigned long m_num_labels;
    };
}

#endif //WLFAST_GENERATETHREE_H
