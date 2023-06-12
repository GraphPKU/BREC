#ifndef WLFAST_AUXILIARYMETHODS_H
#define WLFAST_AUXILIARYMETHODS_H

#include <fstream>
#include <iostream>
#include <string>
#include <sstream>
#include <algorithm>

#include <unordered_map>
#include "Graph.h"

using namespace std;
using namespace GraphLibrary;



namespace AuxiliaryMethods {
    // Simple function for converting a comma separated string into a vector of integers.
    vector<int> split_string(string s);

    // Simple function for converting a comma separated string into a vector of floats.
    vector<float> split_string_float(string s);

    // Reading a graph database from txt file.
    GraphDatabase read_graph_txt_file(string data_set_name);

    vector<int> read_classes(string data_set_name);
    vector<vector<float>> read_multi_targets(string data_set_name);
    vector<float> read_targets(string data_set_name);

    // Pairing function to map to a pair of Labels to a single label.
    Label pairing(const Label a, const Label b);
}

#endif // WLFAST_AUXILIARYMETHODS_H