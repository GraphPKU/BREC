#include <cstdio>
#include "src/AuxiliaryMethods.h"
#include "src/ColorRefinementKernel.h"
#include "src/ShortestPathKernel.h"
#include "src/GraphletKernel.h"
#include "src/GenerateTwo.h"
#include "src/GenerateThree.h"
#include "src/Graph.h"
#include <iostream>
#include <chrono>

using namespace std::chrono;
using namespace std;
using namespace GraphLibrary;
using namespace std;

//template<typename T>
//std::ostream &operator<<(std::ostream &out, const std::vector<T> &v) {
//    if (!v.empty()) {
//        out << '[';
//        std::copy(v.begin(), v.end(), std::ostream_iterator<T>(out, ", "));
//        out << "\b\b]";
//    }
//    return out;
//}

int main() {
    // k = 2.
    {
        vector<pair<string, bool>> datasets = {make_pair("ENZYMES", true), make_pair("IMDB-BINARY", false),
                                               make_pair("IMDB-MULTI", false), make_pair("NCI1", true),
                                               make_pair("NCI109", true), make_pair("PTC_FM", true),
                                               make_pair("PROTEINS", true), make_pair("REDDIT-BINARY", false)};

        for (auto &d: datasets) {
            {
                string ds = std::get<0>(d);
                bool use_labels = std::get<1>(d);

                string kernel = "LWL2";
                GraphDatabase gdb = AuxiliaryMethods::read_graph_txt_file(ds);
                gdb.erase(gdb.begin() + 0);
                vector<int> classes = AuxiliaryMethods::read_classes(ds);

                GenerateTwo::GenerateTwo wl(gdb);
                for (uint i = 0; i <= 5; ++i) {
                    cout << ds + "__" + kernel + "_" + to_string(i) << endl;
                    GramMatrix gm;

                    if (i == 5) {
                        high_resolution_clock::time_point t1 = high_resolution_clock::now();
                        gm = wl.compute_gram_matrix(i, use_labels, false, "local", true, true);
                        high_resolution_clock::time_point t2 = high_resolution_clock::now();
                        auto duration = duration_cast<seconds>(t2 - t1).count();
                        cout << duration << endl;
                    } else {
                        gm = wl.compute_gram_matrix(i, use_labels, false, "local", true, true);
                    }

                    AuxiliaryMethods::write_libsvm(gm, classes,
                                                   "./svm/GM/EXP/" + ds + "__" + kernel + "_" + to_string(i) +
                                                   ".gram");
                }
            }

            {
                string ds = std::get<0>(d);
                bool use_labels = std::get<1>(d);

                string kernel = "LWLP2";
                GraphDatabase gdb = AuxiliaryMethods::read_graph_txt_file(ds);
                gdb.erase(gdb.begin() + 0);
                vector<int> classes = AuxiliaryMethods::read_classes(ds);

                GenerateTwo::GenerateTwo wl(gdb);
                for (uint i = 0; i <= 5; ++i) {
                    cout << ds + "__" + kernel + "_" + to_string(i) << endl;
                    GramMatrix gm;

                    if (i == 5) {
                        high_resolution_clock::time_point t1 = high_resolution_clock::now();
                        gm = wl.compute_gram_matrix(i, use_labels, false, "localp", true, true);
                        high_resolution_clock::time_point t2 = high_resolution_clock::now();
                        auto duration = duration_cast<seconds>(t2 - t1).count();
                        cout << duration << endl;
                    } else {
                        gm = wl.compute_gram_matrix(i, use_labels, false, "localp", true, true);
                    }

                    AuxiliaryMethods::write_libsvm(gm, classes,
                                                   "./svm/GM/EXP/" + ds + "__" + kernel + "_" + to_string(i) +
                                                   ".gram");
                }
            }
        }
    }

    {
        vector<pair<string, bool>> datasets = {make_pair("ENZYMES", true), make_pair("IMDB-BINARY", false),
                                               make_pair("IMDB-MULTI", false), make_pair("NCI1", true),
                                               make_pair("NCI109", true), make_pair("PTC_FM", true),
                                               make_pair("PROTEINS", true)};

        for (auto &d: datasets) {
            {
                string ds = std::get<0>(d);
                bool use_labels = std::get<1>(d);

                string kernel = "WL2";
                GraphDatabase gdb = AuxiliaryMethods::read_graph_txt_file(ds);
                gdb.erase(gdb.begin() + 0);
                vector<int> classes = AuxiliaryMethods::read_classes(ds);

                GenerateTwo::GenerateTwo wl(gdb);
                for (uint i = 0; i <= 5; ++i) {
                    cout << ds + "__" + kernel + "_" + to_string(i) << endl;
                    GramMatrix gm;

                    if (i == 5) {
                        high_resolution_clock::time_point t1 = high_resolution_clock::now();
                        gm = wl.compute_gram_matrix(i, use_labels, false, "wl", true, true);
                        high_resolution_clock::time_point t2 = high_resolution_clock::now();
                        auto duration = duration_cast<seconds>(t2 - t1).count();
                        cout << duration << endl;
                    } else {
                        gm = wl.compute_gram_matrix(i, use_labels, false, "wl", true, true);
                    }

                    AuxiliaryMethods::write_libsvm(gm, classes,
                                                   "./svm/GM/EXP/" + ds + "__" + kernel + "_" + to_string(i) +
                                                   ".gram");
                }
            }

            {
                string ds = std::get<0>(d);
                bool use_labels = std::get<1>(d);

                string kernel = "DWL2";
                GraphDatabase gdb = AuxiliaryMethods::read_graph_txt_file(ds);
                gdb.erase(gdb.begin() + 0);
                vector<int> classes = AuxiliaryMethods::read_classes(ds);

                GenerateTwo::GenerateTwo wl(gdb);
                for (uint i = 0; i <= 5; ++i) {
                    cout << ds + "__" + kernel + "_" + to_string(i) << endl;
                    GramMatrix gm;

                    if (i == 5) {
                        high_resolution_clock::time_point t1 = high_resolution_clock::now();
                        gm = wl.compute_gram_matrix(i, use_labels, false, "malkin", true, true);
                        high_resolution_clock::time_point t2 = high_resolution_clock::now();
                        auto duration = duration_cast<seconds>(t2 - t1).count();
                        cout << duration << endl;
                    } else {
                        gm = wl.compute_gram_matrix(i, use_labels, false, "malkin", true, true);
                    }

                    AuxiliaryMethods::write_libsvm(gm, classes,
                                                   "./svm/GM/EXP/" + ds + "__" + kernel + "_" + to_string(i) +
                                                   ".gram");
                }
            }
        }
    }

    // k = 3.
    {
        vector<pair<string, bool>> datasets = {make_pair("ENZYMES", true), make_pair("IMDB-BINARY", false),
                                               make_pair("IMDB-MULTI", false), make_pair("NCI1", true),
                                               make_pair("NCI109", true), make_pair("PTC_FM", true)};

        for (auto &d: datasets) {
            {
                string ds = std::get<0>(d);
                bool use_labels = std::get<1>(d);

                string kernel = "LWL3";
                GraphDatabase gdb = AuxiliaryMethods::read_graph_txt_file(ds);
                gdb.erase(gdb.begin() + 0);
                vector<int> classes = AuxiliaryMethods::read_classes(ds);

                GenerateThree::GenerateThree wl(gdb);
                for (uint i = 0; i <= 5; ++i) {
                    cout << ds + "__" + kernel + "_" + to_string(i) << endl;
                    GramMatrix gm;

                    if (i == 5) {
                        high_resolution_clock::time_point t1 = high_resolution_clock::now();
                        gm = wl.compute_gram_matrix(i, use_labels, "local", true, true);
                        high_resolution_clock::time_point t2 = high_resolution_clock::now();
                        auto duration = duration_cast<seconds>(t2 - t1).count();
                        cout << duration << endl;
                    } else {
                        gm = wl.compute_gram_matrix(i, use_labels, "local", true, true);
                    }

                    AuxiliaryMethods::write_libsvm(gm, classes,
                                                   "./svm/GM/EXP/" + ds + "__" + kernel + "_" + to_string(i) +
                                                   ".gram");
                }
            }

            {
                string ds = std::get<0>(d);
                bool use_labels = std::get<1>(d);

                string kernel = "LWLP3";
                GraphDatabase gdb = AuxiliaryMethods::read_graph_txt_file(ds);
                gdb.erase(gdb.begin() + 0);
                vector<int> classes = AuxiliaryMethods::read_classes(ds);

                GenerateThree::GenerateThree wl(gdb);
                for (uint i = 0; i <= 5; ++i) {
                    cout << ds + "__" + kernel + "_" + to_string(i) << endl;
                    GramMatrix gm;

                    if (i == 5) {
                        high_resolution_clock::time_point t1 = high_resolution_clock::now();
                        gm = wl.compute_gram_matrix(i, use_labels, "localp", true, true);
                        high_resolution_clock::time_point t2 = high_resolution_clock::now();
                        auto duration = duration_cast<seconds>(t2 - t1).count();
                        cout << duration << endl;
                    } else {
                        gm = wl.compute_gram_matrix(i, use_labels, "localp", true, true);
                    }

                    AuxiliaryMethods::write_libsvm(gm, classes,
                                                   "./svm/GM/EXP/" + ds + "__" + kernel + "_" + to_string(i) +
                                                   ".gram");
                }
            }
        }
    }

    {
        vector<pair<string, bool>> datasets = {make_pair("ENZYMES", true), make_pair("IMDB-BINARY", false),
                                               make_pair("IMDB-MULTI", false), make_pair("PTC_FM", true)};

        for (auto &d: datasets) {
            {
                string ds = std::get<0>(d);
                bool use_labels = std::get<1>(d);

                string kernel = "WL3";
                GraphDatabase gdb = AuxiliaryMethods::read_graph_txt_file(ds);
                gdb.erase(gdb.begin() + 0);
                vector<int> classes = AuxiliaryMethods::read_classes(ds);

                GenerateThree::GenerateThree wl(gdb);
                for (uint i = 0; i <= 5; ++i) {
                    cout << ds + "__" + kernel + "_" + to_string(i) << endl;
                    GramMatrix gm;

                    if (i == 5) {
                        high_resolution_clock::time_point t1 = high_resolution_clock::now();
                        gm = wl.compute_gram_matrix(i, use_labels, "wl", true, true);
                        high_resolution_clock::time_point t2 = high_resolution_clock::now();
                        auto duration = duration_cast<seconds>(t2 - t1).count();
                        cout << duration << endl;
                    } else {
                        gm = wl.compute_gram_matrix(i, use_labels, "wl", true, true);
                    }

                    AuxiliaryMethods::write_libsvm(gm, classes,
                                                   "./svm/GM/EXP/" + ds + "__" + kernel + "_" + to_string(i) +
                                                   ".gram");
                }
            }

            {
                string ds = std::get<0>(d);
                bool use_labels = std::get<1>(d);

                string kernel = "DWL3";
                GraphDatabase gdb = AuxiliaryMethods::read_graph_txt_file(ds);
                gdb.erase(gdb.begin() + 0);
                vector<int> classes = AuxiliaryMethods::read_classes(ds);

                GenerateThree::GenerateThree wl(gdb);
                for (uint i = 0; i <= 5; ++i) {
                    cout << ds + "__" + kernel + "_" + to_string(i) << endl;
                    GramMatrix gm;

                    if (i == 5) {
                        high_resolution_clock::time_point t1 = high_resolution_clock::now();
                        gm = wl.compute_gram_matrix(i, use_labels, "malkin", true, true);
                        high_resolution_clock::time_point t2 = high_resolution_clock::now();
                        auto duration = duration_cast<seconds>(t2 - t1).count();
                        cout << duration << endl;
                    } else {
                        gm = wl.compute_gram_matrix(i, use_labels, "malkin", true, true);
                    }

                    AuxiliaryMethods::write_libsvm(gm, classes,
                                                   "./svm/GM/EXP/" + ds + "__" + kernel + "_" + to_string(i) +
                                                   ".gram");
                }
            }
        }
    }

    // Simple kernel baselines.
    {
        vector<pair<string, bool>> datasets = {make_pair("ENZYMES", true), make_pair("IMDB-BINARY", false),
                                               make_pair("IMDB-MULTI", false), make_pair("NCI1", true),
                                               make_pair("NCI109", true), make_pair("PTC_FM", true),
                                               make_pair("PROTEINS", true), make_pair("REDDIT-BINARY", false)};

        for (auto &d: datasets) {
            {
                string ds = std::get<0>(d);
                bool use_labels = std::get<1>(d);

                string kernel = "WL1";
                GraphDatabase gdb = AuxiliaryMethods::read_graph_txt_file(ds);
                gdb.erase(gdb.begin() + 0);
                vector<int> classes = AuxiliaryMethods::read_classes(ds);

                ColorRefinement::ColorRefinementKernel wl(gdb);
                for (uint i = 0; i <= 5; ++i) {
                    cout << ds + "__" + kernel + "_" + to_string(i) << endl;
                    GramMatrix gm;

                    if (i == 5) {
                        high_resolution_clock::time_point t1 = high_resolution_clock::now();
                        gm = wl.compute_gram_matrix(i, use_labels, false, true, false);
                        high_resolution_clock::time_point t2 = high_resolution_clock::now();
                        auto duration = duration_cast<seconds>(t2 - t1).count();
                        cout << duration << endl;
                    } else {
                        gm = wl.compute_gram_matrix(i, use_labels, false, true, false);
                    }

                    AuxiliaryMethods::write_libsvm(gm, classes,
                                                   "./svm/GM/EXP/" + ds + "__" + kernel + "_" + to_string(i) +
                                                   ".gram");
                }
            }

            {
                string ds = std::get<0>(d);
                bool use_labels = std::get<1>(d);

                string kernel = "WLOA";
                GraphDatabase gdb = AuxiliaryMethods::read_graph_txt_file(ds);
                gdb.erase(gdb.begin() + 0);
                vector<int> classes = AuxiliaryMethods::read_classes(ds);

                ColorRefinement::ColorRefinementKernel wl(gdb);
                for (uint i = 1; i <= 5; ++i) {
                    cout << ds + "__" + kernel + "_" + to_string(i) << endl;
                    GramMatrix gm;

                    if (i == 5) {
                        high_resolution_clock::time_point t1 = high_resolution_clock::now();
                        gm = wl.compute_gram_matrix(i, use_labels, false, true, true);
                        high_resolution_clock::time_point t2 = high_resolution_clock::now();
                        auto duration = duration_cast<seconds>(t2 - t1).count();
                        cout << duration << endl;
                    } else {
                        gm = wl.compute_gram_matrix(i, use_labels, false, true, true);
                    }

                    AuxiliaryMethods::write_libsvm(gm, classes,
                                                   "./svm/GM/EXP/" + ds + "__" + kernel + "_" + to_string(i) +
                                                   ".gram");
                }
            }

            {
                string ds = std::get<0>(d);
                bool use_labels = std::get<1>(d);

                string kernel = "SP";
                GraphDatabase gdb = AuxiliaryMethods::read_graph_txt_file(ds);
                gdb.erase(gdb.begin() + 0);
                vector<int> classes = AuxiliaryMethods::read_classes(ds);

                ShortestPathKernel::ShortestPathKernel sp(gdb);

                cout << ds + "__" + kernel + "_" + to_string(0) << endl;
                GramMatrix gm;

                high_resolution_clock::time_point t1 = high_resolution_clock::now();
                gm = sp.compute_gram_matrix(use_labels, true);
                high_resolution_clock::time_point t2 = high_resolution_clock::now();
                auto duration = duration_cast<seconds>(t2 - t1).count();
                cout << duration << endl;

                AuxiliaryMethods::write_libsvm(gm, classes,
                                               "./svm/GM/EXP/" + ds + "__" + kernel + "_" + to_string(0) + ".gram");
            }

            {
                string ds = std::get<0>(d);
                bool use_labels = std::get<1>(d);

                string kernel = "GR";
                GraphDatabase gdb = AuxiliaryMethods::read_graph_txt_file(ds);
                gdb.erase(gdb.begin() + 0);
                vector<int> classes = AuxiliaryMethods::read_classes(ds);

                GraphletKernel::GraphletKernel sp(gdb);

                cout << ds + "__" + kernel + "_" + to_string(0) << endl;
                GramMatrix gm;

                high_resolution_clock::time_point t1 = high_resolution_clock::now();
                gm = sp.compute_gram_matrix(use_labels, false, true);
                high_resolution_clock::time_point t2 = high_resolution_clock::now();
                auto duration = duration_cast<seconds>(t2 - t1).count();
                cout << duration << endl;

                AuxiliaryMethods::write_libsvm(gm, classes,
                                               "./svm/GM/EXP/" + ds + "__" + kernel + "_" + to_string(0) + ".gram");
            }
        }
    }


    // Larger Datasets.
    vector<string> d = {"Yeast", "YeastH", "UACC257", "UACC257H", "OVCAR-8", "OVCAR-8H"};
    for (auto &ds: d) {
        bool use_labels = true;
        bool use_edge_labels = true;
        {
            string kernel = "LWL2";
            GraphDatabase gdb = AuxiliaryMethods::read_graph_txt_file(ds);
            gdb.erase(gdb.begin() + 0);
            vector<int> classes = AuxiliaryMethods::read_classes(ds);

            GenerateTwo::GenerateTwo wl(gdb);
            for (uint i = 0; i <= 5; ++i) {
                cout << ds + "__" + kernel + "_" + to_string(i) << endl;
                GramMatrix gm;

                if (i == 5) {
                    high_resolution_clock::time_point t1 = high_resolution_clock::now();
                    gm = wl.compute_gram_matrix(i, use_labels, use_edge_labels, "local", true, false);
                    high_resolution_clock::time_point t2 = high_resolution_clock::now();
                    auto duration = duration_cast<seconds>(t2 - t1).count();
                    cout << duration << endl;
                } else {
                    gm = wl.compute_gram_matrix(i, use_labels, use_edge_labels, "local", true, false);
                }

                AuxiliaryMethods::write_sparse_gram_matrix(gm, "./svm/GM/EXPSPARSE/" + ds +
                                                              "__" + kernel + "_" + to_string(i));
            }
        }

        {
            string kernel = "LWLP2";
            GraphDatabase gdb = AuxiliaryMethods::read_graph_txt_file(ds);
            gdb.erase(gdb.begin() + 0);
            vector<int> classes = AuxiliaryMethods::read_classes(ds);

            GenerateTwo::GenerateTwo wl(gdb);
            for (uint i = 0; i <= 5; ++i) {
                cout << i << endl;
                cout << ds + "__" + kernel + "_" + to_string(i) << endl;
                GramMatrix gm;

                if (i == 5) {
                    high_resolution_clock::time_point t1 = high_resolution_clock::now();
                    gm = wl.compute_gram_matrix(i, use_labels, use_edge_labels, "localp", true, false);
                    high_resolution_clock::time_point t2 = high_resolution_clock::now();
                    auto duration = duration_cast<seconds>(t2 - t1).count();
                    cout << duration << endl;
                } else {
                    gm = wl.compute_gram_matrix(i, use_labels, use_edge_labels, "localp", true, false);
                }

                AuxiliaryMethods::write_sparse_gram_matrix(gm, "./svm/GM/EXPSPARSE/" + ds +
//                                                               "__" + kernel + "_" + to_string(i));
            }
        }

        {
            string kernel = "WL";
            GraphDatabase gdb = AuxiliaryMethods::read_graph_txt_file(ds);
            gdb.erase(gdb.begin() + 0);
            vector<int> classes = AuxiliaryMethods::read_classes(ds);

            ColorRefinement::ColorRefinementKernel wl(gdb);
            for (uint i = 0; i <= 5; ++i) {
                cout << ds + "__" + kernel + "_" + to_string(i) << endl;
                GramMatrix gm;

                if (i == 5) {
                    high_resolution_clock::time_point t1 = high_resolution_clock::now();
                    gm = wl.compute_gram_matrix(i, use_labels, use_edge_labels, false, false);
                    high_resolution_clock::time_point t2 = high_resolution_clock::now();
                    auto duration = duration_cast<seconds>(t2 - t1).count();
                    cout << duration << endl;
                } else {
                    gm = wl.compute_gram_matrix(i, use_labels, use_edge_labels, false, false);
                }

                AuxiliaryMethods::write_sparse_gram_matrix(gm, "./svm/GM/EXPSPARSE/" + ds +
                                                               "__" + kernel + "_" + to_string(i));
            }
        }
    }

    return 0;
}
