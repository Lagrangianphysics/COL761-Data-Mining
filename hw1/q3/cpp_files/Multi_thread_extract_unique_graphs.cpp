#include <iostream>
#include <vector>
#include <thread>
#include <mutex>
#include <algorithm>
#include "Graph.h"
#include "read_write_graphs.h"

using namespace std;

mutex cout_mutex; // Mutex for synchronized output

void processGraphs(vector<Graph>& graphs, int start, int end, int thread_id) {
    for (int i = start; i < end; ++i) {
        graphs[i].generate_canonical_label();
    }
}

int main(int argc, char* argv[]) {
    if (argc != 3) {
        cerr << "Usage: " << argv[0] << " <input file full address> <output file full address>" << endl;
        return 1; // Exit with error
    }

    string input_filename = argv[1];
    string output_filename = argv[2];

    cout << "- - - - - Loading Graphs - - - - -" << endl;
    vector<Graph> graphs = readGraphs(input_filename);

    cout << "- - - - - Generating Canonical Labels (Multi-threaded) - - - - -" << endl;

    int N = graphs.size();
    int num_threads = thread::hardware_concurrency(); // Get number of available cores
    vector<thread> threads;

    int chunk_size = (N + num_threads - 1) / num_threads; // Divide work among threads

    for (int i = 0; i < num_threads; ++i) {
        int start = i * chunk_size;
        int end = min(start + chunk_size, N);
        if (start < end) {
            threads.emplace_back(processGraphs, ref(graphs), start, end, i);
        }
    }

    for (auto& t : threads) {
        t.join(); // Ensure all threads complete
    }

    cout << endl;

    cout << "- - - - - Extracting Unique Graphs - - - - -" << endl;

    auto cmp = [](const Graph& a, const Graph& b) {
        return (a.canonical_label <= b.canonical_label);
    };
    sort(graphs.begin(), graphs.end(), cmp);

    vector<Graph> final_graphs;
    vector<int> prev_label;

    for (const auto& graph : graphs) {
        if (graph.canonical_label != prev_label) {
            final_graphs.push_back(graph);
            prev_label = graph.canonical_label;
        }
    }

    int N2 = final_graphs.size();
    for (int i = 0; i < N2; ++i) {
        final_graphs[i].graph_id = i;
    }

    cout << "- - - - - Saving Unique Graphs - - - - -" << endl;
    printGraphs(final_graphs, output_filename);

    return 0;
}
