#include <iostream>
#include <fstream>
#include <string>
#include <omp.h>
#include <algorithm>
#include <chrono>
#include <numeric>
#include <queue>
#include <random>
#include <sstream>
#include <tuple>
#include <vector>
#include <ctime>
#include "my_utils.h"

using namespace std;
using Graph = vector<vector<pair<int, double>>>;

int main(int argc, char* argv[]) {

    clock_t START_TIME = clock();

    // Check if the correct number of arguments is provided
    if (argc != 5) {
        cerr << "Usage: ./main <absolute_path_to_graph> <absolute_output_file_path> <k> <#_random_instances>" << endl;
        return 1;
    }
    // Parse command-line arguments
    string graphPath = argv[1];
    string outputPath = argv[2];
    int k = stoi(argv[3]);
    int random_instances = stoi(argv[4]);

    cout << "==========================================\n";
    cout << "            COL761 Homework 2             \n";
    cout << "        [ Influence Maximization ]        \n";
    cout << "------------------------------------------\n";
    cout << "           by Team MathMiners             \n";
    cout << "==========================================\n";

    cout << endl << "Input Successful, parameters are: " << endl 
        << "absolute_path_to_graph: <" << graphPath << ">" << endl 
        << "absolute_output_file_path: <" << outputPath << ">" << endl 
        << "k: " << k << endl 
        << "#_random_instances: " << random_instances << endl << endl ;

    int iterations = 1000 ;

    cout << "Loading Graph from <" << graphPath << ">..." << endl;

    string Index_graph_path = "index_graph.txt" ;

    // CONVERT TO 0 BASED INDEXING
    map<int , int> new_to_org = convert_to_zero_index(graphPath , Index_graph_path) ;

    int numNodes = 0;
    Graph G = loadGraph(Index_graph_path , numNodes);
    cout << "Graph loaded with " << numNodes << " nodes" << endl;



    // MINE TOP k

    cout << "Mining k seeds by CLEF..." << endl ;
    vector<int> seeds = CELF(G, k, iterations , outputPath , new_to_org) ;
    double spread = independentCascade(G, seeds , 10000) ;
    cout << "Spread of k seeds = " << spread << endl ;

    // Convert to original indexing
    for(int &s:seeds) {
        s = new_to_org[s] ;
    }

    cout << "Selected seed nodes: \n" ;
    for (int s : seeds) cout << s << " , " ;
    cout << endl ;
    ofstream outFile(outputPath);
    for(int seed : seeds) {
        outFile << seed << "\n"; // Write each seed on a new line
    }
    outFile.close() ;

    cout << "TOTAL RUN TIME : [" << ((double)(clock() - START_TIME) / CLOCKS_PER_SEC) << "]" << endl ;
    return 0;
}
