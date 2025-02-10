#include <iostream>
#include <vector>
#include "Graph.h"
#include "read_write_graphs.h"
using namespace std ;

int main(int argc, char* argv[]) {

    if (argc != 3) {
        cerr << "Usage: " << argv[0] << " <input file full address> <output file full address>" << endl;
        return 1; // Exit with error
    }

    string input_filename = argv[1];
    string output_filename = argv[2];
    
    // string input_filename = "C:/Users/subho/OneDrive - IIT Delhi/Subhojit/SEM 6/COL761 data mining/Homework/HW1/Q3/freq_graphs.txt" ;
    // string output_filename = "C:/Users/subho/OneDrive - IIT Delhi/Subhojit/SEM 6/COL761 data mining/Homework/HW1/Q3/final_freq_graphs.txt" ;

    cout << "- - - - - Loading Graphs - - - - -" << endl ;
    vector<Graph> graphs = readGraphs(input_filename) ;
    
    cout << "- - - - - Generting Canonical Labels - - - - -" << endl ;
    
    int N = graphs.size() ;
    
    for(int i = 0 ; i < N ; i+=10) {
        cout << "= " ;
    } cout << endl ;

    for(int i = 0 ; i < N ; i++) {
        if(i%10 == 0) {cout << "= " ;}
        graphs[i].generate_canonical_label() ;
    }
    cout << endl ;

    cout << "- - - - - Extracting Unique Graphs - - - - -" << endl ;
    
    auto cmp = [](const Graph &a, const Graph &b) {
        return (a.canonical_label <= b.canonical_label) ;
    };
    sort(graphs.begin(), graphs.end(), cmp) ;

    vector<Graph> final_graphs(0) ;
    vector<int> prev_label = vector<int>(0) ;
    for(int i = 0 ; i < N ; i++) {
        if(graphs[i].canonical_label != prev_label) {
            final_graphs.push_back(graphs[i]) ;
            prev_label = graphs[i].canonical_label ;
        }
    }
    int N2 = final_graphs.size() ;
    for(int i = 0 ; i < N2 ; i++) {
        final_graphs[i].graph_id = i ;
    }

    cout << "- - - - - Saving Unique Graphs - - - - -" << endl ;
    printGraphs(final_graphs , output_filename) ;

    return 0 ;
}