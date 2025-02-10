#ifndef READ_WRITE_GRAPHs_H
#define READ_WRITE_GRAPHs_H

#include <iostream>
#include <fstream>
#include <vector>
#include "Graph.h"
using namespace std ;

vector<Graph> readGraphs(string filename) {
    ifstream file(filename);
    vector<Graph> graphs(0) ;
    char x ;
    int idx = -1 ;
    while(file >> x) {
        if(x == 't') {
            int id ;
            file >> x >> id ; // # id
            graphs.push_back(Graph(id)) ;
            idx++ ;
        } else if(x == 'v') {
            int node_id , label ;
            file >> node_id >> label ;
            graphs[idx].addNode(node_id , label) ;
        } else if(x == 'e') {
            int u_id , v_id , label ;
            file >> u_id >> v_id >> label ;
            graphs[idx].addEdge(u_id , v_id , label) ;
        }
    }
    file.close() ;
    return graphs ;
}

void printGraphs(vector<Graph> &graphs , string filename) {
    ofstream outFile(filename);
    for (const Graph& graph : graphs) {
        outFile << "#"<< endl;
        
        for (const Node& node : graph.nodes) {
            outFile << "v " << node.id << " " << node.label() << endl;
        }

        for (const Edge& edge : graph.edges) {
            outFile << "e " << edge.u_id << " " << edge.v_id << " " << edge.label << endl;
        }
    }
}

#endif