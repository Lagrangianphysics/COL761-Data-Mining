#ifndef GRAPH_H
#define GRAPH_H

#include <iostream>
#include <map>
#include <vector>
#include <algorithm>
#include "Node.h"
#include "Edge.h"
using namespace std ;

class Graph {
public:
    
    int graph_id = -1 ;
    vector<Node> nodes ;
    vector<Edge> edges ;
    vector<Node> id_to_node ;
    /*
    if there is edge between id i and j then 
    adjacency_matrix[i][j] = adjacency_matrix[j][i] = label of edge
    else
    adjacency_matrix[i][j] = adjacency_matrix[j][i] = -1
    */
    vector<vector<int>> adjacency_matrix ;
    vector<int> canonical_label ;
    
    Graph() {}
    
    Graph(int _graph_id_) {
        graph_id = _graph_id_ ;
        canonical_label = vector<int>(0) ;
        edges = vector<Edge>(0) ;
        nodes = vector<Node>(0) ;
    }

    /*
    NOTE: Node ids should be strictly 0 based indexed
    Don't add same Node twice
    max Node id = (total number of nodes - 1) Should be strictly followed
    */
    void addNode(int id, int label) {
        nodes.push_back(Node(id, label));
    }

    /*
    label should me non-negative integer
    */
    void addEdge(int u_id, int v_id, int label) {
        edges.push_back(Edge(u_id, v_id, label));
    }
    
    void generate_id_to_node() {
        if(id_to_node.size() != 0) {
            return ;
        }
        int n = nodes.size() ;
        id_to_node = vector<Node>(n) ;
        for(int i = 0 ; i < n ; i++) {
            id_to_node[nodes[i].id] = nodes[i] ;
        }
    }
    
    // also set node degrees
    void generate_adjacency_matrix() {
        if(adjacency_matrix.size() != 0) {
            return ;
        }
        int n = nodes.size() ;
        adjacency_matrix = vector<vector<int>>(n,vector<int>(n , -1)) ;
        for(const Edge &edg : edges) {
            adjacency_matrix[edg.u_id][edg.v_id] = edg.label ;
            adjacency_matrix[edg.v_id][edg.u_id] = edg.label ;
        }

        generate_id_to_node() ;
        for(int i = 0 ; i < n ; i++) {
            int deg = 0 ;
            for(int j = 0 ; j < n ; j++) {
                if(adjacency_matrix[i][j] != -1) {
                    deg++ ;
                }
            }
            id_to_node[i].invariant.degree = deg ;
        }
    }
    
    /*
    algo to generate all possible labels:
    1. sort all nodes based on degree 
    (this is the partial ordering)
    each partition have same vertex invariants
    2. Create a corresponding graph G
    3. for each node in nodes vector, in G make a directed edge 
    from that node id to all other nodes in the partition preceding that node's partition
    4. Each toposort of G is a possible label
    5. Generate all possible topological sort of G
    6. for each toposort generate corresponding adjacency matrix and coresponding label
    7. select the smallest label
    */


    /*
    ordering of node id is given, return corresponging label
    */
    vector<int> label_from_ordering(vector<int> &ordering) {
        int n = nodes.size() ;
        vector<vector<int>> adj(n,vector<int>(n)) ;
        for(int i = 0 ; i < n ; i++) {
            for(int j = 0 ; j < n ; j++) {
                int node_id_1 = ordering[i] ;
                int node_id_2 = ordering[j] ;
                adj[i][j] = adjacency_matrix[node_id_1][node_id_2] ;
                adj[j][i] = adj[i][j] ;
            }
        }
        vector<int> label ;
        for(int i = 0 ; i < n ; i++) {
            label.push_back(id_to_node[ordering[i]].label()) ;
        }
        for(int j = 1 ; j < n ; j++) {
            for(int i = 0 ; i < j ; i++) {
                label.push_back(adj[i][j]) ;
            }
        }
        return label ;
    }

    void topoSort(vector<vector<int>> &adj , vector<int> &indegree , vector<bool> &visited , vector<int> &cur_res , vector<int> &min_label) {
        int n = nodes.size() ;
        bool flag = false ;

        for(int node = 0 ; node < n ; node++) {
            if(indegree[node] == 0 && !visited[node]) {
                for(int nbd : adj[node]) {
                    indegree[nbd]-- ;
                }
                cur_res.push_back(node) ;
                visited[node] = true ;
                topoSort(adj , indegree , visited , cur_res , min_label) ;
                visited[node] = false ;
                cur_res.pop_back() ;
                for(int nbd : adj[node]) {
                    indegree[nbd]++ ;
                }
                flag = true ;
            }
        }

        if(!flag) {
            vector<int> cur_label = label_from_ordering(cur_res) ;
            if(cur_label < min_label) {
                min_label = cur_label ;
            }
        }
    }

    void generate_canonical_label() {
        if(canonical_label.size() != 0) {
            return ;
        }

        generate_id_to_node() ;
        generate_adjacency_matrix() ;

        // sort all vertext based on vertex invariant
        auto cmp = [](const Node &a, const Node &b) {
            return (a.invariant <= b.invariant) ;
        };

        sort(nodes.begin(), nodes.end(), cmp) ;

        int n = nodes.size() ;
        vector<vector<int>> adj(n) ;
        vector<int> indegree(n) ;
        int prev_s = 0 ;
        int prev_e = -1 ;

        for(int i = 1 ; i < n ; i++) {
            if(nodes[i].invariant != nodes[i-1].invariant) {
                prev_s = prev_e + 1 ;
                prev_e = i - 1 ;
            }
            for(int j = prev_s ; j <= prev_e ; j++) {
                adj[nodes[i].id].push_back(nodes[j].id) ;
                indegree[nodes[j].id]++ ;
            }
        }

        vector<int> cur_topo ;
        vector<int> mini_label = {(int)1e9} ;
        vector<bool> visited(n,false) ;
        topoSort(adj , indegree , visited , cur_topo , mini_label) ;
        canonical_label = mini_label ;

    }

    vector<int> get_canonical_label() {
        if(canonical_label.size() == 0) {
            generate_canonical_label() ;
        }
        return canonical_label ;
    }

    void printGraph() {
        for (const auto& node : nodes) {
            cout << "Node ID: " << node.id << ", Label: " << node.label() << "\n";
        }
        
        for (const auto& edge : edges) {
            cout << "Edge: " << edge.u_id << " -- " << edge.v_id << ", Label: " << edge.label << "\n";
        }
    }
    
};

#endif