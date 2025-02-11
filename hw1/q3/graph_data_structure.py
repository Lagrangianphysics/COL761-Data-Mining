"""
This file contains datastructure of graph that we are using and related function

<GraphData>
    Member Variables:
        <graph>
            undirected graph data structure provided by networkx
        <graph_id>
            0,1,2,...
    
    Methods :

        append_to_file_normal(self, filename):
            print G like this in filename:
            *Note this append this text in the existing file*

            #
            v 0 0
            v 1 4
            v 2 0
            e 0 1 0
            e 0 2 1
            e 0 3 0
        
        append_to_file_gaston(self, filename):
            print G like this in filename:
                *Note this append this text in the existing file*

                t # 7
                v 0 0
                v 1 4
                v 2 0
                e 0 1 0
                e 0 2 1
                e 0 3 0

Other functions:

read_normal_graphs_from_file(filename , idx = 0):
    reads graphs from filename and return a list

"""


import sys
import retworkx as rx

class GraphData:
    def __init__(self, graph_id):
        self.graph = rx.PyGraph(multigraph=False)
        self.graph_id = graph_id
        self.node_map = {}  # Maps node_id to retworkx internal index
        self.edge_labels = {}  # Maps (u, v) -> label

    def add_node(self, node_id, label=None):
        """Adds a node with an optional label."""
        index = self.graph.add_node(node_id)  # Store as dict
        self.node_map[node_id] = label  # Store mapping

    def add_edge(self, node1, node2, label=None):
        if (node1<node2):
            node1,node2=node2,node1
        self.graph.add_edge(node1, node2, label)  
        self.graph.add_edge(node2, node1, label) 
        self.edge_labels[(node1, node2)] = label
        self.edge_labels[(node2, node1)] = label

    def append_to_file_normal(self, filename):

        with open(filename, "a") as f:
            f.write("#\n")
            
            for node_id in self.graph.nodes():
                label = self.node_map[node_id]
                # label = self.graph[index]["label"] if self.graph[index] else ""
                f.write(f"v {node_id} {label}\n")

            # for (node1, node2), label in self.edge_labels.items():
            #     label = label if label is not None else ""
            #     f.write(f"e {node1} {node2} {label}\n")
            
            for node1, node2 in self.graph.edge_list():
                label = self.edge_labels[(node1, node2)]
                f.write(f"e {node1} {node2} {label}\n")
                    # label = label if label is not None else ""

    def append_to_file_gaston(self, filename):
        
        with open(filename, "a") as f:

            f.write(f"t # {self.graph_id}\n")

            for node_id in self.graph.nodes():
                # print(node_id)
                label = self.node_map[node_id]
                # label = self.graph[index]["label"] if self.graph[index] else ""
                f.write(f"v {node_id} {label}\n")

            # for (node1, node2), label in self.edge_labels.items():
            #     label = label if label is not None else ""
            #     f.write(f"e {node1} {node2} {label}\n")
            
            # print(self.graph.edge_list())

            for node1, node2 in self.graph.edge_list():
                label = self.edge_labels[(node1, node2)]
                f.write(f"e {node1} {node2} {label}\n")
                    # label = label if label is not None else ""


def read_normal_graphs_from_file(file_name , idx = 0):
    G = None
    graph_data = []
    sys.stdin = open(file_name, 'r')

    while True :
        x = None
        try:
            x = input()
        except:
            break

        if(x == "#") :
            G = GraphData(idx)
            graph_data.append(G)
            idx += 1
        elif(x[0] == 'v'):
            v,N,L = x.split()
            N = int(N)
            L = int(L)
            graph_data[-1].add_node(N,L)
        elif(x[0] == 'e'):
            e,S,E,L = x.split()
            S = int(S)
            E = int(E)
            L = int(L)
            graph_data[-1].add_edge(S,E,L)

    sys.stdin = sys.__stdin__
    return graph_data

def read_freq_graphs_from_file(file_name , idx = 0):
    G = None
    graph_data = []
    sys.stdin = open(file_name, 'r')

    while True :
        x = None
        try:
            x = input()
        except:
            break
        
        if(x[0] == '#') :
            G = GraphData(idx)
            graph_data.append(G)
            idx += 1
            trash = input()
        elif(x[0] == 'v'):
            v,N,L = x.split()
            N = int(N)
            L = int(L)
            graph_data[-1].add_node(N,L)
        elif(x[0] == 'e'):
            e,S,E,L = x.split()
            S = int(S)
            E = int(E)
            L = int(L)
            graph_data[-1].add_edge(S,E,L)
            # if (S<E):
        

    sys.stdin = sys.__stdin__
    return graph_data


def edge_matcher_helper(e1,e2) :
    # print(f"Comparing edges: {e1} vs {e2}")
    if (e1 == e2):
        print(e1,e2,e1==e2)
    else:
        print(e1,e2,e1==e2)
    return (e1==e2)






# # Example Usage:
# G1 = GraphData(1)
# G1.add_node(1, label='A')
# G1.add_node(2, label='B')
# G1.add_node(3, label='C')
# G1.add_edge(1, 2, label='X')
# G1.add_edge(2, 3, label='Y')

# G2 = GraphData(2)
# G2.add_node(1, label='A')
# G2.add_node(2, label='B')
# G2.add_edge(1, 2, label='X')

# # print(G1)

# print(G1.is_subgraph_isomorphic(G2))  # Should return True
