from graph_data_structure import *
import retworkx as rx
import time
import numpy as np
from argparse import ArgumentParser


if __name__ == "__main__":
    
    start_time = time.time()

    # Argument parsing
    parser = ArgumentParser()
    parser.add_argument('--graphs', type=str, required=True, help='Path to testing graphs')
    parser.add_argument('--subgraphs', type=str, required=True, help='Path to discriminative subgraphs')
    parser.add_argument('--output', type=str, required=True, help='Path to store feature vector file')
    args = parser.parse_args()
    
    # Store paths in separate variables
    Graphs_path = args.graphs
    Discriminative_subgraphs_path = args.subgraphs
    Feature_path = args.output

    print("- - - - - Running Python Script for Converiting graphs into Features - - - - - ")
    print(f"Graphs from {Graphs_path}")

    print("- - - - - Loading Graphs - - - - -")
    graphs = read_normal_graphs_from_file(Graphs_path)
    discriminative_subgraphs = read_normal_graphs_from_file(Discriminative_subgraphs_path)

    vec = []
    N = len(graphs)
    print("- - - - - converting into feature vector - - - - -")

    for i in range(N):
        G = graphs[i]
        vec.append([])
        for subgraph in discriminative_subgraphs:
            flag = rx.is_subgraph_isomorphic(
                G.graph
                , subgraph.graph
                , node_matcher=lambda n1, n2: G.node_map[n1]==subgraph.node_map[n2]
                , edge_matcher=lambda e1, e2: e1 == e2
                , id_order=False
                , induced=False
            )
            if flag:
                vec[-1].append(1)
            else: 
                vec[-1].append(0)

    print("- - - - - Saving feature vector - - - - -")
    vec = np.array(vec)
    np.save(Feature_path, vec)



    end_time = time.time()
    total_time_taken = end_time - start_time
    print(f"total time taken = [{total_time_taken}]")