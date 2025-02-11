from graph_data_structure import *
import subprocess
import retworkx as rx
import time
import numpy as np
from argparse import ArgumentParser
from sklearn.feature_selection import mutual_info_classif, chi2, f_classif

thres = 30

graph_data = []
labels = []
graphs_path = ""
labels_path = ""
subgraphs_path = ""
No_graphs = 0

"""
Helper functions for feature selection:
"""



def jaccard_similarity(set1, set2):
    """Compute Jaccard Similarity between two binary feature vectors."""
    return len(set1 & set2) / len(set1 | set2) if set1 | set2 else 0


def top_features_idx(occ_list, labels):
    print("shape of list",occ_list.shape)
    # occ_list = np.array(occ_list)
    labels = np.array(labels)
    N, D = occ_list.shape

    if D < 100:
        return list(range(D))

    mi_scores = mutual_info_classif(occ_list, labels)
    chi2_scores, _ = chi2(occ_list, labels)
    feature_counts = np.sum(occ_list > 0, axis=0)
    
    mi_scores = (mi_scores - np.mean(mi_scores)) / np.var(mi_scores)
    chi2_scores = (chi2_scores - np.mean(chi2_scores)) / np.var(chi2_scores)
    feature_counts = (feature_counts - np.mean(feature_counts)) / np.var(feature_counts)

    mi_coef = 0.33
    chi_coef = 0.33
    count_coef = 0.33

    combined_scores = (mi_coef*mi_scores) + (chi_coef*chi2_scores) + (count_coef * feature_counts)

    sorted_indices = np.argsort(-combined_scores)
    selected_indices = set()

    occurrence_sets = [set(np.where(occ_list[:, i] > 0)[0]) for i in range(D)]

    for idx in sorted_indices:
        if len(selected_indices) >= 100:
            break
        if all(jaccard_similarity(occurrence_sets[idx], occurrence_sets[prev_idx]) < 0.8 for prev_idx in selected_indices):
            selected_indices.add(idx)

    return list(selected_indices)


if __name__ == "__main__":
    
    start_time = time.time()

    # Argument parsing
    parser = ArgumentParser()
    parser.add_argument('--graphs', type=str, required=True, 
                        help='Path to training graphs')
    parser.add_argument('--labels', type=str, required=True, 
                        help='Path to training labels')
    parser.add_argument('--subgraphs', type=str, required=True, 
                        help='Path to discriminative subgraphs')
    args = parser.parse_args()
    
    # Store paths in separate variables
    graphs_path = args.graphs
    labels_path = args.labels
    discriminative_subgraphs_file = args.subgraphs

    print("- - - - - Running Python Script for Identifing Features - - - - - ")

    print("- - - - - Loading Graphs - - - - -")

    graph_data = read_normal_graphs_from_file(graphs_path)
    No_graphs = len(graph_data)
    
    gaston_file = "gaston_file.txt"
    unique_freq_graphs_file = "final_freq_graphs.txt"

    with open(gaston_file , 'w') as file:
        file.write("")
    for G in graph_data :
        G.append_to_file_gaston(gaston_file)

    min_sup = (len(graph_data)*thres)//100
    print(f"total graphs = [{len(graph_data)}] , minsup = {min_sup}")
    subprocess.run(["./gaston", str(min_sup), gaston_file, unique_freq_graphs_file] , capture_output=True , text=True , check=True)

    with open(labels_path, 'r') as file:
        labels = list(map(int, file.read().splitlines()))

    # df = pd.DataFrame({'graph': graph_data, 'label': labels})

    # df_0 = df[df['label'] == 0].reset_index(drop=True)
    # df_1 = df[df['label'] == 1].reset_index(drop=True)

    # graphs_0 = df_0['graph'].tolist()
    # graphs_1 = df_1['graph'].tolist()

    # print(df_0,df_1)
    # print("- - - - - Getting ready to apply Gaston on [0] Labels - - - - -")
    # gaston_0_graphs_file = "gaston_0_graphs.txt"
    # gaston_0_freq_file = "gaston_0_freq.txt"
    # with open(gaston_0_graphs_file,'w') as file:
    #     file.write("")
    # for G in graphs_0 :
    #     G.append_to_file_gaston(gaston_0_graphs_file)

    # print("- - - - - getting freq subgraphs of [0] label graphs - - - - -")
    # min_sup = (len(graphs_0)*thres)//100
    # print(f"total graphs = [{len(graphs_0)}] , minsup = {min_sup}")
    # subprocess.run(["./gaston", str(min_sup), gaston_0_graphs_file, gaston_0_freq_file] , capture_output=True , text=True , check=True)

    # print("- - - - - Getting ready to apply Gaston on [1] Labels - - - - -")
    # gaston_1_graphs_file = "gaston_1_graphs.txt"
    # gaston_1_freq_file = "gaston_1_freq.txt"
    # with open(gaston_1_graphs_file,'w') as file:
    #     file.write("")
    # for G in graphs_1 :
    #     G.append_to_file_gaston(gaston_1_graphs_file)

    # print("- - - - - getting freq subgraphs of [1] label graphs - - - - -")
    # min_sup = (len(graphs_1)*thres)//100
    # print(f"total graphs = [{len(graphs_1)}] , minsup = {min_sup}")
    # subprocess.run(["./gaston", str(min_sup), gaston_1_graphs_file, gaston_1_freq_file] , capture_output=True , text=True , check=True)

    # print("- - - - - loading freq subgraphs - - - - -")

    # freq_0 = read_freq_graphs_from_file(gaston_0_freq_file)
    # freq_1 = read_freq_graphs_from_file(gaston_1_freq_file , len(freq_0)) # len(freq_0) for distinct graph_id
    # freq_graphs = freq_0+freq_1

    # expected_freq_graphs_file = "freq_graphs.txt"
    # unique_freq_graphs_file = "final_freq_graphs.txt"


    # print("- - - - - ################################################################ - - - - -")
    # with open(expected_freq_graphs_file,'w') as file:
    #     file.write("")
    # for G in freq_graphs :
    #     G.append_to_file_gaston("freq_graphs.txt")
    # print(f"initial number of features = {len(freq_graphs)}")
    # print("- - - - - deleting repeaded occurances in [0] and [1] label freq subgraphs - - - - -")
    # subprocess.run(["./Multi_thread_extract_unique_graphs.out" , expected_freq_graphs_file , unique_freq_graphs_file])
    # print("- - - - - ################################################################ - - - - -")

    # # THIS LINES IF NOT EXICUTE C++ FILE:
    # with open(expected_freq_graphs_file,'w') as file:
    #     file.write("")
    # for G in freq_graphs :
    #     G.append_to_file_normal("freq_graphs.txt")
    # unique_freq_graphs_file = expected_freq_graphs_file

    print("- - - - - loading Unique freq subgraphs - - - - -")
    # freq_graphs = read_normal_graphs_from_file(unique_freq_graphs_file)
    freq_graphs = read_freq_graphs_from_file(unique_freq_graphs_file)
    No_freq_graphs = len(freq_graphs)
    print(f"final number of features = {No_freq_graphs}")

    while ((No_freq_graphs * No_graphs) > 4*10**6) : 
        thres += 5
        print(f"too many subgraphs minned, re-running gaston with thres = {thres}")
        min_sup = (len(graph_data)*thres)//100
        print(f"total graphs = [{len(graph_data)}] , minsup = {min_sup}")
        subprocess.run(["./gaston", str(min_sup), gaston_file, unique_freq_graphs_file] , capture_output=True , text=True , check=True)
        print("- - - - - loading Unique freq subgraphs - - - - -")
        freq_graphs = read_freq_graphs_from_file(unique_freq_graphs_file)
        No_freq_graphs = len(freq_graphs)
        print(f"final number of features = {No_freq_graphs}")


    print("- - - - - Generating Occurance list - - - - -")
    no_of_freq_graphs = len(freq_graphs)
    occurance_list = [[] for _ in range(no_of_freq_graphs)]

    for i in range(no_of_freq_graphs):
        if(i%10 == 0): print("= ",end="")
        
        for j in range(No_graphs):
            flag = rx.is_subgraph_isomorphic(
                graph_data[j].graph
                , freq_graphs[i].graph
                , node_matcher=lambda n1, n2: graph_data[j].node_map[n1]==freq_graphs[i].node_map[n2]
                , edge_matcher=lambda e1, e2: e1 == e2
                , id_order=False
                , induced=False
            )
            if flag:
                occurance_list[i].append(1)
            else:
                occurance_list[i].append(0)
    print("")


    print("- - - - - Feature Selection - - - - -")
    top_100_features = top_features_idx(np.array(occurance_list).T, np.array(labels))

    print("- - - - - Saving Final Graphs - - - - -")

    with open(discriminative_subgraphs_file, 'w') as file: file.write("")
    for i in top_100_features:
        freq_graphs[i].append_to_file_normal(discriminative_subgraphs_file)



    end_time = time.time()
    total_time_taken = end_time - start_time
    print(f"total time taken = [{total_time_taken}]")
    