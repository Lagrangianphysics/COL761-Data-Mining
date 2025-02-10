import sys
from graph_data_structure import *
import pandas as pd
import subprocess
import retworkx as rx
import time
import numpy as np

from argparse import ArgumentParser

thres = 40

graph_data = []
labels = []
graphs_path = ""
labels_path = ""
subgraphs_path = ""
No_graphs = 0

"""
Helper functions for feature selection:
"""

def mutual_info(X, y):
    X = np.array(X)
    y = np.array(y)
    N, D = X.shape
    mi_scores = np.zeros(D)
    for i in range(D):
        feature = X[:, i]
        p_x = np.bincount(feature, minlength=2) / N
        p_y = np.bincount(y, minlength=2) / N
        p_xy = np.zeros((2, 2))
        for a in [0, 1]:
            for b in [0, 1]:
                p_xy[a, b] = np.mean((feature == a) & (y == b))
                if p_xy[a, b] > 0:
                    mi_scores[i] += p_xy[a, b] * np.log(p_xy[a, b] / (p_x[a] * p_y[b]))
    return mi_scores

def chi2_test(X, y):
    X = np.array(X)
    y = np.array(y)
    N, D = X.shape
    chi2_scores = np.zeros(D)
    for i in range(D):
        feature = X[:, i]
        observed = np.array([
            np.sum((feature == 0) & (y == 0)), np.sum((feature == 1) & (y == 0)),
            np.sum((feature == 0) & (y == 1)), np.sum((feature == 1) & (y == 1))
        ]).reshape(2, 2)
        expected = np.outer(observed.sum(axis=1), observed.sum(axis=0)) / N
        with np.errstate(divide='ignore', invalid='ignore'):
            chi2_scores[i] = np.nansum((observed - expected) ** 2 / expected)
    return chi2_scores

def f_classif_test(X, y):
    X = np.array(X)
    y = np.array(y)
    N, D = X.shape
    f_scores = np.zeros(D)
    mean_y0 = np.mean(X[y == 0], axis=0)
    mean_y1 = np.mean(X[y == 1], axis=0)
    var_y0 = np.var(X[y == 0], axis=0, ddof=1)
    var_y1 = np.var(X[y == 1], axis=0, ddof=1)
    with np.errstate(divide='ignore', invalid='ignore'):
        f_scores = ((mean_y0 - mean_y1) ** 2) / (var_y0 / (y == 0).sum() + var_y1 / (y == 1).sum())
    return np.nan_to_num(f_scores)


def gini_index_selection(X, y):
    X = np.array(X)
    y = np.array(y)
    N, D = X.shape
    gini_scores = np.zeros(D)
    for i in range(D):
        feature = X[:, i]
        p_0 = np.mean(feature == 0)
        p_1 = np.mean(feature == 1)
        gini_0 = 1 - sum((np.mean(y[feature == 0] == c) ** 2 for c in [0, 1]))
        gini_1 = 1 - sum((np.mean(y[feature == 1] == c) ** 2 for c in [0, 1]))
        gini_scores[i] = p_0 * gini_0 + p_1 * gini_1
    return 1 - gini_scores  # Lower Gini Index is better



def jaccard_similarity(set1, set2):
    """Compute Jaccard Similarity between two binary feature vectors."""
    return len(set1 & set2) / len(set1 | set2) if set1 | set2 else 0


def filter_redundant_features(X, threshold=0.8):
    N, D = X.shape
    selected_features = []
    for i in range(D):
        feature_i = set(np.where(X[:, i] == 1)[0])  # Indices where feature is present
        is_redundant = any(jaccard_similarity(feature_i, set(np.where(X[:, j] == 1)[0])) > threshold for j in selected_features)
        if not is_redundant:
            selected_features.append(i)
    return selected_features


def select_k_best(X, y, feature_indices, k=100):
    X_filtered = X[:, feature_indices]  # Work only with filtered features
    mi_scores = mutual_info(X_filtered, y)
    chi2_scores = chi2_test(X_filtered, y)
    f_scores = f_classif_test(X_filtered, y)
    gini_scores = gini_index_selection(X_filtered, y)
    
    mi_scores = mi_scores / np.max(mi_scores)
    chi2_scores = chi2_scores / np.max(chi2_scores)
    f_scores = f_scores / np.max(f_scores)
    gini_scores = gini_scores / np.max(gini_scores)
    
    mi_coef = 0.25
    chi_coef = 0.15
    f_coef = 0.55
    gini_coef = 0.05

    combined_scores = (mi_coef*mi_scores) + (chi_coef*chi2_scores) + (f_coef*f_scores) + (gini_coef*gini_scores)

    top_k_indices = np.argsort(combined_scores)[-k:]

    # Map back to original feature indices
    return [feature_indices[i] for i in top_k_indices]


def top_features_idx(occ_list, labels):
    occ_list = np.array(occ_list)
    labels = np.array(labels)
    N, D = occ_list.shape

    if D < 100:
        return list(range(D))

    filtered_features = filter_redundant_features(occ_list, threshold=0.8)
    if len(filtered_features) < 100:
        return filtered_features
    
    top_features = select_k_best(occ_list, labels, filtered_features, k=100)
    
    return top_features


# def top_features_idx(occ_list, labels):
#     occ_list = np.array(occ_list)
#     labels = np.array(labels)
#     N, D = occ_list.shape
    
#     # If the number of features is less than 100, return all feature indices as a list
#     if D < 100:
#         return list(range(D))
    
#     # Compute Mutual Information scores
#     mi_scores = mutual_info(occ_list, labels)
    
#     # Compute Chi-Square scores
#     chi2_scores = chi2_test(occ_list, labels)
    
#     # Compute ANOVA F-Test scores
#     f_scores = f_classif_test(occ_list, labels)
    
#     # Normalize scores to bring them to a common scale
#     mi_scores = mi_scores / np.max(mi_scores)
#     chi2_scores = chi2_scores / np.max(chi2_scores)
#     f_scores = f_scores / np.max(f_scores)
    
#     # Combine scores using a simple sum (or weighted sum if necessary)
#     combined_scores = mi_scores + chi2_scores + f_scores
    
#     # Select top 100 features based on combined scores
#     top_100_indices = np.argsort(combined_scores)[-100:]
    
#     return top_100_indices.tolist()



# # Gini scores
# def top_features_idx(occ_list, labels):
#     N, D = occ_list.shape
#     if D < 100:
#         return list(range(D))
#     gini_scores = gini_index_selection(occ_list, labels)
#     top_100_indices = np.argsort(gini_scores)[-100:]
#     return top_100_indices.tolist()



# # Mutual Information scores
# def top_features_idx(occ_list, labels):
#     N, D = occ_list.shape
#     if D < 100:
#         return list(range(D))
#     # mi_scores = mutual_info_classif(occ_list, labels, discrete_features=True)
#     mi_scores = mutual_info(occ_list, labels)
#     top_100_indices = np.argsort(mi_scores)[-100:]
#     return top_100_indices.tolist()


# # Chi-Square scores
# def top_features_idx(occ_list, labels):
#     N, D = occ_list.shape
#     if D < 100:
#         return list(range(D))
#     # chi2_scores, _ = chi2(occ_list, labels)
#     chi2_scores = chi2_test(occ_list, labels)
#     top_100_indices = np.argsort(chi2_scores)[-100:]
#     return top_100_indices.tolist()


# # ANOVA F-Test scores
# def top_features_idx(occ_list, labels):
#     N, D = occ_list.shape
#     if D < 100:
#         return list(range(D))
#     # f_scores, _ = f_classif(occ_list, labels)
#     f_scores = f_classif_test(occ_list, labels)
#     top_100_indices = np.argsort(f_scores)[-100:]
#     return top_100_indices.tolist()



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

    # graphs_path = "q3_datasets/NCI-H23/train_graphs.txt"
    # labels_path = "q3_datasets/NCI-H23/train_labels.txt"
    # discriminative_subgraphs_file = "NCI-H23/final_graphs.txt"

    print("- - - - - Running Python Script for Identifing Features - - - - - ")

    print("- - - - - Loading Graphs - - - - -")

    graph_data = read_normal_graphs_from_file(graphs_path)
    No_graphs = len(graph_data)

    with open(labels_path, 'r') as file:
        labels = list(map(int, file.read().splitlines()))

    df = pd.DataFrame({'graph': graph_data, 'label': labels})

    df_0 = df[df['label'] == 0].reset_index(drop=True)
    df_1 = df[df['label'] == 1].reset_index(drop=True)

    graphs_0 = df_0['graph'].tolist()
    graphs_1 = df_1['graph'].tolist()

    print("- - - - - Getting ready to apply Gaston on [0] Labels - - - - -")
    gaston_0_graphs_file = "gaston_0_graphs.txt"
    gaston_0_freq_file = "gaston_0_freq.txt"
    with open(gaston_0_graphs_file,'w') as file:
        file.write("")
    for G in graphs_0 :
        G.append_to_file_gaston(gaston_0_graphs_file)

    print("- - - - - getting freq subgraphs of [0] label graphs - - - - -")
    min_sup = (len(graphs_0)*thres)//100
    subprocess.run(["./gaston", str(min_sup), gaston_0_graphs_file, gaston_0_freq_file] , capture_output=True , text=True , check=True)

    print("- - - - - Getting ready to apply Gaston on [1] Labels - - - - -")
    gaston_1_graphs_file = "gaston_1_graphs.txt"
    gaston_1_freq_file = "gaston_1_freq.txt"
    with open(gaston_1_graphs_file,'w') as file:
        file.write("")
    for G in graphs_1 :
        G.append_to_file_gaston(gaston_1_graphs_file)

    print("- - - - - getting freq subgraphs of [1] label graphs - - - - -")
    min_sup = (len(graphs_1)*thres)//100
    subprocess.run(["./gaston", str(min_sup), gaston_1_graphs_file, gaston_1_freq_file] , capture_output=True , text=True , check=True)

    print("- - - - - loading freq subgraphs - - - - -")

    freq_0 = read_freq_graphs_from_file(gaston_0_freq_file)
    freq_1 = read_freq_graphs_from_file(gaston_1_freq_file , len(freq_0)) # len(freq_0) for distinct graph_id
    freq_graphs = freq_0+freq_1

    expected_freq_graphs_file = "freq_graphs.txt"
    unique_freq_graphs_file = "final_freq_graphs.txt"


    print("- - - - - ################################################################ - - - - -")
    with open(expected_freq_graphs_file,'w') as file:
        file.write("")
    for G in freq_graphs :
        G.append_to_file_gaston("freq_graphs.txt")
    print(f"initial number of features = {len(freq_graphs)}")
    print("- - - - - deleting repeaded occurances in [0] and [1] label freq subgraphs - - - - -")
    subprocess.run(["./Multi_thread_extract_unique_graphs.out" , expected_freq_graphs_file , unique_freq_graphs_file])
    print("- - - - - ################################################################ - - - - -")

    # # THIS LINES IF NOT EXICUTE C++ FILE:
    # with open(expected_freq_graphs_file,'w') as file:
    #     file.write("")
    # for G in freq_graphs :
    #     G.append_to_file_normal("freq_graphs.txt")
    # unique_freq_graphs_file = expected_freq_graphs_file

    print("- - - - - loading Unique freq subgraphs - - - - -")
    freq_graphs = read_normal_graphs_from_file(unique_freq_graphs_file)
    print(f"final number of features = {len(freq_graphs)}")

    
    print("- - - - - Generating Occurance list - - - - -")
    no_of_freq_graphs = len(freq_graphs)
    occurance_list = [[] for _ in range(no_of_freq_graphs)]

    for i in range(no_of_freq_graphs):
        if(i%10 == 0): print("= ",end="")
        
        for j in range(No_graphs):
            flag = rx.is_subgraph_isomorphic(
                graph_data[j].graph
                , freq_graphs[i].graph
                , node_matcher=lambda n1, n2: n1["label"] == n2["label"]
                , edge_matcher=lambda e1, e2: e1 == e2
            )
            if flag:
                occurance_list[i].append(1)
            else:
                occurance_list[i].append(0)
    print("")


    # with open("occurance_list.txt", "w") as file:
    #     for row in occurance_list:
    #         file.write(" ".join(map(str, row)) + "\n")


    print("- - - - - Feature Selection - - - - -")

    top_100_features = top_features_idx(np.array(occurance_list).T, np.array(labels))

    # from sklearn.ensemble import RandomForestClassifier
    # import numpy as np

    # X = np.array(occurance_list).T  # Shape (N, D)
    # y = np.array(labels)

    # # Train a Random Forest model
    # rf = RandomForestClassifier(n_estimators=100, random_state=42)
    # rf.fit(X, y)

    # # Get feature importances
    # feature_importances = rf.feature_importances_

    # # Select top 100 features
    # top_100_features = np.argsort(feature_importances)[-100:]
    # X_selected = X[:, top_100_features]

    print("- - - - - Saving Final Graphs - - - - -")

    with open(discriminative_subgraphs_file, 'w') as file: file.write("")
    for i in top_100_features:
        freq_graphs[i].append_to_file_normal(discriminative_subgraphs_file)




    end_time = time.time()
    total_time_taken = end_time - start_time
    print(f"total time taken = [{total_time_taken}]")
    