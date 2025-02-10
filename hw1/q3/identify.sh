#!/bin/bash

# bash identify.sh <path_train_graphs> <path_train_labels> <path_discriminative_subgraphs> 
# $1 = path_train_graphs
# $2 = path_train_labels
# $3 = path_discriminative_subgraphs

# Input parameters
TRAIN_GRAPHS_PATH=$1
TRAIN_LABELS_PATH=$2
DISCRIMINATIVE_SUBGRAPHS_PATH=$3

# Run the Python script to convert the graphs into feature vectors
python3 identify_features.py --graphs $TRAIN_GRAPHS_PATH --labels $TRAIN_LABELS_PATH --subgraphs $DISCRIMINATIVE_SUBGRAPHS_PATH
