#!/bin/bash

# Usage: bash convert.sh <path_graphs> <path_discriminative_subgraphs> <path_features>
# $1 = path to graphs (train/test graphs)
# $2 = path to discriminative subgraphs
# $3 = path to output feature vectors

# Input parameters
GRAPH_PATH=$1
DISCRIMINATIVE_SUBGRAPHS_PATH=$2
FEATURES_PATH=$3

# Run the Python script to convert the graphs into feature vectors
python3 convert_to_feature_vector.py --graphs $GRAPH_PATH --subgraphs $DISCRIMINATIVE_SUBGRAPHS_PATH --output $FEATURES_PATH
