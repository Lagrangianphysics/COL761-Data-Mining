#!/bin/bash

# cd hw1/q2 
# bash q2.sh <path_gspan_executable> <path_fsg_executable> <path_gaston_executable> <path_dataset> <path_out> 
# <path_gspan_executable>: absolute filepath to gspan's compiled executable
# <path_fsg_executable>: aboslute filepath to fsg's compiled executable
# <path_gaston_executable>: aboslute filepath to gaston's compiled executable
# <path_dataset>: absolute filepath to the dataset file 
# <path_out>: absolute folderpath where the plot and the outputs at different minimum supports will be saved


# Input parameters
GSPAN_PATH=$1
FSG_PATH=$2
GASTON_PATH=$3
DATASET_PATH=$4
FOLDER_PATH=$5

# Run the Python script to convert the graphs into feature vectors
python3 script.py --gspan $GSPAN_PATH --fsg $FSG_PATH --gaston $GASTON_PATH --dataset $DATASET_PATH --opfolder $FOLDER_PATH