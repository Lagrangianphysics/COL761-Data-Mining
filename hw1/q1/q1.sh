#!/bin/bash

# bash q1.sh <path_apriori_executable> <path_fp_executable> <path_dataset> <path_out> 
# <path_apriori_executable>: absolute filepath to apriori's compiled executable
# <path_fp_executable>: aboslute filepath to fp-tree's compiled executable
# <path_dataset>: absolute filepath to the dataset file 
# <path_out>: absolute folderpath where the plot and the outputs at different thresholds will be saved

# Input parameters
AP_PATH=$1
FP_PATH=$2
DATA_PATH=$3
FOLDER_PATH=$4

# Run the Python script to convert the graphs into feature vectors
python3 script.py --ap $AP_PATH --fp $FP_PATH --dataset $DATA_PATH --opfolder $FOLDER_PATH
