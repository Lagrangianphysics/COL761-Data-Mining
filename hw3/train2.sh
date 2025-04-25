#!/bin/bash

# Check if the correct number of arguments is provided
if [ "$#" -ne 2 ]; then
    echo "Usage: bash train2.sh <path_to_train_graph> <output_model_file_path>"
    exit 1
fi

# Assign input arguments to variables
TRAIN_GRAPH_PATH=$1
OUTPUT_MODEL_PATH=$2

# Call the Python script with the updated argument parser format
python src/task2/train2.py --trainGraphPath "$TRAIN_GRAPH_PATH" --outModel_path "$OUTPUT_MODEL_PATH"