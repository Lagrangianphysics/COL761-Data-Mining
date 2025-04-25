#!/bin/bash

# Check if the correct number of arguments is provided
if [ "$#" -ne 3 ]; then
    echo "Usage: bash test2.sh <path_to_test_graph> <path_to_model> <output_file_path>"
    exit 1
fi

# Assign arguments to variables
TEST_GRAPH_PATH=$1
MODEL_PATH=$2
OUTPUT_FILE_PATH=$3

# Call the Python script with the provided arguments
python3 src/task2/test2.py --testGraphPath "$TEST_GRAPH_PATH" --modelPath "$MODEL_PATH" --outFile "$OUTPUT_FILE_PATH"