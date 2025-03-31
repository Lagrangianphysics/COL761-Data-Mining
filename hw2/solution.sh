#!/bin/bash

# Check if the correct number of arguments is provided
if [ "$#" -ne 4 ]; then
    echo "Usage: bash solution.sh <absolute_path_to_graph> <absolute_output_file_path> <k> <#_random_instances>"
    exit 1
fi

# Assign arguments to variables
GRAPH_PATH=$1
OUTPUT_PATH=$2
K=$3
RANDOM_INSTANCES=$4

# Define paths
SRC_DIR="./src"
BIN_DIR="$SRC_DIR/bin"

# Compile main.cpp using g++ with C++11 standard and save the executable in src/bin
g++ -std=c++11 -O2 -fopenmp "$SRC_DIR/main.cpp" -o "$BIN_DIR/main" || { echo "Error: Compilation failed."; exit 1; }

# Run the compiled executable with the provided parameters
"$BIN_DIR/main" "$GRAPH_PATH" "$OUTPUT_PATH" "$K" "$RANDOM_INSTANCES"

# bash solution.sh data/dataset_1.txt output/seedset.txt 50 100
# g++ -std=c++11 -O2 -fopenmp "./src/main.cpp" -o "./src/bin/main"
# ./src/bin/main "data/dataset_1.txt" "output/seedset.txt" "50" "100"
