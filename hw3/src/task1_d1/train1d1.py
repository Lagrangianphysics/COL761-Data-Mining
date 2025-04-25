import argparse

train_graph_path = "input_me_milega..."
out_model_path = "input_me_milega..."

if __name__ == "__main__":
    # Set up argument parser
    parser = argparse.ArgumentParser(description="Train a model using the provided training graph.")
    parser.add_argument('--trainGraphPath', type=str, required=True, help="Path to the training graph file.")
    parser.add_argument('--outModel_path', type=str, required=True, help="Path to save the output model file.")
    
    # Parse the arguments
    args = parser.parse_args()
    
    # Extract arguments
    train_graph_path = args.trainGraphPath
    out_model_path = args.outModel_path
    
    # Print the paths (for debugging purposes)
    print(f"Training graph path: {train_graph_path}")
    print(f"Output model path: {out_model_path}")