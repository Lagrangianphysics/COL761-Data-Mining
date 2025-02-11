import sys
from graph_data_structure import *
import pandas as pd
from sklearn.model_selection import train_test_split
import time
import random

dataset_path = "q3_datasets/Mutagenicity/graphs.txt"
labels_path = "q3_datasets/Mutagenicity/labels.txt"

test_dataset_path = "q3_datasets/Mutagenicity/test_graphs.txt"
test_label_path = "q3_datasets/Mutagenicity/test_labels.txt"

train_dataset_path = "q3_datasets/Mutagenicity/train_graphs.txt"
train_label_path = "q3_datasets/Mutagenicity/train_labels.txt"



if __name__ == "__main__":

    start_time = time.time()

    graph_data = read_normal_graphs_from_file(dataset_path)
    
    labels = []
    with open(labels_path, 'r') as file:
        labels = list(map(int, file.read().splitlines()))

    df = pd.DataFrame({'graph': graph_data, 'label': labels})

    print("- - - - - Data load complete - - - - -")

    df['graph'] = df['graph'].astype(object)

    X = df['graph'].copy()
    y = df['label'].copy()


    r_int = random.randint(20,80)

    print(f"random state = {r_int}")

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=r_int)

    with open(train_dataset_path , 'w') as file:
        file.write("")
    for G in X_train:
        G.append_to_file_normal(train_dataset_path)

    with open(train_label_path , 'w') as file:
        file.write("\n".join(map(str, y_train)))



    with open(test_dataset_path , 'w') as file:
        file.write("")
    for G in X_test:
        G.append_to_file_normal(test_dataset_path)

    with open(test_label_path , 'w') as file:
        file.write("\n".join(map(str, y_test)))

    print("- - - - - Split File write complete - - - - -")

    end_time = time.time()
    total_time_taken = end_time - start_time
    print(f"total time taken = [{total_time_taken}]")
    