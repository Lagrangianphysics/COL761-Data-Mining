
bash identify.sh "/home/baadalvm/HW1/Q3_new/q3_datasets/NCI-H23/train_graphs.txt" "/home/baadalvm/HW1/Q3_new/q3_datasets/NCI-H23/train_labels.txt" "/home/baadalvm/HW1/Q3_new/cache_files/discriminative_subgraphs.txt"

bash convert.sh "/home/baadalvm/HW1/Q3_new/q3_datasets/NCI-H23/train_graphs.txt" "/home/baadalvm/HW1/Q3_new/cache_files/discriminative_subgraphs.txt" "/home/baadalvm/HW1/Q3_new/cache_files/feature_train.npy" 
bash convert.sh "/home/baadalvm/HW1/Q3_new/q3_datasets/NCI-H23/test_graphs.txt" "/home/baadalvm/HW1/Q3_new/cache_files/discriminative_subgraphs.txt" "/home/baadalvm/HW1/Q3_new/cache_files/feature_test.npy"

python3 classify.py --ftrain "/home/baadalvm/HW1/Q3_new/cache_files/feature_train.npy" --ftest "/home/baadalvm/HW1/Q3_new/cache_files/feature_test.npy" --ltrain "/home/baadalvm/HW1/Q3_new/q3_datasets/NCI-H23/train_labels.txt" --ltest "/home/baadalvm/HW1/Q3_new/q3_datasets/NCI-H23/test_labels.txt" --proba "/home/baadalvm/HW1/Q3_new/cache_files/prob.txt"