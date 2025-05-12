# COL761 Data Mining Assignments

This repository contains solutions to the assignments for the COL761 Data Mining course (IIT Delhi, 2024-25). Each assignment focuses on a different aspect of data mining, including frequent pattern mining, influence maximization in networks, and graph neural networks.

---

## Table of Contents

- [Assignment Overview](#assignment-overview)
- [HW1: Frequent Pattern Mining & Graph Classification](#hw1-frequent-pattern-mining--graph-classification)
- [HW2: Influence Maximization in Social Networks](#hw2-influence-maximization-in-social-networks)
- [HW3: Graph Neural Networks for Bipartite Graphs](#hw3-graph-neural-networks-for-bipartite-graphs)
- [Directory Structure](#directory-structure)
- [Submission & Evaluation Guidelines](#submission--evaluation-guidelines)
- [References](#references)

---

## Assignment Overview

| Assignment | Topic                                                                                       | Directory |
|------------|--------------------------------------------------------------------------------------------|-----------|
| HW1        | Frequent Itemset Mining, Frequent Subgraph Mining, Graph Classification                    | `hw1/`    |
| HW2        | Influence Maximization in Social Networks (Virality Problem)                               | `hw2/`    |
| HW3        | Graph Neural Networks for Node and User/Product Classification on Bipartite Graphs         | `hw3/`    |

---

## HW1: Frequent Pattern Mining & Graph Classification

**Problem Statement:**  
HW1 consists of three parts:

1. **Frequent Itemset Mining:**  
   Empirically compare the Apriori and FP-tree algorithms for frequent itemset mining on a provided dataset. Analyze efficiency at various support thresholds and visualize the results.  
   *Deliverables:* Scripts for running experiments, plots, and a report with analysis.

2. **Frequent Subgraph Mining:**  
   Run gSpan, FSG (PAFI), and Gaston on the Yeast dataset at various minimum supports. Plot and analyze the running times, and discuss the trends observed.  
   *Deliverables:* Scripts for running experiments, plots, and a report with analysis.

3. **Graph Classification (Competitive):**  
   Identify discriminative subgraphs for molecular graph classification. Convert each graph into a binary feature vector and use these features for classification.  
   *Deliverables:* Scripts for feature extraction, conversion, and classification, along with a report explaining the feature selection process.

For detailed problem statements, see the [hw1.pdf](hw1.pdf) in the `hw1/` directory[3].

---

## HW2: Influence Maximization in Social Networks

**Problem Statement:**  
Given a directed graph representing a town's social network, with edge probabilities indicating the likelihood of disease transmission, select $$ k $$ individuals to vaccinate such that the expected spread of an infection is minimized (i.e., maximize the expected number of protected individuals). The problem is stochastic and requires randomized simulations to estimate expected spread.

**Tasks:**
- **Task 1:** Reduce the virality problem to a well-known NP-hard problem.
- **Task 2:** Propose an efficient approximate algorithm and analyze its complexity.
- **Task 3:** Construct a hypothetical dataset where the algorithm may perform sub-optimally.
- **Task 4:** Implement and run the algorithm on provided datasets; competitive evaluation based on achieved spread.

*For complete details, see [hw2.pdf](hw2.pdf) in the `hw2/` directory[1].*

---

## HW3: Graph Neural Networks for Bipartite Graphs

**Problem Statement:**  
This assignment explores the use of Graph Neural Networks (GNNs) for node and user/product classification on bipartite graphs using PyTorch and PyTorch Geometric.

**Tasks:**
1. **Task 1:**  
   Design and train a GNN to predict node labels in a graph with node features and partial labels. Evaluate the model on unseen nodes using appropriate metrics (ROC-AUC, Accuracy).

2. **Task 2:**  
   Extend the GNN approach to a bipartite graph representing users and products, with the aim of predicting multi-dimensional user personality vectors from user/product features and interaction edges. Evaluate using weighted F1-score.

*For full problem statements and submission instructions, see [hw3.pdf](hw3.pdf) in the `hw3/` directory[2].*

---

## Directory Structure

```
COL761-Data-Mining/
├── hw1/
│   ├── q1/
│   ├── q2/
│   └── q3/
├── hw2/
│   ├── src/
│   ├── solution.sh
│   └── report.pdf
└── hw3/
    └── src/
        ├── train1.sh
        ├── test1.sh
        ├── train2.sh
        └── test2.sh
```

- See each assignment's PDF for the expected subdirectory and file structure[3][1][2].

---

## References

- [HW1 Problem Statement (hw1.pdf)](hw1.pdf)
- [HW2 Problem Statement (hw2.pdf)](hw2.pdf)
- [HW3 Problem Statement (hw3.pdf)](hw3.pdf)

For any implementation-specific or dataset-related queries, refer to the assignment PDFs in their respective directories.
