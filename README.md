
# COL761: Data Mining

Welcome to the **COL761: Data Mining** repository. This repository contains all the assignments and supporting materials for the Data Mining course. You will find three homework assignments (HW1, HW2, HW3), each with full problem statements, starter code, datasets, and instructions for submission.

---

## 📂 Repository Structure

```

.
├── hw1/
│   ├── data/
│   │   ├── dataset1.csv
│   │   └── dataset2.csv
│   ├── hw1.ipynb
│   ├── problem\_statement.md
│   └── README.md
├── hw2/
│   ├── data/
│   │   ├── train.csv
│   │   ├── test.csv
│   │   └── labels.csv
│   ├── hw2.py
│   ├── problem\_statement.md
│   └── README.md
├── hw3/
│   ├── data/
│   │   └── real\_world\_data.csv
│   ├── hw3.R
│   ├── problem\_statement.md
│   └── README.md
├── LICENSE
└── README.md

````

---

## 📝 Homework 1

### Problem Statement (hw1/problem_statement.md)

> **Objective:** Implement and analyze fundamental data mining algorithms from scratch.  
>
> **Tasks:**  
> 1. **Data Preprocessing**  
>    - Handle missing values using mean/mode imputation.  
>    - Normalize continuous features to [0,1] range.  
> 2. **K-Means Clustering**  
>    - Implement K-Means with both random and k-means++ initialization.  
>    - Run on `dataset1.csv` and `dataset2.csv` for k=2,3,4,5.  
>    - Compute and plot the within‐cluster sum of squares (WCSS) vs. k.  
> 3. **Hierarchical Agglomerative Clustering (HAC)**  
>    - Implement HAC with single, complete, and average linkage.  
>    - Plot dendrograms for each linkage on `dataset1.csv`.  
> 4. **Evaluation**  
>    - Compare K-Means and HAC cluster assignments using Adjusted Rand Index (ARI).  

See [hw1/README.md](hw1/README.md) for detailed instructions, code templates, and submission guidelines.

---

## 📝 Homework 2

### Problem Statement (hw2/problem_statement.md)

> **Objective:** Explore supervised learning methods and performance evaluation.  
>
> **Tasks:**  
> 1. **Decision Trees**  
>    - Implement ID3 algorithm from scratch.  
>    - Train on `train.csv`, evaluate on `test.csv`.  
>    - Plot decision tree structure and report accuracy, precision, recall, F1‐score.  
> 2. **k-Nearest Neighbors (kNN)**  
>    - Implement kNN for k = 1,3,5,7.  
>    - Use Euclidean distance on normalized features.  
>    - Compare classification metrics across k.  
> 3. **Naïve Bayes**  
>    - Implement Gaussian Naïve Bayes for continuous features.  
>    - Compare against Decision Tree and kNN.  
> 4. **Cross-Validation**  
>    - Perform 5-fold CV on the training set for each method.  
>    - Report mean and standard deviation of accuracy.  

See [hw2/README.md](hw2/README.md) for starter code, dataset descriptions, and submission instructions.

---

## 📝 Homework 3

### Problem Statement (hw3/problem_statement.md)

> **Objective:** Apply data mining techniques to a real‐world dataset and interpret your findings.  
>
> **Tasks:**  
> 1. **Exploratory Data Analysis (EDA)**  
>    - Generate summary statistics and visualizations (histograms, boxplots).  
>    - Identify and handle outliers or anomalies.  
> 2. **Feature Engineering**  
>    - Create at least three new meaningful features.  
>    - Document your rationale.  
> 3. **Predictive Modeling**  
>    - Train at least two different models (e.g., logistic regression, random forest).  
>    - Perform hyperparameter tuning (grid search or random search).  
>    - Evaluate on a held‐out test set and report detailed metrics.  
> 4. **Interpretation & Report**  
>    - Use SHAP or feature‐importance to interpret model decisions.  
>    - Prepare a concise report summarizing insights, challenges, and conclusions.  

See [hw3/README.md](hw3/README.md) for full details, code templates (in R), and dataset guidelines.

---

## 🚀 Getting Started

1. **Clone the repository:**
   ```bash
   git clone https://github.com/Lagrangianphysics/COL761-Data-Mining.git
   cd COL761-Data-Mining
````

2. **Install dependencies:**

   * For Python assignments (HW1 & HW2):

     ```bash
     pip install -r hw1/requirements.txt
     pip install -r hw2/requirements.txt
     ```
   * For R assignment (HW3): ensure you have R ≥ 4.0 and install packages in `hw3/README.md`.
3. **Run notebooks / scripts:**

   * HW1: open `hw1/hw1.ipynb` in Jupyter.
   * HW2: execute `python hw2/hw2.py --help` for usage.
   * HW3: run `Rscript hw3/hw3.R` after setting working directory.

---

## 📦 Dependencies

* **Python 3.8+**

  * numpy
  * pandas
  * scikit-learn
  * matplotlib
  * scipy
* **R ≥ 4.0**

  * tidyverse
  * randomForest
  * shapforxgboost (or SHAP package)

---

## 📄 License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.

---

## 🤝 Contributing

Contributions, issues, and feature requests are welcome!
Please open an issue to discuss your idea or submit a pull request.

---

## 📬 Contact

Maintainer: **Lagrangianphysics**
Email: [subhojitiitd@gmail.com](mailto:subhojitiitd@gmail.com)

Feel free to reach out with questions or clarifications. Good luck and happy mining!

```
```
