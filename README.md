# Machine Learning Models for Fraud Detection in Credit Card Transactions

[![Document](https://img.shields.io/badge/Read-Full%20Technical%20Report-blue?style=for-the-badge)](reports/REPORT.md)

An end-to-end data science project analyzing, preprocessing, and modeling severe class imbalance in financial transaction datasets. For a detailed report, please read a [Full Technical Report](reports/REPORT.md).
By [Nazarii Dubelovskyi](https://github.com/NDubelovskyi), 2026.

## Topic and Objective
* **Domain:** Credit Card Fraud Detection (handling high-volume, high-stakes financial data).
* **The Challenge:** Over 2 billion daily transactions globally where fraud costs billions, yet anomalies represent a tiny fraction of the data.
* **Core Goal:** Build, evaluate, and compare machine learning frameworks to maximize **Recall** (capturing maximum fraud vectors) while maintaining acceptable precision.

## Data Highlights
Before model training, a rigorous preprocessing pipeline was built to handle the complex [IEEE-CIS dataset](https://www.kaggle.com/competitions/ieee-fraud-detection/overview) (~590k transactions, ~430 features, with only 3.5% of fraud observations):
* **Zero Data Leakage:** 80/20 train-test split executed *before* any statistical transformations.
* **Dimensionality Reduction:** Reduced feature space from ~430 down to 202 columns using missing-value cutoffs and cross-correlation filtering (>0.95 threshold).
* **Cardinality Control:** Implemented a custom Top-10 frequency filter for high-cardinality categorical variables to prevent one-hot encoding explosion.
* **Advanced Imputation:** Implemented dynamic numerical imputation, utilizing feature **median** for highly skewed long-tail distributions and **mean** for normal distributions.

## Learning Models
A combination of hyperparameter grid searches, class-weight balancing, and prediction threshold tuning was used to optimize performance on non-linear patterns.

**Optimal Hyperparameters for Each Model**

| Model               | Key Hyperparameters                              | Best Values                   |
| ------------------- | ------------------------------------------------ | ----------------------------- |
| Logistic Regression | C, penalty, class_weight, threshold              | C=10, L1, {0:1,1:5}, 0.4      |
| K-Nearest Neighbors | n_neighbors, weights, distance metric, threshold | k=5, distance, Manhattan, 0.1 |
| K-Means             | n_clusters, batch_size, anomaly threshold        | 70, 5120, 92%                 |

## Results
* KNN performed best overall due to non-linear patterns in the data. 
* Supervised models significantly outperformed unsupervised K-Means.

**Performance Comparison of Fraud Detection Models**

| Model               | Accuracy | F1-Score | Recall | Precision |
| ------------------- | -------: | -------: | -----: | --------: |
| Logistic Regression |    66.3% |    12.5% |  69.9% |      6.8% |
| K-Nearest Neighbors |    91.2% |    32.5% |  61.9% |     22.1% |
| K-Means             |    90.6% |    17.9% |  29.9% |     12.8% |