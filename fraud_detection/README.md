# Fraud Detection Experiment

This project is a standalone, reproducible machine learning experiment designed to scientifically compare fraud detection performance **without preprocessing** versus **with preprocessing**.

## Project Overview

The main objective is to evaluate whether adding a preprocessing pipeline (handling class imbalance via SMOTE and outlier handling via RobustScaler) improves fraud detection performance compared to a minimal baseline model. The same exact train/test split, random state, and underlying algorithm (Logistic Regression) are used for both pipelines to isolate the effect of preprocessing.

## Dataset

- **Name:** Credit Card Fraud Detection (OpenML ID: 1597)
- **Source:** Automatically downloaded via `sklearn.datasets.fetch_openml` if not present.
- **Target Variable:** `Class` (0 = Legitimate, 1 = Fraud)
- **Characteristics:** The dataset is highly imbalanced with fraud accounting for less than 1% of transactions. All features (V1-V28, Time, Amount) are continuous/numerical.

## Methodology

### 1. Without Preprocessing Pipeline
- **Goal:** Establish a baseline performance with minimal alterations to the original data.
- **Process:** We only ensure features are numeric, drop the target variable, and train the model on the original scale. Imbalanced classes are addressed minimally via the model's built-in `class_weight="balanced"`.

### 2. With Preprocessing Pipeline
- **Goal:** Apply rigorous preprocessing steps to handle data distribution and imbalance.
- **Process:** 
  - Duplicate records are removed to clean the data.
  - Features are scaled using `RobustScaler` to handle outliers effectively.
  - Class imbalance in the *training* set is mitigated using Synthetic Minority Over-sampling Technique (SMOTE).
- **Leakage Prevention:** SMOTE is only applied to the training data. The scaler is fitted only on the training set and applied to the test set.

## Model Configuration

Both pipelines use the identical core model configuration:
```python
LogisticRegression(
    max_iter=1000,
    random_state=42,
    class_weight="balanced"
)
```

## Metrics

We evaluate performance with a focus on metrics suited for highly imbalanced datasets:
- **Accuracy:** The overall correctness (less informative for imbalanced problems).
- **Precision (Fraud):** The proportion of predicted frauds that are actual frauds (reduces false positives).
- **Recall (Fraud):** The proportion of actual frauds successfully identified.
- **F1-Score:** The harmonic mean of precision and recall.
- **ROC-AUC:** The model's ability to discriminate between classes across thresholds.
- **PR-AUC (Average Precision):** The area under the Precision-Recall curve, highly sensitive to imbalanced positive class.

## Reproducibility

To ensure the results are robust and strictly comparable:
- The random seed `RANDOM_STATE = 42` is used for all train-test splits, model initializations, and SMOTE resampling.
- The evaluation procedure strictly measures performance against a held-out test set unseen during training or preprocessing.

## Execution

Ensure you have installed the requirements:
```bash
pip install -r requirements.txt
```

Run the complete pipeline with the following commands:
```bash
python without_preprocessing/train.py
python without_preprocessing/evaluate.py

python with_preprocessing/train.py
python with_preprocessing/evaluate.py

python comparison/compare.py
```

## Results

*Please refer to `results/comparison/comparison.csv` and `results/comparison/conclusion.txt` after running the execution steps.*
