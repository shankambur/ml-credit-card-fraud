# C`redit Card Fraud Detection

## Problem Statement

Credit card fraud is a highly imbalanced classification problem where fraudulent transactions are rare compared to legitimate ones. The goal is to accurately detect fraud while minimizing false positives.

---

## Dataset

* Source: Kaggle Credit Card Fraud Dataset
* Highly imbalanced dataset (fraud < 1%)

---

## Approach

### 1. Data Preprocessing

* Handled class imbalance using **SMOTE**
* Feature scaling applied

### 2. Model Building

* Logistic Regression
* Random Forest
* XGBoost (final model)

### 3. Model Evaluation

* Precision
* Recall
* F1-score
* Confusion Matrix

---

## Final Model

* **XGBoost Classifier**
* Optimized using GridSearchCV
* Handles imbalance effectively

---

## Key Learnings

* Importance of handling imbalanced datasets
* Trade-off between precision and recall
* Model tuning improves performance significantly

---

## How to Run

```bash
python credit_card_fraud_detection.py
```

---

## Tech Stack

* Python
* Pandas, NumPy
* Scikit-learn
* XGBoost
* Imbalanced-learn (SMOTE)

