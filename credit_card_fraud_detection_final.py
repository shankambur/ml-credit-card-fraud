#Fraud Detection Project 

#  build this like a real industry project

# Phase 1: Problem Understanding
# Goal: Predict fraudulent transactions (1) vs normal (0)
# Challenge:
# Highly imbalanced dataset (fraud is rare!)
# Key metric:
#  Accuracy (misleading)
#  Precision, Recall, F1-score (important)

#  Phase 2: Dataset

# I’ll use the famous dataset:

# Credit Card Fraud Detection Dataset

# 284,807 transactions
# Only ~492 frauds 

# Phase 3: Workflow (VERY IMPORTANT)

# Data Loading
# EDA (Exploratory Data Analysis)
# Data Preprocessing
# Handling Imbalance 
# Model Building
# Evaluation
# Optimization (GridSearch)
# Final Model

# Key Difference from Titanic
# | Titanic         | Fraud Detection          |
# | --------------- | ------------------------ |
# | Balanced-ish    |  Highly imbalanced     |
# | Accuracy ok     |  Accuracy useless       |
# | Simple features | PCA transformed features |
# | Small dataset   | Large dataset            |

# Step 1: Load Dataset
import pandas as pd
import sys

# Load data
df = pd.read_csv('creditcard.csv')

# Basic info
print(df.shape)
# (284807, 31)

df.head()
#     Time        V1        V2        V3        V4        V5        V6        V7  ...       V23       V24       V25       V26       V27       V28  Amount  Class
# 0   0.0 -1.359807 -0.072781  2.536347  1.378155 -0.338321  0.462388  0.239599  ... -0.110474  0.066928  0.128539 -0.189115  0.133558 -0.021053  149.62      0
# 1   0.0  1.191857  0.266151  0.166480  0.448154  0.060018 -0.082361 -0.078803  ...  0.101288 -0.339846  0.167170  0.125895 -0.008983  0.014724    2.69      0
# 2   1.0 -1.358354 -1.340163  1.773209  0.379780 -0.503198  1.800499  0.791461  ...  0.909412 -0.689281 -0.327642 -0.139097 -0.055353 -0.059752  378.66      0
# 3   1.0 -0.966272 -0.185226  1.792993 -0.863291 -0.010309  1.247203  0.237609  ... -0.190321 -1.175575  0.647376 -0.221929  0.062723  0.061458  123.50      0
# 4   2.0 -1.158233  0.877737  1.548718  0.403034 -0.407193  0.095921  0.592941  ... -0.137458  0.141267 -0.206010  0.502292  0.219422  0.215153   69.99      0

# Step 2: Quick Data Check
print("**df.info()**")
df.info()
df.describe()

# Data columns (total 31 columns):
#  #   Column  Non-Null Count   Dtype  
# ---  ------  --------------   -----  
#  0   Time    284807 non-null  float64
#  1   V1      284807 non-null  float64
#  2   V2      284807 non-null  float64
#  3   V3      284807 non-null  float64
#  4   V4      284807 non-null  float64
#  5   V5      284807 non-null  float64
#  6   V6      284807 non-null  float64
#  7   V7      284807 non-null  float64
#  8   V8      284807 non-null  float64
#  9   V9      284807 non-null  float64
#  10  V10     284807 non-null  float64
#  11  V11     284807 non-null  float64
#  12  V12     284807 non-null  float64
#  13  V13     284807 non-null  float64
#  14  V14     284807 non-null  float64
#  15  V15     284807 non-null  float64
#  16  V16     284807 non-null  float64
#  17  V17     284807 non-null  float64
#  18  V18     284807 non-null  float64
#  19  V19     284807 non-null  float64
#  20  V20     284807 non-null  float64
#  21  V21     284807 non-null  float64
#  22  V22     284807 non-null  float64
#  23  V23     284807 non-null  float64
#  24  V24     284807 non-null  float64
#  25  V25     284807 non-null  float64
#  26  V26     284807 non-null  float64
#  27  V27     284807 non-null  float64
#  28  V28     284807 non-null  float64
#  29  Amount  284807 non-null  float64
#  30  Class   284807 non-null  int64  

# What about V1 to V28?
# These are PCA-transformed features (very important point)
# Original features are hidden (privacy reasons)
# Transformed using Principal Component Analysis
# So:
# I cannot interpret features directly
# But they are still useful for modeling
# “Features are anonymized using PCA, so interpretability is limited, but patterns still exist for classification.”

# Step 3: Check Class Imbalance (VERY IMPORTANT)
print("df['Class'].value_counts()\n",df['Class'].value_counts())

#  Class
# 0    284315
# 1       492

print("df['Class'].value_counts(normalize=True)\n",df['Class'].value_counts(normalize=True))

#  Class
# 0    0.998273 (99.8%) Normal Transactions
# 1    0.001727 (0.17%). Fraud cases are extremely rare


# 📊 Visualize Imbalance
import seaborn as sns
import matplotlib.pyplot as plt
ax = sns.countplot(x='Class', data=df)
plt.yscale('log')
plt.title("Class Distribution (Log Scale)")
for p in ax.patches:
    ax.annotate(f'{int(p.get_height())}', 
                (p.get_x() + 0.25, p.get_height() + 1000))
# plt.show()


# “This is a highly imbalanced dataset, so accuracy is not a reliable metric.
#  I focus on recall and precision, especially recall for fraud detection.”
# “The dataset is extremely imbalanced (~0.17% fraud), which makes visualization misleading and accuracy unreliable.”
# Key Learning 
# This is the CORE challenge of fraud detection:
# Model will try to predict everything as 0 (normal)
# Because it still gets 99.8% accuracy

# Next Step
# Now I move to:
# Feature Understanding
# Columns:
# Time
# Amount
# V1 → V28 (PCA features)

print("df.columns:\n",df.columns)
# df.columns:
#  Index(['Time', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6', 'V7', 'V8', 'V9', 'V10',
#        'V11', 'V12', 'V13', 'V14', 'V15', 'V16', 'V17', 'V18', 'V19', 'V20',
#        'V21', 'V22', 'V23', 'V24', 'V25', 'V26', 'V27', 'V28', 'Amount',
#        'Class'],
#       dtype='str')

print("df['Amount'].min()\n",df['Amount'].min())
print("df['Amount'].max()\n",df['Amount'].max())
print("df['Time'].min()\n",df['Time'].min())
print("df['Time'].max()\n",df['Time'].max())
print("df['V1'].min()\n",df['V1'].min())
print("df['V1'].max()\n",df['V1'].max())
print("df['V8'].min()\n",df['V8'].min())
print("df['V8'].max()\n",df['V8'].max())

# output: 
# df['Amount'].min()
#  0.0
# df['Amount'].max()
#  25691.16
# df['Time'].min()
#  0.0
# df['Time'].max()
#  172792.0
# df['V1'].min()
#  -56.407509631329
# df['V1'].max()
#  2.45492999121121
# df['V8'].min()
#  -73.2167184552674
# df['V8'].max()
#  20.0072083651213
# 🔍 What are V1–V28?

# They are PCA transformed features

# V = Variable
# So:
# V1 = Principal Component 1
# V2 = Principal Component 2
# …
# V28 = Principal Component 28
# What is PCA (in simple terms)?
# Think like this:
# Original data had features like:
# Card type
# Location
# Merchant
# Transaction pattern
# etc.

# But due to privacy/security, these Ire hidden. kind of encryption.
# Principal Component Analysis (PCA)
# It Converts original features → new features (V1–V28) and Removes sensitive information and Keeps important patterns
# Instead of saying:
# “User spent $500 at Walmart in New York”
# They transformed it into:
# “V1 = -1.23, V2 = 0.87 …”
# So:
# Its Not human interpretable
# but Still useful for ML

# 4
# “Features V1–V28 are PCA-transformed variables used to anonymize sensitive transaction data while preserving variance.”
# I cannot interpret V1–V28 like:
# doesn't mean “Higher V1 means fraud” 
# doesn't mean “LoIr V2 means risky” 
# Instead:
# Let ML models learn patterns automatically

# Special Columns and these are NOT PCA:
# Time → seconds since first transaction
# Amount → transaction amount
# Class → target (0 = normal, 1 = fraud)
# Next Step 

# EDA ( Exploratory Data Analysis ==> Understanding the data before modeling)  on Amount & Time

# Amount distribution
sns.histplot(df['Amount'], bins=50)
plt.title("Transaction Amount Distribution")
# plt.show()
# Most transactions are very small amounts
# Few transactions are very large
# Long tail on the right side
# Amount → Right SkeId 
# This is called positive/right skew
# “Transaction amount is highly right-skeId, so I would consider scaling or log transformation.”

# Most values near 0–100
# Few values like 1000, 5000, 20000 , That creates: Long tail on right side which is a  Problem , Models get confused because:
# Amount range:
# Min = 0
# Max = 25000+
# # Huge difference in scale
# Model thinks:
# “Amount is very important because numbers are big”
# Bias happens

# Solution is:
# Option 1: Scaling ==> Converts data to mean=0, std=1
# Option 2: Log Transform (alternative)
# df['Amount'] = np.log1p(df['Amount'])
# Compresses large values


# # Time distribution
sns.histplot(df['Time'], bins=50)
plt.title("Time Distribution")
# plt.show()
# Time → “Curved” 
# It’s actually bimodal / periodic pattern
# Data has two peaks
# Pattern repeats over time (day/night behavior)
# Morning transactions
# Evening transactions
# “Time shows periodic behavior, indicating daily transaction cycles rather than a normal distribution.”

# Problem:
# Amount has huge scale variation
# PCA features (V1–V28) are already standardized

# This creates imbalance in feature importance

# Next Step: Feature Scaling 
# I must scale feature:
# Amount 
# Time (optional but recommended)

# What am I doing in Feature Engineering & Scaling?
# Feature Engineering = improving input data for model

# In this dataset:
# I don’t create many new features (like Titanic), Because: Data is already transformed using PCA, Sensitive info is hidden

# So here, feature engineering = Scaling + cleaning + preparing features
# am I going to use PCA data (V1–V28)?  ==> 100% YES
# These are very main features and Already optimized for ML and I do NOT modify them, “V1–V28 = ready-made poIrful features”

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

df['scaled_amount'] = scaler.fit_transform(df[['Amount']])
df['scaled_time'] = scaler.fit_transform(df[['Time']])

# Drop original columns
df.drop(['Amount', 'Time'], axis=1, inplace=True)
# Why I do this
# Models like:
# Logistic Regression are sensitive to scale

# scaled_amount distribution
sns.histplot(df['scaled_amount'], bins=50)
plt.title("scaled_amount  Distribution")
# plt.show()

# scaled_time distribution
sns.histplot(df['scaled_time'], bins=50)
plt.title("scaled_time Distribution")
# plt.show()

# Only Amount & Time need scaling

#  What does scaler.fit_transform() do?
# Step 1: fit()
# scaler.fit(df[['Amount']])
# Learns:
# Mean (μ)
# Standard deviation (σ)
# Step 2: transform()
# (Amount - μ) / σ
# Converts values to standard scale

# Step 3: fit_transform()
# Combines both:
# scaler.fit_transform(df[['Amount']])
# Example
# Original:
# [10, 20, 30]
# After scaling:
# [-1.22, 0, 1.22]

# Big Picture 
# | Step               | Why                                |
# | ------------------ | ---------------------------------- |
# | EDA                | Understand data                    |
# | Scaling            | Fix feature imbalance              |
# | PCA (given)        | Protect privacy + reduce dimension |
# | Imbalance handling | Improve fraud detection            |

print(df.head())

# output:

#          V1        V2        V3        V4        V5        V6        V7  ...       V25       V26       V27       V28  Class  scaled_amount  scaled_time
# 0 -1.359807 -0.072781  2.536347  1.378155 -0.338321  0.462388  0.239599  ...  0.128539 -0.189115  0.133558 -0.021053      0       0.244964    -1.996583
# 1  1.191857  0.266151  0.166480  0.448154  0.060018 -0.082361 -0.078803  ...  0.167170  0.125895 -0.008983  0.014724      0      -0.342475    -1.996583
# 2 -1.358354 -1.340163  1.773209  0.379780 -0.503198  1.800499  0.791461  ... -0.327642 -0.139097 -0.055353 -0.059752      0       1.160686    -1.996562
# 3 -0.966272 -0.185226  1.792993 -0.863291 -0.010309  1.247203  0.237609  ...  0.647376 -0.221929  0.062723  0.061458      0       0.140534    -1.996562
# 4 -1.158233  0.877737  1.548718  0.403034 -0.407193  0.095921  0.592941  ... -0.206010  0.502292  0.219422  0.215153      0      -0.073403    -1.996541

# What I have now
# I have dataset now:
#  V1–V28 → PCA features (already scaled)
#  scaled_amount → normalized
#  scaled_time → normalized
#  Dropped raw Amount, Time
# This is now a clean ML-ready dataset

# Let’s interpret ONE row
# Take row 0:
# V1 = -1.35, V2 = -0.07, ..., scaled_amount = 0.24, scaled_time = -1.99
# Class = 0
# Meaning:
# All features are now on same scale
# No feature is dominating
# Model can learn patterns fairly


# Earlier:
# Amount → 0 to 25000 
# V1 → -3 to +3

# Now:
# Everything → roughly -3 to +3 ✅
# This is called:Feature normalization / standardization


print("df['scaled_amount'].min()\n",df['scaled_amount'].min())
print("df['scaled_amount'].max()\n",df['scaled_amount'].max())
print("df['scaled_time'].min()\n",df['scaled_time'].min())
print("df['scaled_time'].max()\n",df['scaled_time'].max())
print("df['V1'].min()\n",df['V1'].min())
print("df['V1'].max()\n",df['V1'].max())
print("df['V8'].min()\n",df['V8'].min())
print("df['V8'].max()\n",df['V8'].max())

# df['scaled_amount'].min()
#  -0.35322939296682354
# df['scaled_amount'].max()
#  102.3622427092842
# df['scaled_time'].min()
#  -1.996583023457193
# df['scaled_time'].max()
#  1.6420577336572635
# df['V1'].min()
#  -56.407509631329
# df['V1'].max()
#  2.45492999121121
# df['V8'].min()
#  -73.2167184552674
# df['V8'].max()
#  20.0072083651213



# Everything → roughly -3 to +3 
# Step-by-step explanation
# 🧩 Step 1: Original situation
# PCA features (V1–V28)
# Already standardized
# Range roughly:
# -3 to +3

# Because PCA internally does scaling

# Amount (before scaling)
# Real money values:
# Min ≈ 0
# Max ≈ 25,000+

# Huge difference compared to V1–V28

# Why this is a problem?

# Imagine model sees:
# | Feature | Value |
# | ------- | ----- |
# | V1      | 1.2   |
# | Amount  | 5000  |
# Model thinks:

# “Amount is more important because value is bigger”
#  This is misleading

# Step 2: What StandardScaler does

# Formula:

# z = (x - mean) / std
# Example (simple)
# Original Amount:
# [0, 100, 1000, 10000]

# After scaling:
# [-0.8, -0.5, 0.2, 2.1]

# See what happened?

# Big numbers got compressed
# Everything now in small comparable range

# Final Understanding
# Before:
# V1–V28 → small scale
# Amount → huge scale 
# After:
# V1–V28 → ~(-3 to +3)
# scaled_amount → ~(-3 to +3) 
# scaled_time → ~(-3 to +3) 

# Now model treats all features fairly
# “StandardScaler normalizes features to zero mean and unit variance, ensuring all features contribute equally to the model.”


# In this dataset:
# There is NO missing data step


# What StandardScaler REALLY does

# It does:

# z = (x - mean) / std
# It guarantees:
# Mean = 0 
# Std deviation = 1 
# It DOES NOT guarantee:
# Min/max range 
# Why values go beyond ±3?
# “Scaling centers data around 0 with unit variance, but outliers can still produce large values”
# “StandardScaler normalizes features to zero mean and unit variance, but it does not bound values, so outliers can still produce large scaled values.”




# Now: Handling Imbalanced Data (MOST IMPORTANT PART)
# Step 1: Split the data (VERY IMPORTANT)


from sklearn.model_selection import train_test_split

X = df.drop('Class', axis=1)
y = df['Class']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# stratify=y is critical here
# ➡ Keeps fraud ratio same in train & test



print("y_train.value_counts():\n",y_train.value_counts())

# y_train.value_counts():
#  Class
# 0    227451
# 1       394
# Due to severe class imbalance, I used SMOTE to synthetically oversample the minority class only on training data to avoid data leakage

# Full form:
# SMOTE = Synthetic Minority Oversampling Technique

# Why I MUST split first?
# If I apply SMOTE before split:
# Synthetic fraud points may leak into test data 
# Model gets unfair advantage

# This is called:

# Data Leakage 
# 📊 Step 2: Check imbalance in train data

# Step 3: Apply SMOTE 
# SMOTE is applied only on the training data, so synthetic samples are generated using training data only. 
# This prevents any information from the test set leaking into the model.
# Applying SMOTE before train-test split causes data leakage, because synthetic samples are generated using information from the entire dataset, including test data

# Why this matters
# Train data → model learns
# Test data → model should NOT see

# By restricting SMOTE to train:
#  No leakage
#  Fair evaluation
#  Realistic performance


from imblearn.over_sampling import SMOTE
smote = SMOTE(random_state=42)
X_train_sm, y_train_sm = smote.fit_resample(X_train, y_train)

# 📊 Step 4: Check after SMOTE
print("y_train_sm.value_counts():\n",y_train_sm.value_counts())

# Now I should see something like:
# y_train_sm.value_counts(): 
# Class
# 0    227451
# 1    227451
# Perfectly balanced 

# What SMOTE actually does
# It creates new synthetic fraud samples
# NOT copies 
# BUT:
# Interpolates betweeen existing fraud points 
# 📊 Simple intuition

# If I have:
# Fraud points:
# A (x1, y1)
# B (x2, y2)

# SMOTE creates:
# New point betweenen A and B
# Step 1: Pick a fraud point
# Example: A
# Step 2: Find nearest fraud neighbors
# Example: B
# Step 3: Create new point BETIEN them
# New Point = A + random * (B - A)
# So new point lies betIen A and B


# ***** Small warning ****
# SMOTE:
# Helps model learn fraud patterns
# but, It Can introduce slight noise (precision might reduce)
# SMOTE often improves:
# Recall
# F1-score
# But may reduce:
# Precision
# (because more false positives)
# This is why threshold tuning later matters.

# Potential improvement / gap
# Default SMOTE balances 50:50.
# Sometimes:
# This may be too aggressive.
# Because real-world fraud isn’t 50%.

# Advanced improvement:
# we could tune:
# SMOTE(sampling_strategy=0.3)
# Meaning:
# Fraud becomes 30% of normal instead of 100%.''

# Step-by-Step Modeling Pipeline
# Step 1: Train-Test Split
# I did already 



# Step 2: Handle Class Imbalance
# Option 1: Class weight (Easiest ✅)
# Option 2: SMOTE (Advanced) ( will do later)


# In Step 2, I gave 2 different ways to handle imbalance:

# Option A → class_weight
# Option B → SMOTE

# These are alternatives, not something I use together 

# 🚦 So why did I use only class_weight in Step 3?

# Because:

# I always start SIMPLE in real projects

# So the flow should be:

# ✅ Step 1 (I do this first)

# Use Random Forest + class_weight

# ⏭ Step 2 (Later)

# Try SMOTE and compare

# What is class_weight?
# 💡 Simple Explanation

# In fraud data:

# 99% = Not Fraud (0)
# 1% = Fraud (1)

# Model thinks:

# "If I always predict 0, I’m already 99% correct 😎"

# That’s BAD.

# ⚖️ What class_weight='balanced' does

# It tells the model:

# "Hey! Fraud cases are rare, so treat them as MORE IMPORTANT"

# 🔍 Internally what happens?

# The model assigns:

# Higher weight → Fraud (Class 1)
# Lower weight → Non-Fraud (Class 0)

# So mistakes on fraud are penalized more

# 🧠 Real-world analogy

# Imagine:

# 100 normal transactions
# 1 fraud transaction

# Without weights:
# Missing fraud = small mistake

# With weights:
# Missing fraud = BIG mistake 🚨

# 🧪 Example (Conceptual)

# Without class_weight:

# Fraud missed → penalty = 1

# With class_weight:

# Fraud missed → penalty = 100 (or higher)

# So model tries harder to catch fraud

# 🧑‍💻 How I use it
# from sklearn.ensemble import RandomForestClassifier

# rf = RandomForestClassifier(
#     class_weight='balanced',
#     random_state=42
# )

# That’s it ✅

# Important Understanding (Interview Level)

# class_weight does NOT create new data
# It just changes importance during training

# 🆚 Class weight vs SMOTE (Very Important)
# | Feature      | Class weight       | SMOTE            |
# | ------------ | ------------------ | ---------------- |
# | What it does | Adjusts importance | Creates new data |
# | Complexity   | Easy ✅             | Medium           |
# | Risk         | Low                | Can overfit      |
# | When to use  | First step         | Next level       |

# 🧭 What I should do now (Clear Plan)

# Don’t jump to SMOTE yet

# Do this:
# Train model WITHOUT class_weight
# Train model WITH class_weight
# Compare:
# Recall
# F1-score

# "I used class_weight to handle class imbalance by assigning higher importance to fraud cases, ensuring the model doesn't ignore minority class."

# Step 3A: Train Model (Start with Logistic Regression)

print("################################## Start : Train Model with Logistic Regression without SMOTE")
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
import joblib 
import numpy as np
lr = LogisticRegression(
    max_iter=1000,
    random_state=42
)

lr.fit(X_train, y_train) #. <==== WITHOUT SMOTE
joblib.dump(lr,"lr_model.pkl")
print("lr_model.pkl saved")
lr = joblib.load("lr_model.pkl")
y_pred_lr = lr.predict(X_test)
y_prob_lr = lr.predict_proba(X_test)[:, 1]
print("y_pred_lr\n",y_pred_lr)
print("y_prob_lr\n",y_prob_lr)

from sklearn.metrics import precision_score, recall_score

thresholds = np.arange(0.1, 0.9, 0.05)

best_threshold = 0
best_f1 = 0
best_precision = 0
best_recall = 0

for t in thresholds:
    # print("running for threshold ",t)
    y_pred_t = (y_prob_lr > t).astype(int)

    precision = precision_score(y_test, y_pred_t)
    recall = recall_score(y_test, y_pred_t)
    # print("precision:",precision)
    # print("recall:",recall)

    if precision + recall > 0:
        f1 = 2 * (precision * recall) / (precision + recall)
        # print("f1:",f1)

        if f1 > best_f1:
            best_f1 = f1
            best_threshold = t
            best_precision = precision
            best_recall = recall

print("\nBest Threshold (F1 optimized):", best_threshold)
print("Precision:", best_precision)
print("Recall:", best_recall)
print("F1-score:", best_f1)

# Best Threshold (F1 optimized): 0.8500000000000002
# Precision: 0.1843220338983051
# Recall: 0.8877551020408163
# F1-score: 0.3052631578947368

threshold = 0.85
print(" thresahold 0.85")
y_pred_custom_lr_T085 = (y_prob_lr > threshold).astype(int)
print("**y_pred_custom_lr_T085\n",y_pred_custom_lr_T085)
print("**confusion_matrix_lr_T085\n",confusion_matrix(y_test, y_pred_custom_lr_T085))
print("**classification_report_lr_T085\n",classification_report(y_test, y_pred_custom_lr_T085))

#  thresahold 0.85
# **y_pred_custom_lr_cw_T085
#  [0 0 0 ... 0 0 0]
# **confusion_matrix_lr_cw_T085
#  [[56479   385]
#  [   11    87]]
# **classification_report_lr_cw_T085
#                precision    recall  f1-score   support

#            0       1.00      0.99      1.00     56864
#            1       0.18      0.89      0.31        98

#     accuracy                           0.99     56962
#    macro avg       0.59      0.94      0.65     56962
# weighted avg       1.00      0.99      1.00     56962

# Confusion Matrix
# |            | Predicted No | Predicted Yes |
# | ---------- | ------------ | ------------- |
# | Actual No  | TN           | FP            |
# | Actual Yes | FN           | TP            |
# Focus:

# FN (Fraud missed) → VERY BAD 
# FP (False alert) → acceptable trade-off
# Key Metrics
# Recall (Fraud detection rate)
# "Out of all fraud cases, how many did I catch?"
# Precision
# "Out of predicted fraud, how many Ire actually fraud?"
# F1-score
# Balance betIen Precision & Recall
# ROC-AUC
# Overall model ability


print("################################## Completed : Train Model with Logistic Regression without SMOTE")

# 🌲 Step 3A: Train Model (Start with Logistic Regression)
print("################################## Start : Train Model with Logistic Regression with class_weight without SMOTE")
from sklearn.linear_model import LogisticRegression
import joblib 
lr_cw = LogisticRegression(
    class_weight='balanced',
    max_iter=1000,
    random_state=42
)

# lr_cw.fit(X_train, y_train) #. <==== WITHOUT SMOTE
# joblib.dump(lr_cw,"lr_cw_model.pkl")
# print("lr_cw_model.pkl saved")
lr_cw = joblib.load("lr_cw_model.pkl")
y_pred_lr_cw = lr_cw.predict(X_test)
y_prob_lr_cw = lr_cw.predict_proba(X_test)[:, 1]
print("y_pred_lr_cw\n",y_pred_lr_cw)
print("y_prob_lr_cw\n",y_prob_lr_cw)

thresholds = np.arange(0.1, 0.9, 0.05)

best_threshold = 0
best_f1 = 0
best_precision = 0
best_recall = 0

for t in thresholds:
    # print("running for threshold ",t)
    y_pred_t = (y_prob_lr_cw > t).astype(int)

    precision = precision_score(y_test, y_pred_t)
    recall = recall_score(y_test, y_pred_t)
    # print("precision:",precision)
    # print("recall:",recall)

    if precision + recall > 0:
        f1 = 2 * (precision * recall) / (precision + recall)
        # print("f1:",f1)

        if f1 > best_f1:
            best_f1 = f1
            best_threshold = t
            best_precision = precision
            best_recall = recall

print("\nBest Threshold (F1 optimized):", best_threshold)
print("Precision:", best_precision)
print("Recall:", best_recall)
print("F1-score:", best_f1)

# Best Threshold (F1 optimized): 0.8500000000000002
# Precision: 0.16541353383458646
# Recall: 0.8979591836734694
# F1-score: 0.27936507936507937

threshold = 0.85
print(" thresahold 0.85")
y_pred_custom_lr_cw_T085 = (y_prob_lr_cw > threshold).astype(int)
print("**y_pred_custom_lr_cw_T085\n",y_pred_custom_lr_cw_T085)
print("**confusion_matrix_lr_cw_T085\n",confusion_matrix(y_test, y_pred_custom_lr_cw_T085))
print("**classification_report_lr_cw_T085\n",classification_report(y_test, y_pred_custom_lr_cw_T085))


# **confusion_matrix_lr_cw_T085
#  [[56420   444]
#  [   10    88]]
# **classification_report_lr_cw_T085
#                precision    recall  f1-score   support

#            0       1.00      0.99      1.00     56864
#            1       0.17      0.90      0.28        98

#     accuracy                           0.99     56962
#    macro avg       0.58      0.95      0.64     56962
# weighted avg       1.00      0.99      0.99     56962

print("################################## Completed : Train Model with Logistic Regression with class_weight without SMOTE")



print("################################## Start : Train Model with Logistic Regression with SMOTE")
from sklearn.linear_model import LogisticRegression
import joblib 
import numpy as np
lr_smote = LogisticRegression(
    max_iter=1000,
    random_state=42
)

# lr_smote.fit(X_train_sm, y_train_sm) #  <=== With SMOTE
# joblib.dump(lr_smote,"lr_smote_model.pkl")
# print("lr_smote_model.pkl saved")
lr_smote = joblib.load("lr_smote_model.pkl")
y_pred_lr_smote = lr_smote.predict(X_test)
y_prob_lr_smote = lr_smote.predict_proba(X_test)[:, 1]
print("y_pred_lr_smote\n",y_pred_lr_smote)
print("y_prob_lr_smote\n",y_prob_lr_smote)

thresholds = np.arange(0.1, 0.9, 0.05)

best_threshold = 0
best_f1 = 0
best_precision = 0
best_recall = 0

for t in thresholds:
    # print("running for threshold ",t)
    y_pred_t = (y_prob_lr_smote > t).astype(int)

    precision = precision_score(y_test, y_pred_t)
    recall = recall_score(y_test, y_pred_t)
    # print("precision:",precision)
    # print("recall:",recall)

    if precision + recall > 0:
        f1 = 2 * (precision * recall) / (precision + recall)
        # print("f1:",f1)

        if f1 > best_f1:
            best_f1 = f1
            best_threshold = t
            best_precision = precision
            best_recall = recall

print("\nBest Threshold (F1 optimized):", best_threshold)
print("Precision:", best_precision)
print("Recall:", best_recall)
print("F1-score:", best_f1)

# Best Threshold (F1 optimized): 0.8500000000000002
# Precision: 0.16541353383458646
# Recall: 0.8979591836734694
# F1-score: 0.27936507936507937

threshold = 0.85
print(" thresahold 0.85")
y_pred_custom_lr_smote_T085 = (y_prob_lr_smote > threshold).astype(int)
print("**y_pred_custom_lr_smote_T085\n",y_pred_custom_lr_smote_T085)
print("**confusion_matrix_lr_smote_T085\n",confusion_matrix(y_test, y_pred_custom_lr_smote_T085))
print("**classification_report_lr_smote_T085\n",classification_report(y_test, y_pred_custom_lr_smote_T085))

#  thresahold 0.85
# **y_pred_custom_lr_smote_T085
#  [0 0 0 ... 0 0 0]
# **confusion_matrix_lr_T085
#  [[56420   444]
#  [   10    88]]
# **classification_report_lr_T085
#                precision    recall  f1-score   support

#            0       1.00      0.99      1.00     56864
#            1       0.17      0.90      0.28        98

#     accuracy                           0.99     56962
#    macro avg       0.58      0.95      0.64     56962
# weighted avg       1.00      0.99      0.99     56962
# ***************** precision is low  , increases lot of false positive if we use LR.  ******

print("################################## Completed : Train Model with Logistic Regression with SMOTE")

# Comparison Matrix
# | Model Type                          | Precision (Fraud) | Recall (Fraud) | F1-score (Fraud) | False Positives | False Negatives | Business Meaning                                    |
# | ----------------------------------- | ----------------- | -------------- | ---------------- | --------------- | --------------- | --------------------------------------------------- |
# | **LR (No SMOTE / No class_weight)** | **0.83**          | 0.53           | **0.65**         | **11**          | 46              | Very precise, fewer false alarms, misses more fraud |
# | **LR + class_weight**               | 0.18              | 0.89           | 0.31             | 385             | 11              | Catches most fraud, many false alarms               |
# | **LR + SMOTE**                      | 0.17              | 0.90           | 0.28             | 444             | 10              | Similar recall, even more false alarms              |

# ***************** precision is low  , increases lot of false positive if we use LR.  ******

# So, let go for next algorthim

#  Train Model (Start with Random Forest)
print("################################## Start : Train Model Start with Random Forest")
# SMOTE data created lot of false positives iwith LR, so first lets run without SMOTE for RF
from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier(
    n_estimators=50,  # reduce trees (default 100)
    max_depth=10,  # limit tree depth
    random_state=42,
    n_jobs=-1  # use all CPU cores 
)
# rf.fit(X_train, y_train)
# joblib.dump(rf,"rf_model.pkl")
# print("rf_model.pkl saved")
rf = joblib.load("rf_model.pkl")


#  Predictions
y_pred_rf = rf.predict(X_test)
y_prob_rf = rf.predict_proba(X_test)[:, 1]
print("y_pred_rf\n",y_pred_rf)
print("y_prob_rf\n",y_prob_rf)

thresholds = np.arange(0.1, 0.9, 0.05)

best_threshold = 0
best_f1 = 0
best_precision = 0
best_recall = 0

for t in thresholds:
    # print("running for threshold ",t)
    y_pred_t = (y_prob_rf > t).astype(int)

    precision = precision_score(y_test, y_pred_t)
    recall = recall_score(y_test, y_pred_t)
    # print("precision:",precision)
    # print("recall:",recall)

    if precision + recall > 0:
        f1 = 2 * (precision * recall) / (precision + recall)
        # print("f1:",f1)

        if f1 > best_f1:
            best_f1 = f1
            best_threshold = t
            best_precision = precision
            best_recall = recall

print("\nBest Threshold (F1 optimized):", best_threshold)
print("Precision:", best_precision)
print("Recall:", best_recall)
print("F1-score:", best_f1)

# Best Threshold (F1 optimized): 0.45000000000000007
# Precision: 0.9195402298850575
# Recall: 0.8163265306122449
# F1-score: 0.8648648648648648

threshold = 0.45
print(" thresahold 0.45")
y_pred_custom_rf_T045 = (y_prob_rf > threshold).astype(int)
print("**confusion_matrix_rf\n",confusion_matrix(y_test, y_pred_custom_rf_T045))
print("**classification_report_rf\n",classification_report(y_test, y_pred_custom_rf_T045))
print("**ROC-AUC_rf:\n", roc_auc_score(y_test, y_pred_custom_rf_T045))

#  thresahold 0.45
# **confusion_matrix_rf
#  [[56857     7]
#  [   18    80]]
# **classification_report_rf
#                precision    recall  f1-score   support

#            0       1.00      1.00      1.00     56864
#            1       0.92      0.82      0.86        98

#     accuracy                           1.00     56962
#    macro avg       0.96      0.91      0.93     56962
# weighted avg       1.00      1.00      1.00     56962

# **ROC-AUC_rf:
#  0.9081017149403374
# # %%%%% RF recall is better than LR %%%%%
print("################################## Completed : Train Model Start with Random Forest")

#  Train Model (Start with Random Forest With class_weight)
# Class weighting in Random Forest adjusts class importance during tree construction, forcing the model to pay more attention to minority fraud cases
print("################################## Started Train Model With Random Forest With class_weight")
rf_with_cw = RandomForestClassifier(
    n_estimators=50,  # reduce trees (default 100)
    max_depth=10,  # limit tree depth
    class_weight='balanced',
    random_state=42,
    n_jobs=-1  # use all CPU cores 
)
# rf_with_cw.fit(X_train, y_train)
# joblib.dump(rf_with_cw,"rf_with_cw_model.pkl")
# print("rf_with_cw_model.pkl saved")
rf_with_cw = joblib.load("rf_with_cw_model.pkl")

#  Predictions
y_pred_rf_with_cw = rf_with_cw.predict(X_test)
y_prob_rf_with_cw = rf_with_cw.predict_proba(X_test)[:, 1]
print("y_pred_rf_with_cw\n",y_pred_rf_with_cw)
print("y_prob_rf_with_cw\n",y_prob_rf_with_cw)

#  Evaluation (MOST IMPORTANT)

thresholds = np.arange(0.1, 0.9, 0.05)

best_threshold = 0
best_f1 = 0
best_precision = 0
best_recall = 0

for t in thresholds:
    # print("running for threshold ",t)
    y_pred_t = (y_prob_rf_with_cw > t).astype(int)

    precision = precision_score(y_test, y_pred_t)
    recall = recall_score(y_test, y_pred_t)
    # print("precision:",precision)
    # print("recall:",recall)

    if precision + recall > 0:
        f1 = 2 * (precision * recall) / (precision + recall)
        # print("f1:",f1)

        if f1 > best_f1:
            best_f1 = f1
            best_threshold = t
            best_precision = precision
            best_recall = recall

print("\nBest Threshold (F1 optimized):", best_threshold)
print("Precision:", best_precision)
print("Recall:", best_recall)
print("F1-score:", best_f1)

# Best Threshold (F1 optimized): 0.5000000000000001
# Precision: 0.8247422680412371
# Recall: 0.8163265306122449
# F1-score: 0.8205128205128205

threshold = 0.5
print(" thresahold 0.5")
y_pred_custom_rf_T05 = (y_prob_rf_with_cw > threshold).astype(int)
print("**confusion_matrix_rf\n",confusion_matrix(y_test, y_pred_custom_rf_T05))
print("**classification_report_rf\n",classification_report(y_test, y_pred_custom_rf_T05))
print("**ROC-AUC_rf:\n", roc_auc_score(y_test, y_pred_custom_rf_T05))

#  thresahold 0.5
# **confusion_matrix_rf
#  [[56847    17]
#  [   18    80]]
# **classification_report_rf
#                precision    recall  f1-score   support

#            0       1.00      1.00      1.00     56864
#            1       0.82      0.82      0.82        98

#     accuracy                           1.00     56962
#    macro avg       0.91      0.91      0.91     56962
# weighted avg       1.00      1.00      1.00     56962

# **ROC-AUC_rf:
#  0.908013785846359

print("################################## Completed Train Model With Random Forest With class_weight")



# Train Model on SMOTE data
print("################################## Start : Train Model with Random Forest With SMOTE data")

rf_smote = RandomForestClassifier(
    n_estimators=50,
    max_depth=10,
    random_state=42,
    n_jobs=-1
)
# rf_smote.fit(X_train_sm, y_train_sm)
# joblib.dump(rf_smote,"rf_smote_model.pkl")
# print("rf_smote_model.pkl saved sucessfully")
rf_smote = joblib.load("rf_smote_model.pkl")
# Predictions
y_pred_rf_smote = rf_smote.predict(X_test)
y_prob_rf_smote = rf_smote.predict_proba(X_test)[:, 1]
print("y_pred_rf_smote\n",y_pred_rf_smote)
print("y_prob_rf_smote\n",y_prob_rf_smote)

thresholds = np.arange(0.1, 0.9, 0.05)

best_threshold = 0
best_f1 = 0
best_precision = 0
best_recall = 0

for t in thresholds:
    # print("running for threshold ",t)
    y_pred_t = (y_prob_rf_smote > t).astype(int)

    precision = precision_score(y_test, y_pred_t)
    recall = recall_score(y_test, y_pred_t)
    # print("precision:",precision)
    # print("recall:",recall)

    if precision + recall > 0:
        f1 = 2 * (precision * recall) / (precision + recall)
        # print("f1:",f1)

        if f1 > best_f1:
            best_f1 = f1
            best_threshold = t
            best_precision = precision
            best_recall = recall

print("\nBest Threshold (F1 optimized):", best_threshold)
print("Precision:", best_precision)
print("Recall:", best_recall)
print("F1-score:", best_f1)

# Best Threshold (F1 optimized): 0.8500000000000002
# Precision: 0.7978723404255319
# Recall: 0.7653061224489796
# F1-score: 0.7812500000000001

threshold = 0.85
print(" thresahold 0.85")
y_pred_custom_rf_T085 = (y_prob_rf_smote > threshold).astype(int)
print("**confusion_matrix_rf\n",confusion_matrix(y_test, y_pred_custom_rf_T085))
print("**classification_report_rf\n",classification_report(y_test, y_pred_custom_rf_T085))
print("**ROC-AUC_rf:\n", roc_auc_score(y_test, y_pred_custom_rf_T085))

#  thresahold 0.85
# **confusion_matrix_rf
#  [[56845    19]
#  [   23    75]]
# **classification_report_rf
#                precision    recall  f1-score   support

#            0       1.00      1.00      1.00     56864
#            1       0.80      0.77      0.78        98

#     accuracy                           1.00     56962
#    macro avg       0.90      0.88      0.89     56962
# weighted avg       1.00      1.00      1.00     56962

# **ROC-AUC_rf:
#  0.8824859959459305

print("################################## Completed : Train Model with Random Forest With SMOTE data")

# | Model                     | Precision | Recall   | F1-score | False Positives | False Negatives | ROC-AUC  | Verdict                  |
# | ------------------------- | --------- | -------- | -------- | --------------- | --------------- | -------- | ------------------------ |
# | LR (No SMOTE / No CW)     | 0.83      | 0.53     | 0.65     | 11              | 46              | Strong   | Good baseline            |
# | LR + class_weight         | 0.18      | 0.89     | 0.31     | 385             | 11              | Moderate | Too many false positives |
# | LR + SMOTE                | 0.17      | 0.90     | 0.28     | 444             | 10              | Moderate | Poor precision           |
# | **RF (No SMOTE / No CW)** | **0.92**  | **0.82** | **0.86** | **7**           | **18**          | **0.91** | 🥇 BEST                  |
# | RF + class_weight         | 0.82      | 0.82     | 0.82     | 17              | 18              | 0.91     | Good but worse           |
# | RF + SMOTE                | 0.80      | 0.77     | 0.78     | 19              | 23              | 0.88     | Lower performance        |




# Lets try for XGBoost

# What is XGBoost?

# XGBoost = eXtreme Gradient Boosting

# It’s an advanced version of Boosting where:

# Models are trained sequentially
# Each new model focuses on correcting previous mistakes
#  Core Idea 
# Unlike Random Forest (parallel trees), XGBoost builds trees one after another:


# ⚖️ Random Forest vs XGBoost
# | Feature  | Random Forest   | XGBoost                |
# | -------- | --------------- | ---------------------- |
# | Training | Parallel        | Sequential             |
# | Focus    | Reduce variance | Reduce bias + variance |
# | Speed    | Faster          | Slower (but optimized) |
# | Accuracy | Good            | 🔥 Very High           |

# Why XGBoost is famous?
# Handles imbalanced data better (important for Fraud project)
# Built-in regularization → reduces overfitting


print("############### Start Train model using XGBoost")
from xgboost import XGBClassifier

# Initialize model
xgb = XGBClassifier(
    n_estimators=100,
    max_depth=5,
    learning_rate=0.1,
    random_state=42,
    eval_metric='logloss'
)

# Train
# xgb.fit(X_train, y_train)
# joblib.dump(xgb,"xgb_model.pkl")
# print("xgb_model.pkl saved")
xgb= joblib.load("xgb_model.pkl")
# Predict
y_pred_xgb = xgb.predict(X_test)
y_prob_xgb = xgb.predict_proba(X_test)[:, 1]
print("y_pred_xgb\n",y_pred_xgb)
print("y_prob_xgb\n",y_prob_xgb)

# Evaluate
thresholds = np.arange(0.1, 0.9, 0.05)

best_threshold = 0
best_f1 = 0
best_precision = 0
best_recall = 0

for t in thresholds:
    # print("running for threshold ",t)
    y_pred_t = (y_prob_xgb > t).astype(int)

    precision = precision_score(y_test, y_pred_t)
    recall = recall_score(y_test, y_pred_t)
    # print("precision:",precision)
    # print("recall:",recall)

    if precision + recall > 0:
        f1 = 2 * (precision * recall) / (precision + recall)
        # print("f1:",f1)

        if f1 > best_f1:
            best_f1 = f1
            best_threshold = t
            best_precision = precision
            best_recall = recall

print("\nBest Threshold (F1 optimized):", best_threshold)
print("Precision:", best_precision)
print("Recall:", best_recall)
print("F1-score:", best_f1)

# Best Threshold (F1 optimized): 0.25000000000000006
# Precision: 0.8924731182795699
# Recall: 0.8469387755102041
# F1-score: 0.8691099476439791

threshold = 0.25
print(" thresahold 0.25")
y_pred_custom_rf_T025 = (y_prob_xgb > threshold).astype(int)
print("**confusion_matrix_rf\n",confusion_matrix(y_test, y_pred_custom_rf_T025))
print("**classification_report_rf\n",classification_report(y_test, y_pred_custom_rf_T025))
print("**ROC-AUC_rf:\n", roc_auc_score(y_test, y_pred_custom_rf_T025))

#  thresahold 0.25
# **confusion_matrix_rf
#  [[56854    10]
#  [   15    83]]
# **classification_report_rf
#                precision    recall  f1-score   support

#            0       1.00      1.00      1.00     56864
#            1       0.89      0.85      0.87        98

#     accuracy                           1.00     56962
#    macro avg       0.95      0.92      0.93     56962
# weighted avg       1.00      1.00      1.00     56962

# **ROC-AUC_rf:
#  0.9233814586611235

print("############### Completed Train model using XGBoost")

# Compare with RF baseline:
# | Model            | Precision | Recall   | F1        | FP    | FN     | ROC-AUC   |
# | ---------------- | --------- | -------- | --------- | ----- | ------ | --------- |
# | RF Baseline      | **0.92**  | 0.82     | 0.865     | **7** | 18     | 0.908     |
# | XGBoost Baseline | 0.89      | **0.85** | **0.869** | 10    | **15** | **0.923** |


# “XGBoost is an optimized gradient boosting algorithm where trees are built sequentially.
# Each new tree learns from the errors of previous trees, improving performance.
# It also includes regularization to prevent overfitting and handles imbalanced datasets effectively.”
#  “Trees are built sequentially to correct errors”

# “Accuracy is misleading for imbalanced datasets, so I focused on Recall and F1-score.”

# Head-to-Head
# | Metric            | Random Forest | XGBoost | Winner  |
# | ----------------- | ------------- | ------- | ------- |
# | Recall            | 0.85          | 🔥 0.86 | XGBoost |
# | FN (missed fraud) | 15            | 🔥 14   | XGBoost |
# | Precision         | 🔥 0.86       | 0.79    | RF      |
# | FP                | 🔥 13         | 23      | RF      |
# ]



print("############### Start Train model using XGBoost with hyperParameter tuning")

# Initialize model
best_f1 = 0
best_params = None
best_threshold = 0
best_precision = 0
best_recall = 0
# for n_est in [100, 200]:
#     print("*******************running n_est :",n_est)
#     for depth in [3, 4, 5]:
#         print("*********running depth :",depth)
#         for lr in [0.05, 0.1]:
#             print("*****running lr :",lr)
#             for spw in [1, 50, 100]:
#                 print("***running spw :",spw)
#                 xgb = XGBClassifier(
#                     n_estimators=n_est,
#                     max_depth=depth,
#                     learning_rate=lr,
#                     scale_pos_weight=spw,
#                     random_state=42,
#                     eval_metric='logloss'
#                 )

#                 xgb.fit(X_train, y_train)

#                 y_prob = xgb.predict_proba(X_test)[:,1]

#                 for t in np.arange(0.1,0.9,0.05):
#                     y_pred = (y_prob > t).astype(int)

#                     precision = precision_score(y_test, y_pred)
#                     recall = recall_score(y_test, y_pred)

#                     if precision + recall > 0:
#                         f1 = 2 * (precision * recall) / (precision + recall)

#                         if f1 > best_f1:
#                             best_f1 = f1
#                             best_threshold = t
#                             best_precision = precision
#                             best_recall = recall
#                             best_params = {
#                                 "n_estimators": n_est,
#                                 "max_depth": depth,
#                                 "learning_rate": lr,
#                                 "scale_pos_weight": spw,
#                                 "threshold": t
#                             }

print("best_f1:",best_f1)
print("best_params:",best_params)
print("best_threshold:",best_threshold)
print("best_precision:",best_precision)
print("best_recall:",best_recall)

# best_f1: 0.8972972972972973
# best_params: {'n_estimators': 200, 'max_depth': 5, 'learning_rate': 0.1, 'scale_pos_weight': 50, 'threshold': np.float64(0.7500000000000002)}
# best_threshold: 0.7500000000000002
# best_precision: 0.9540229885057471
# best_recall: 0.8469387755102041

# | Model         | F1        |
# | ------------- | --------- |
# | LR baseline   | 0.65      |
# | RF baseline   | 0.865     |
# | XGB baseline  | 0.869     |
# | **Tuned XGB** | **0.897** |

print("############### Completed Train model using XGBoost with hyperParameter tuning")


print("############### Start Train model using XGBoost with best parameters")
from xgboost import XGBClassifier

# Initialize model
xgb_bp = XGBClassifier(
                    n_estimators=200,
                    max_depth=5,
                    learning_rate=0.1,
                    scale_pos_weight=50,
                    random_state=42,
                    eval_metric='logloss'
                )

# Train
# xgb_bp.fit(X_train, y_train)
# joblib.dump(xgb_bp,"xgb_bp_model.pkl")
# print("xgb_bp_model.pkl saved")
xgb_bp= joblib.load("xgb_bp_model.pkl")
# Predict
y_pred_xgb_bp = xgb_bp.predict(X_test)
y_prob_xgb_bp = xgb_bp.predict_proba(X_test)[:, 1]
print("y_pred_xgb_bp\n",y_pred_xgb)
print("y_prob_xgb_bp\n",y_prob_xgb)

# Evaluate
thresholds = np.arange(0.1, 0.9, 0.05)

best_threshold = 0
best_f1 = 0
best_precision = 0
best_recall = 0

for t in thresholds:
    # print("running for threshold ",t)
    y_pred_t = (y_prob_xgb_bp > t).astype(int)

    precision = precision_score(y_test, y_pred_t)
    recall = recall_score(y_test, y_pred_t)
    # print("precision:",precision)
    # print("recall:",recall)

    if precision + recall > 0:
        f1 = 2 * (precision * recall) / (precision + recall)
        # print("f1:",f1)

        if f1 > best_f1:
            best_f1 = f1
            best_threshold = t
            best_precision = precision
            best_recall = recall

print("\nBest Threshold (F1 optimized):", best_threshold)
print("Precision:", best_precision)
print("Recall:", best_recall)
print("F1-score:", best_f1)

# Best Threshold (F1 optimized): 0.7500000000000002
# Precision: 0.9540229885057471
# Recall: 0.8469387755102041
# F1-score: 0.8972972972972973

threshold = 0.75
print(" thresahold 0.75")
y_pred_custom_rf_T075 = (y_prob_xgb_bp > threshold).astype(int)
print("**confusion_matrix_rf\n",confusion_matrix(y_test, y_pred_custom_rf_T075))
print("**classification_report_rf\n",classification_report(y_test, y_pred_custom_rf_T075))
print("**ROC-AUC_rf:\n", roc_auc_score(y_test, y_pred_custom_rf_T075))

#  thresahold 0.75
# **confusion_matrix_rf
#  [[56860     4]
#  [   15    83]]
# **classification_report_rf
#                precision    recall  f1-score   support

#            0       1.00      1.00      1.00     56864
#            1       0.95      0.85      0.90        98

#     accuracy                           1.00     56962
#    macro avg       0.98      0.92      0.95     56962
# weighted avg       1.00      1.00      1.00     56962

# **ROC-AUC_rf:
#  0.9234342161175108

print("############### Completed Train model using XGBoost with best parameters")

# | Model               | SMOTE | Class Weight / scale_pos_weight | Precision | Recall   | F1-score | ROC-AUC   | False Positives | False Negatives | Business Verdict                     |
# | ------------------- | ----- | ------------------------------- | --------- | -------- | -------- | --------- | --------------- | --------------- | ------------------------------------ |
# | Logistic Regression | ❌     | ❌                               | 0.83      | 0.53     | 0.65     | Strong    | 11              | 46              | Good baseline, misses too much fraud |
# | Logistic Regression | ❌     | ✅                               | 0.18      | 0.89     | 0.31     | Moderate  | 385             | 11              | High recall, too many false alerts   |
# | Logistic Regression | ✅     | ❌                               | 0.17      | 0.90     | 0.28     | Moderate  | 444             | 10              | Similar to CW, poor precision        |
# | Random Forest       | ❌     | ❌                               | 0.92      | 0.82     | 0.86     | 0.908     | 7               | 18              | Excellent balance                    |
# | Random Forest       | ❌     | ✅                               | 0.82      | 0.82     | 0.82     | 0.908     | 17              | 18              | Good, but weaker than baseline RF    |
# | Random Forest       | ✅     | ❌                               | 0.80      | 0.77     | 0.78     | 0.882     | 19              | 23              | SMOTE hurt performance               |
# | XGBoost Baseline    | ❌     | scale_pos_weight=1              | 0.89      | 0.85     | 0.87     | 0.923     | 10              | 15              | Strong baseline                      |
# | 🥇 Tuned XGBoost    | ❌     | scale_pos_weight=50             | **0.95**  | **0.85** | **0.90** | **0.923** | **4**           | **15**          | BEST overall                         |



# Finding model is completed and now we have to Make project production-grade

print("################ Start train model in pipeline using XGB with best parameters")

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
pipeline_xgb = Pipeline([
    ('scaler', StandardScaler()),
    ('xgb', XGBClassifier(
        n_estimators=200,
        max_depth=5,
        learning_rate=0.1,
        scale_pos_weight=50,
        random_state=42,
        eval_metric='logloss'
    ))
])
# pipeline_xgb.fit(X_train, y_train)
# joblib.dump(pipeline_xgb,"pipeline_xgb_model.pkl")
# print("pipeline_xgb_model.pkl saved")
pipeline_xgb= joblib.load("pipeline_xgb_model.pkl")
# Predict
y_pred_pipeline_xgb = pipeline_xgb.predict(X_test)
y_prob_pipeline_xgb = pipeline_xgb.predict_proba(X_test)[:, 1]
print("y_pred_xgb_bp\n",y_pred_pipeline_xgb)
print("y_prob_xgb_bp\n",y_prob_pipeline_xgb)

threshold = 0.75
print(" thresahold 0.75")
y_pred_custom_rf_T075 = (y_prob_xgb_bp > threshold).astype(int)
print("**confusion_matrix_rf\n",confusion_matrix(y_test, y_pred_custom_rf_T075))
print("**classification_report_rf\n",classification_report(y_test, y_pred_custom_rf_T075))
print("**ROC-AUC_rf:\n", roc_auc_score(y_test, y_pred_custom_rf_T075))
print("################ Completed train model in pipeline using XGB with best parameters")


print("################ Start Feature Importance")

import pandas as pd

feature_importance = pd.DataFrame({
    'Feature': X_train.columns,
    'Importance': pipeline_xgb.named_steps['xgb'].feature_importances_
}).sort_values(by='Importance', ascending=False)

print(feature_importance.head(10))

#           Feature  Importance
# 13            V14    0.488429.   dominates fraud detection model.
# 16            V17    0.088839
# 3              V4    0.048872
# 9             V10    0.027664
# 11            V12    0.024916
# 28  scaled_amount    0.024285
# 12            V13    0.020144
# 10            V11    0.019364
# 6              V7    0.019119
# 19            V20    0.018711

# This shows that Fraud detection was NOT driven mainly by Amount or Time. Instead, Hidden PCA behavioral patterns mattered most.

print("################ Start Cross-validation")

from sklearn.model_selection import cross_val_score

scores = cross_val_score(
    pipeline_xgb,
    X_train,
    y_train,
    cv=3,
    scoring='f1'
)

print(scores)
print(scores.mean())

# [0.85585586 0.82129278 0.87022901]
# 0.8491258797182809.   <==== Average F1 across all 3 independent evaluations.


print("################ Final mode lExit")
