print("######### Fraud Detection Project. #################")

#Problem Understanding
# Goal: Predict fraudulent transactions (1) vs normal (0)

# Dataset : We’ll use the famous dataset: Credit Card Fraud Detection Dataset
# 284,807 transactions
# Only ~492 frauds 

# Step 1: Load Dataset
# First — Download dataset
import pandas as pd
# Load data
df = pd.read_csv('creditcard.csv')
# Basic info
print("df.shape():\n",df.shape)
# (284807, 31)
print("df.head():\n",df.head())


# Step 2: Quick Data Check
print("df.info()\n",df.info())
print("df.describe()\n",df.describe())
# Class column is target : need to predict 


# 🚨 Step 3: Check Class Imbalance (VERY IMPORTANT)
print("df['Class'].value_counts()\n",df['Class'].value_counts())
#  Class
# 0    284315
# 1       492      <====. Fraud transactions are very minimal 
# normalize data : converts counts into percentages (proportions)
print("df['Class'].value_counts(normalize=True)\n",df['Class'].value_counts(normalize=True))
#  Class
# 0    0.998273
# 1    0.001727

# 📊 Visualize Imbalance
import seaborn as sns
import matplotlib.pyplot as plt
# ax = sns.countplot(x='Class', data=df)
# plt.yscale('log')
# plt.title("Class Distribution (Log Scale)")
# for p in ax.patches:
#     ax.annotate(f'{int(p.get_height())}', 
#                 (p.get_x() + 0.25, p.get_height() + 1000))
# plt.show()
# “The dataset is extremely imbalanced (~0.17% fraud), which makes visualization misleading and accuracy unreliable.”

#Next Step
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
print("df.head()\n",df.head())
#     Time        V1        V2        V3        V4        V5        V6        V7  ...       V23       V24       V25       V26       V27       V28  Amount  Class
# 0   0.0 -1.359807 -0.072781  2.536347  1.378155 -0.338321  0.462388  0.239599  ... -0.110474  0.066928  0.128539 -0.189115  0.133558 -0.021053  149.62      0
# 1   0.0  1.191857  0.266151  0.166480  0.448154  0.060018 -0.082361 -0.078803  ...  0.101288 -0.339846  0.167170  0.125895 -0.008983  0.014724    2.69      0
# 2   1.0 -1.358354 -1.340163  1.773209  0.379780 -0.503198  1.800499  0.791461  ...  0.909412 -0.689281 -0.327642 -0.139097 -0.055353 -0.059752  378.66      0
# 3   1.0 -0.966272 -0.185226  1.792993 -0.863291 -0.010309  1.247203  0.237609  ... -0.190321 -1.175575  0.647376 -0.221929  0.062723  0.061458  123.50      0
# 4   2.0 -1.158233  0.877737  1.548718  0.403034 -0.407193  0.095921  0.592941  ... -0.137458  0.141267 -0.206010  0.502292  0.219422  0.215153   69.99      0

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
# V1–V28 are PCA transformed features


# Special Columns (VERY IMPORTANT)
# These are NOT PCA:
# Time → seconds since first transaction
# Amount → transaction amount
# Class → target (0 = normal, 1 = fraud)


# ⏭️ Next Step (VERY IMPORTANT)
# EDA on Amount & Time
# Amount distribution
# sns.histplot(df['Amount'], bins=50)
# plt.title("Transaction Amount Distribution")
# plt.show()
# “Transaction amount is highly right-skewed, huge scale variation, must scale”
# Time distribution
# sns.histplot(df['Time'], bins=50)
# plt.title("Time Distribution")
# plt.show()
# “Time shows periodic behavior, indicating daily transaction cycles rather than a normal distribution.”


# Next Step: Feature Scaling (VERY IMPORTANT)
# must scale Amount
# Time (optional but recommended)
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()

df['scaled_amount'] = scaler.fit_transform(df[['Amount']])
df['scaled_time'] = scaler.fit_transform(df[['Time']])      

# Drop original columns
df.drop(['Amount', 'Time'], axis=1, inplace=True)

# scaled_amount distribution
# sns.histplot(df['scaled_amount'], bins=50)
# plt.title("scaled_amount  Distribution")
# plt.show()

# scaled_time distribution
# sns.histplot(df['scaled_time'], bins=50)
# plt.title("scaled_time Distribution")
# plt.show()


print("df.head() after scaling\n",df.head())

# output:

#          V1        V2        V3        V4        V5        V6        V7  ...       V25       V26       V27       V28  Class  scaled_amount  scaled_time
# 0 -1.359807 -0.072781  2.536347  1.378155 -0.338321  0.462388  0.239599  ...  0.128539 -0.189115  0.133558 -0.021053      0       0.244964    -1.996583
# 1  1.191857  0.266151  0.166480  0.448154  0.060018 -0.082361 -0.078803  ...  0.167170  0.125895 -0.008983  0.014724      0      -0.342475    -1.996583
# 2 -1.358354 -1.340163  1.773209  0.379780 -0.503198  1.800499  0.791461  ... -0.327642 -0.139097 -0.055353 -0.059752      0       1.160686    -1.996562
# 3 -0.966272 -0.185226  1.792993 -0.863291 -0.010309  1.247203  0.237609  ...  0.647376 -0.221929  0.062723  0.061458      0       0.140534    -1.996562
# 4 -1.158233  0.877737  1.548718  0.403034 -0.407193  0.095921  0.592941  ... -0.206010  0.502292  0.219422  0.215153      0      -0.073403    -1.996541

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

print("df.describe() after scaling\n",df.describe())

#  Handling Imbalanced Data (MOST IMPORTANT PART)

# Step 1: Split the data (VERY IMPORTANT)

from sklearn.model_selection import train_test_split

X = df.drop('Class', axis=1)
y = df['Class']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print("y_train.value_counts():\n",y_train.value_counts())
# y_train.value_counts():
#  Class
# 0    227451
# 1       394

print("y_test.value_counts():\n",y_test.value_counts())
# y_test.value_counts():
#  Class
# 0    56864
# 1       98


# Step 3: Apply SMOTE (only on training data)

from imblearn.over_sampling import SMOTE
smote = SMOTE(random_state=42)
X_train_res, y_train_res = smote.fit_resample(X_train, y_train)

# Step 4: Check after SMOTE
print("y_train_res.value_counts():\n",y_train_res.value_counts())


# 🌲 Step 3: Train Model (Start with Random Forest)
print("################################## Start : Train Model Start with Random Forest")
from sklearn.ensemble import RandomForestClassifier
import joblib 
# rf = RandomForestClassifier(
#     n_estimators=50,
#     max_depth=10,
#     random_state=42,
#     n_jobs=-1
# )
# rf.fit(X_train, y_train)
# joblib.dump(rf,"rf_model.pkl")
# print("rf_model.pkl saved")
rf = joblib.load("rf_model.pkl")
# Step 4: Predictions
y_pred_rf = rf.predict(X_test)
y_prob_rf = rf.predict_proba(X_test)[:, 1]
print("y_pred_rf\n",y_pred_rf)
print("y_prob_rf\n",y_prob_rf)

# Step 5: Evaluation (MOST IMPORTANT)
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
print("**********Default thresahold 0.5")
print("confusion_matrix_rf",confusion_matrix(y_test, y_pred_rf))
print("classification_report_rf",classification_report(y_test, y_pred_rf))
print("ROC-AUC_rf:", roc_auc_score(y_test, y_prob_rf))

import numpy as np
threshold = 0.3
print(" thresahold 0.3")
y_pred_custom_rf_T03 = (y_prob_rf > threshold).astype(int)
print("y_pred_custom_rf\n",y_pred_custom_rf_T03)
print("confusion_matrix_rf_T03",confusion_matrix(y_test, y_pred_custom_rf_T03))
print("classification_report_rf_T03",classification_report(y_test, y_pred_custom_rf_T03))
# 👉 Lower threshold → Higher Recall
# 👉 Higher threshold → Higher Precision
print("################################## Completed : Train Model Start with Random Forest")


# 🌲 Step 3: Train Model (Start with Random Forest With class_weight)
print("################################## Started Train Model With Random Forest With class_weight")
from sklearn.ensemble import RandomForestClassifier
# rf_with_max_depth = RandomForestClassifier(
#     n_estimators=50,  # reduce trees (default 100)
#     max_depth=10,  # limit tree depth
#     class_weight='balanced',
#     random_state=42,
#     n_jobs=-1  # use all CPU cores 🚀
# )
# rf_with_max_depth.fit(X_train, y_train)
# joblib.dump(rf_with_max_depth,"rf_with_max_depth_model.pkl")
# print("rf_with_max_depth_model.pkl saved")
rf_with_max_depth = joblib.load("rf_with_max_depth_model.pkl")

# 🔮 Step 4: Predictions
y_pred_rf_with_max_depth = rf_with_max_depth.predict(X_test)
y_prob_rf_with_max_depth = rf_with_max_depth.predict_proba(X_test)[:, 1]
print("y_pred_rf_with_max_depth\n",y_pred_rf_with_max_depth)
print("y_prob_rf_with_max_depth\n",y_prob_rf_with_max_depth)

# 📊 Step 5: Evaluation (MOST IMPORTANT)
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score

print("**********Default thresahold 0.5")
print("confusion_matrix_rf_with_max_depth",confusion_matrix(y_test, y_pred_rf_with_max_depth))
print("classification_report_rf_with_max_depth",classification_report(y_test, y_pred_rf_with_max_depth))
print("ROC-AUC_rf_with_max_depth:", roc_auc_score(y_test, y_prob_rf_with_max_depth))
y_pred_custom_rf_with_max_depth = (y_prob_rf_with_max_depth > threshold).astype(int)
print("y_pred_custom_rf_with_max_depth\n",y_pred_custom_rf_with_max_depth)


threshold = 0.3
print(" ********** thresahold 0.3")
y_pred_custom_rf_with_max_depth_T03 = (y_prob_rf_with_max_depth > threshold).astype(int)
print("y_pred_custom_rf_with_max_depth_T03\n",y_pred_custom_rf_with_max_depth_T03)
print("confusion_matrix_rf_with_max_depth_T03",confusion_matrix(y_test, y_pred_custom_rf_with_max_depth_T03))
print("classification_report_rf_with_max_depth_T03",classification_report(y_test, y_pred_custom_rf_with_max_depth_T03))

print("################################## Completed Train Model With Random Forest With class_weight")


# Apply SMOTE
X_train_sm, y_train_sm = smote.fit_resample(X_train, y_train)

# 🔍 Check new distribution
print("y_train.value_counts()\n",y_train.value_counts())
print("y_train_sm.value_counts()\n",y_train_sm.value_counts())
#should see balanced classes

# Step 3: Train Model on SMOTE data
print("################################## Start : Train Model with Random Forest With SMOTE data")
from sklearn.ensemble import RandomForestClassifier
# rf_smote = RandomForestClassifier(
#     n_estimators=50,
#     max_depth=10,
#     random_state=42,
#     n_jobs=-1
# )
# rf_smote.fit(X_train_sm, y_train_sm)
# joblib.dump(rf_smote,"rf_smote_model.pkl")
# print("rf_smote_model.pkl saved sucessfully")
rf_smote = joblib.load("rf_smote_model.pkl")
# Step 4: Predictions
y_pred_rf_smote = rf_smote.predict(X_test)
y_prob_rf_smote = rf_smote.predict_proba(X_test)[:, 1]
print("y_pred_rf_smote\n",y_pred_rf_smote)
print("y_prob_rf_smote\n",y_prob_rf_smote)

# Step 5: Evaluate (Default threshold first)
print("**********Default thresahold 0.5")
print("confusion_matrix_rf_smote\n",confusion_matrix(y_test, y_pred_rf_smote))
print("classification_report_rf_smote\n",classification_report(y_test, y_pred_rf_smote))
print("ROC-AUC_rf_smote\n", roc_auc_score(y_test, y_prob_rf_smote))

# Step 6: Threshold tuning (same as before)
threshold = 0.3

y_pred_rf_smote_T03 = (y_prob_rf_smote > threshold).astype(int)

print("\nSMOTE - Threshold 0.3")
print("confusion_matrix_rf_smote\n",confusion_matrix(y_test, y_pred_rf_smote_T03))
print("classification_report_rf_smote\n",classification_report(y_test, y_pred_rf_smote_T03))
print("################################## Completed : Train Model with Random Forest With SMOTE data")


print("############### Start Train model using XGBoost")
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, confusion_matrix

# Initialize model
xgb = XGBClassifier(
    n_estimators=100,
    max_depth=5,
    learning_rate=0.1,
    scale_pos_weight=99,   # adjust based on dataset
    random_state=42
)

# Train
# xgb.fit(X_train, y_train)
# joblib.dump(xgb,"xgb_model.pkl")
# print("xgb_model.pkl saved")
xgb= joblib.load("xgb_model.pkl")
# Predict
y_pred = xgb.predict(X_test)

# Evaluate
print("Confusion Matrix_XGB:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Repor_XGBt:\n", classification_report(y_test, y_pred))

print("############### Completed Train model using XGBoost")


print("################ Start train model using XGB GridSearch")

from sklearn.model_selection import GridSearchCV
from xgboost import XGBClassifier
# xgb = XGBClassifier(random_state=42,
#                      n_jobs=1   # 👈 IMPORTANT
#                      )
# param_grid = {
#     'n_estimators': [150, 200],
#     'max_depth': [4, 5],
#     'learning_rate': [0.05, 0.1],
#     'scale_pos_weight': [80, 100]
# }
# xgb_grid_search = GridSearchCV(
#     estimator=xgb,
#     param_grid=param_grid,
#     scoring='f1',   # IMPORTANT for balance
#     cv=3,
#     verbose=2,
#     n_jobs=-1
# )
xgb_grid_search = joblib.load("xgb_grid_search_model.pkl")
# xgb_grid_search.fit(X_train, y_train)
# joblib.dump(xgb_grid_search, "xgb_grid_search_model.pkl")
# print("✅ xgb_grid_search Model saved successfully!")
# best_model_xgb_grid_search = xgb_grid_search.best_estimator_
best_model_xgb_grid_search = joblib.load("best_xgb_model.pkl")
# joblib.dump(best_model_xgb_grid_search, "best_xgb_model.pkl")
# print("✅ best_model_xgb_grid_search Model saved successfully!")
# print("best_model_xgb_grid_search\n",best_model_xgb_grid_search)
# print("Best Parameters_xgb_grid_search\n:", xgb_grid_search.best_params_)

y_pred_xgb_grid_search = best_model_xgb_grid_search.predict(X_test)
y_prob_xgb_grid_search= best_model_xgb_grid_search.predict_proba(X_test)[:, 1]
print("y_pred_xgb_grid_search\n",y_pred_xgb_grid_search)
print("y_prob_xgb_grid_search\n",y_prob_xgb_grid_search)
# Evaluate
from sklearn.metrics import precision_score, recall_score

print("Precision_xgb_grid_search:", precision_score(y_test, y_pred_xgb_grid_search))
print("Recall_xgb_grid_search:", recall_score(y_test, y_pred_xgb_grid_search))
print("Confusion Matrix_xgb_grid_search:\n", confusion_matrix(y_test, y_pred_xgb_grid_search))
print("\nClassification Report_xgb_grid_search:\n", classification_report(y_test, y_pred_xgb_grid_search))
print("ROC-AUC_xgb_grid_search\n", roc_auc_score(y_test, y_prob_xgb_grid_search))
# Try better threshold
for t in [0.5, 0.4, 0.3]:
    y_pred_t = (y_prob_xgb_grid_search > t).astype(int)

    print(f"\n====== Threshold: {t} ======")
    print("Precision:", precision_score(y_test, y_pred_t))
    print("Recall:", recall_score(y_test, y_pred_t))
    print(confusion_matrix(y_test, y_pred_t))
    print(classification_report(y_test, y_pred_t))

print("################ Completed train model using XGB  GridSearch")



print("#### Started train  model using XGB  with best estimators parameters")
# xgb_best = XGBClassifier(
#     n_estimators=200,
#     max_depth=5,
#     learning_rate=0.1,
#     scale_pos_weight=100,
#     random_state=42,
#     n_jobs=-1
# )
# xgb_best.fit(X_train,y_train)
# joblib.dump(xgb_best,"xgb_best_model.pkl")
# print("xgb_best model saved")
xgb_best = joblib.load("xgb_best_model.pkl")
y_pred_xgb_best = xgb_best.predict(X_test)
y_prob_xgb_best = xgb_best.predict_proba(X_test)[:, 1]
# Evaluate
print("Precision_xgb_best:", precision_score(y_test, y_pred_xgb_best))
print("Recall_xgb_best:", recall_score(y_test, y_pred_xgb_best))
print("Confusion Matrix_xgb_best:\n", confusion_matrix(y_test, y_pred_xgb_best))
print("\nClassification Report_xgb_best\n", classification_report(y_test, y_pred_xgb_best))
print("ROC-AUC_xgb_best\n", roc_auc_score(y_test, y_prob_xgb_best))
# ✅ Feature Importance
import pandas as pd
feature_importance_df = pd.DataFrame({
    'Feature': X_train.columns,
    'Importance': xgb_best.feature_importances_
}).sort_values(by='Importance', ascending=False)
pd.set_option('display.max_rows', None)
print("\nTop 10 Feature Importances:\n")
print(feature_importance_df.head(10))

threshold = 0.5
y_pred_xgb_best_05 = (y_prob_xgb_best > threshold).astype(int)
print(f"\n====== Threshold: 0.5 ======")
print("Precision:", precision_score(y_test, y_pred_xgb_best_05))
print("Recall:", recall_score(y_test, y_pred_xgb_best_05))
print(confusion_matrix(y_test, y_pred_xgb_best_05))
print(classification_report(y_test, y_pred_xgb_best_05))

print("####tCompleted rain  model using XGB  GridSearch with only best estimators parameters")


# SHAP 

import shap
import matplotlib.pyplot as plt
plt.clf()
plt.figure(figsize=(10,6))
explainer = shap.Explainer(xgb_best)
shap_values = explainer(X_test[:1000])  # sample for speed
# shap.plots.bar(shap_values, max_display=10)

print("######################################################################################################")
print("################# Start xgb model with gridsearch in pipeline by getting threshold automatically")
# from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


# Use: imblearn Pipeline

# Step-by-Step: SMOTE + Pipeline
# Step 1: Import
from imblearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE

# 🔧 Step 2: Create Pipeline
pipeline = Pipeline([
    ('scaler', StandardScaler()),   # safe to include
    ('smote', SMOTE(sampling_strategy=0.2, random_state=42, k_neighbors=2)),
    ('model', XGBClassifier(random_state=42, n_jobs=1,  
                            tree_method='hist',   # 🔥 MUCH faster
                            max_bin=256
                            ))
])
# 🔧 Step 3: GridSearch with Pipeline

# Key change: use model__ prefix
param_grid = {
    'model__n_estimators': [100],
    'model__max_depth': [4],
    'model__learning_rate': [0.1]
}

# 🔧 Step 4: Train
grid_pipeline = GridSearchCV(
    pipeline,
    param_grid,
    scoring='recall',   # IMPORTANT for fraud
    cv=2,
    verbose=1,
    n_jobs=1
)
# X_train = X_train[:5000].  # partial data
# y_train = y_train[:5000].  # partial data
# grid_pipeline.fit(X_train, y_train)
# joblib.dump(grid_pipeline,"grid_pipeline_model.pkl")
# print("grid_pipeline_model.pkl saved successfully")
grid_pipeline = joblib.load("grid_pipeline_model.pkl")

# 🔧 Step 4: Predictions
best_model_grid_pipeline = grid_pipeline.best_estimator_
y_prob_grid_pipeline = best_model_grid_pipeline.predict_proba(X_test)[:, 1]

# 🚀 3. Automatic Threshold Optimization (🔥 IMPORTANT)
# Instead of manually checking [0.5, 0.4, 0.3], do this:
# 🔧 Step 5: Find Best Threshold
# thresholds_grid_pipeline = np.arange(0.1, 0.9, 0.05)
# best_threshold_grid_pipeline = 0
# best_recall_grid_pipeline = 0
# for t in thresholds_grid_pipeline:
#     y_pred_t = (y_prob_grid_pipeline > t).astype(int)
#     recall = recall_score(y_test, y_pred_t)
#     if recall > best_recall_grid_pipeline:
#         best_recall_grid_pipeline = recall
#         best_threshold_grid_pipeline = t

# for t in thresholds_grid_pipeline:
#     y_pred_t = (y_prob_grid_pipeline > t).astype(int)

#     precision = precision_score(y_test, y_pred_t)
#     recall = recall_score(y_test, y_pred_t)

#     if precision + recall > 0:
#         f1 = 2 * (precision * recall) / (precision + recall)

#         if f1 > best_f1:
#             best_f1 = f1
#             best_threshold = t

# print("Best Threshold (F1):", best_threshold)

# print("Best Threshold_grid_pipeline:", best_threshold_grid_pipeline)
# print("Best Recall_grid_pipeline:", best_recall_grid_pipeline)

from sklearn.metrics import precision_score, recall_score

thresholds = np.arange(0.1, 0.9, 0.05)

best_threshold = 0
best_f1 = 0
best_precision = 0
best_recall = 0

for t in thresholds:
    y_pred_t = (y_prob_grid_pipeline > t).astype(int)

    precision = precision_score(y_test, y_pred_t)
    recall = recall_score(y_test, y_pred_t)

    if precision + recall > 0:
        f1 = 2 * (precision * recall) / (precision + recall)

        if f1 > best_f1:
            best_f1 = f1
            best_threshold = t
            best_precision = precision
            best_recall = recall

print("\nBest Threshold (F1 optimized):", best_threshold)
print("Precision:", best_precision)
print("Recall:", best_recall)
print("F1-score:", best_f1)

# 🔧 Step 6: Final Evaluation
y_pred_final_grid_pipeline = (y_prob_grid_pipeline > best_threshold).astype(int)

print("Confusion_Matrix_grid_pipeline\n",confusion_matrix(y_test, y_pred_final_grid_pipeline))
print("classification_report__grid_pipeline\n",classification_report(y_test, y_pred_final_grid_pipeline))

# 🚀 4. (OPTIONAL BUT 🔥🔥) Precision–Recall Tradeoff
from sklearn.metrics import precision_recall_curve
precision, recall, thresholds = precision_recall_curve(y_test, y_prob_grid_pipeline)


print("################# Completed xgb model with gridsearch in pipeline by getting threshold automatically")

print("######################################################################################################")

# Final Model Comparison Table

# | Model                    | Threshold | Precision | Recall | F1 Score | Key Insight                            |
# | ------------------------ | --------- | --------- | ------ | -------- | -------------------------------------- |
# | Random Forest (Base)     | 0.5       | 0.93      | 0.77   | 0.84     | High precision, missed frauds          |
# | Random Forest            | 0.3       | 0.86      | 0.85   | 0.86     | Balanced after tuning                  |
# | RF + class_weight        | 0.5       | 0.82      | 0.82   | 0.82     | Stable but not optimal                 |
# | RF + SMOTE               | 0.5       | 0.42      | 0.86   | 0.57     | Too many false positives               |
# | XGBoost (Base)           | 0.5       | 0.79      | 0.86   | 0.82     | Strong baseline                        |
# | XGBoost (GridSearch)     | 0.5       | 0.885     | 0.867  | 0.88     | Best raw performance (risk of leakage) |
# | XGBoost Pipeline (Final) | 0.85      | 0.71      | 0.83   | 0.76     | Production-ready, balanced model       |


thresholds = np.arange(0.1, 0.9, 0.05)

precision_list = []
recall_list = []

for t in thresholds:
 y_pred = (y_prob_grid_pipeline > t).astype(int)
 precision_list.append(precision_score(y_test, y_pred))
 recall_list.append(recall_score(y_test, y_pred))

plt.figure()
plt.plot(thresholds, precision_list, label="Precision")
plt.plot(thresholds, recall_list, label="Recall")
plt.xlabel("Threshold")
plt.ylabel("Score")
plt.title("Precision vs Recall vs Threshold")
plt.legend()
# plt.show()