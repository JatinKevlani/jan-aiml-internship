# ==========================================
# Task 7: Logistic Regression – Titanic Survival Prediction
# FIXED VERSION (No Warnings, No NaNs)
# ==========================================

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    roc_curve,
    roc_auc_score
)

# ==========================================
# 1. LOAD DATASET
# ==========================================

print("\n===== TASK 7: TITANIC SURVIVAL PREDICTION =====\n")

titanic = sns.load_dataset("titanic")
print(titanic.head())

# ==========================================
# 2. HANDLE MISSING VALUES (CORRECT WAY)
# ==========================================

titanic["age"] = titanic["age"].fillna(titanic["age"].median())
titanic["embarked"] = titanic["embarked"].fillna(titanic["embarked"].mode()[0])

# ==========================================
# 3. DROP UNNECESSARY / HIGH-NaN COLUMNS
# ==========================================

titanic = titanic.drop(
    columns=["deck", "alive", "who", "adult_male", "class", "embark_town"]
)

# Drop any remaining NaNs (SAFE FOR LOGISTIC REGRESSION)
titanic = titanic.dropna()

# ==========================================
# 4. ENCODE CATEGORICAL FEATURES
# ==========================================

titanic = pd.get_dummies(
    titanic,
    columns=["sex", "embarked"],
    drop_first=True
)

# ==========================================
# 5. SCALE NUMERICAL FEATURES
# ==========================================

scaler = StandardScaler()
titanic[["age", "fare"]] = scaler.fit_transform(
    titanic[["age", "fare"]]
)

# ==========================================
# 6. TRAIN-TEST SPLIT (STRATIFIED)
# ==========================================

X = titanic.drop(columns=["survived"])
y = titanic["survived"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("\nTraining samples:", X_train.shape[0])
print("Testing samples :", X_test.shape[0])

# ==========================================
# 7. TRAIN LOGISTIC REGRESSION MODEL
# ==========================================

model = LogisticRegression(max_iter=300)
model.fit(X_train, y_train)

print("\nModel trained successfully")

# ==========================================
# 8. PREDICTION & METRICS
# ==========================================

y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print("\n===== EVALUATION METRICS =====")
print(f"Accuracy  : {accuracy:.2f}")
print(f"Precision : {precision:.2f}")
print(f"Recall    : {recall:.2f}")
print(f"F1-score  : {f1:.2f}")

# ==========================================
# CONFUSION MATRIX
# ==========================================

cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(5, 4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()

# ==========================================
# ROC CURVE & AUC
# ==========================================

fpr, tpr, _ = roc_curve(y_test, y_prob)
auc_score = roc_auc_score(y_test, y_prob)

plt.figure(figsize=(6, 5))
plt.plot(fpr, tpr, label=f"AUC = {auc_score:.2f}")
plt.plot([0, 1], [0, 1], linestyle="--")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend()
plt.show()

print("\nAUC Score:", round(auc_score, 2))
print("\n===== TASK 7 COMPLETED SUCCESSFULLY =====")
