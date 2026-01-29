# ==========================================
# Task 9: Random Forest – Credit Card Fraud Detection
# Pandas-Free Version (Python 3.14 SAFE)
# ==========================================

import csv
import numpy as np
import matplotlib.pyplot as plt
import joblib

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

# ==========================================
# 1. LOAD CSV USING BUILT-IN CSV MODULE
# ==========================================

print("\n===== TASK 9: CREDIT CARD FRAUD DETECTION =====\n")

X = []
y = []

with open("D:/intern/jan-aiml-internship/task9/creditcard.csv", "r") as file:
    reader = csv.reader(file)
    header = next(reader)

    class_index = header.index("Class")

    for row in reader:
        y.append(int(row[class_index]))
        X.append([float(val) for i, val in enumerate(row) if i != class_index])

X = np.array(X)
y = np.array(y)

print("Dataset loaded successfully")
print("Total samples:", len(y))

# ==========================================
# 2. CHECK CLASS IMBALANCE
# ==========================================

unique, counts = np.unique(y, return_counts=True)
print("\nFraud vs Non-Fraud Counts:")
for u, c in zip(unique, counts):
    print(f"Class {u}: {c}")

# ==========================================
# 3. TRAIN-TEST SPLIT (STRATIFIED)
# ==========================================

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# ==========================================
# 4. BASELINE MODEL – LOGISTIC REGRESSION
# ==========================================

baseline = LogisticRegression(max_iter=500)
baseline.fit(X_train, y_train)
baseline_preds = baseline.predict(X_test)

print("\n===== BASELINE MODEL (LOGISTIC REGRESSION) =====")
print(classification_report(y_test, baseline_preds))

# ==========================================
# 5. RANDOM FOREST MODEL
# ==========================================

rf = RandomForestClassifier(
    n_estimators=100,
    random_state=42,
    class_weight="balanced"
)

rf.fit(X_train, y_train)
rf_preds = rf.predict(X_test)

print("\n===== RANDOM FOREST MODEL =====")
print(classification_report(y_test, rf_preds))

# ==========================================
# 6. FEATURE IMPORTANCE PLOT
# ==========================================

importances = rf.feature_importances_
indices = np.argsort(importances)[-10:]

plt.figure(figsize=(8, 5))
plt.barh(range(len(indices)), importances[indices])
plt.yticks(range(len(indices)), indices)
plt.xlabel("Importance")
plt.title("Top Feature Importances")
plt.show()

# ==========================================
# 7. SAVE MODEL
# ==========================================

joblib.dump(rf, "fraud_random_forest.pkl")
print("\nModel saved as fraud_random_forest.pkl")

print("\n===== TASK 9 COMPLETED SUCCESSFULLY =====")
