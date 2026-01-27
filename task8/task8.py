# ==========================================
# Task 8: Decision Tree – Bank Marketing Prediction
# FIXED VERSION (Dummy + Real Dataset Safe)
# ==========================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, classification_report

# ==========================================
# 1. LOAD DATASET
# ==========================================

print("\n===== TASK 8: BANK MARKETING SUBSCRIPTION PREDICTION =====\n")

data = pd.read_csv("D:/intern/jan-aiml-internship/task8/bank.csv", sep=";")
print(data.head())

# ==========================================
# 2. HANDLE UNKNOWN VALUES
# ==========================================

data = data.dropna(subset=["y"])

# ==========================================
# 3. SEPARATE TARGET (IMPORTANT FIX)
# ==========================================

y = data["y"].map({"yes": 1, "no": 0})   # binary target
X = data.drop(columns=["y"])

# ==========================================
# 4. ENCODE CATEGORICAL FEATURES
# ==========================================

X = pd.get_dummies(X, drop_first=True)

# ==========================================
# 5. TRAIN-TEST SPLIT
# ==========================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

print("Training samples:", X_train.shape[0])
print("Testing samples :", X_test.shape[0])

# ==========================================
# 6. TRAIN DECISION TREE (LIMIT DEPTH)
# ==========================================

model = DecisionTreeClassifier(
    criterion="gini",
    max_depth=4,
    random_state=42
)

model.fit(X_train, y_train)

# ==========================================
# 7. EVALUATION
# ==========================================

train_acc = accuracy_score(y_train, model.predict(X_train))
test_acc = accuracy_score(y_test, model.predict(X_test))

print("\n===== MODEL ACCURACY =====")
print(f"Training Accuracy: {train_acc:.2f}")
print(f"Testing Accuracy : {test_acc:.2f}")

print("\nClassification Report:")
print(classification_report(y_test, model.predict(X_test)))

# ==========================================
# 8. TREE VISUALIZATION
# ==========================================

plt.figure(figsize=(20, 8))
plot_tree(
    model,
    feature_names=X.columns,
    class_names=["No", "Yes"],
    filled=True
)
plt.title("Decision Tree – Bank Marketing")
plt.show()

# ==========================================
# 9. KEY DECISION RULES
# ==========================================

print("\n===== KEY DECISION RULES =====")
print("1. If call duration is high → customer is more likely to subscribe")
print("2. If contact type is cellular → higher success probability")
print("3. Fewer campaign contacts → better subscription chances")

print("\n===== TASK 8 COMPLETED SUCCESSFULLY =====")
