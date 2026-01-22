# ===============================
# Task 5: Train-Test Split & Evaluation Metrics
# Model: Logistic Regression
# Dataset: Iris (Processed)
# ===============================

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix, classification_report

# ===============================
# LOAD DATASET
# ===============================

print("\n===== TASK 5: TRAIN-TEST SPLIT & EVALUATION =====\n")

# Load processed Iris dataset
data = pd.read_csv("iris_processed.csv")
print("Dataset Loaded Successfully")
print(data.head())

# ===============================
# FEATURE & TARGET SPLIT
# ===============================

X = data.drop(columns=["species_encoded"])
y = data["species_encoded"]

# ===============================
# TRAIN-TEST SPLIT
# ===============================

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42
)

print("\nTraining Samples:", X_train.shape[0])
print("Testing Samples:", X_test.shape[0])

# ===============================
# MODEL TRAINING
# ===============================

model = LogisticRegression(max_iter=200)
model.fit(X_train, y_train)

print("\nModel Training Completed")

# ===============================
# PREDICTION
# ===============================

y_pred = model.predict(X_test)

# ===============================
# EVALUATION METRICS
# ===============================

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average="macro")
recall = recall_score(y_test, y_pred, average="macro")
conf_matrix = confusion_matrix(y_test, y_pred)

print("\n===== EVALUATION METRICS =====")
print(f"Accuracy  : {accuracy:.2f}")
print(f"Precision : {precision:.2f}")
print(f"Recall    : {recall:.2f}")

print("\nConfusion Matrix:")
print(conf_matrix)

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# ===============================
# INTERPRETATION
# ===============================

print("\n===== RESULT INTERPRETATION =====")
print("- High accuracy indicates good overall performance")
print("- Precision shows correctness of positive predictions")
print("- Recall shows ability to find all relevant classes")
print("- Confusion matrix shows class-wise prediction results")

print("\n===== TASK 5 COMPLETED SUCCESSFULLY =====")
