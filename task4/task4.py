# ===============================
# Task 4: Feature Encoding & Scaling
# Tools: Pandas, NumPy, Scikit-learn
# ===============================

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler

# ===============================
# PART 1: IRIS DATASET
# ===============================

print("\n===== TASK 4: IRIS DATASET =====\n")

# Load Iris dataset
iris = pd.read_csv("https://raw.githubusercontent.com/mwaskom/seaborn-data/master/iris.csv")
print("Original Iris Dataset:")
print(iris.head())

# Identify features
categorical_features = ["species"]
numerical_features = iris.drop(columns=categorical_features).columns.tolist()

print("\nCategorical Features:", categorical_features)
print("Numerical Features:", numerical_features)

# Label Encoding (species has order in ML target context)
label_encoder = LabelEncoder()
iris["species_encoded"] = label_encoder.fit_transform(iris["species"])

# Scaling numerical features
scaler = StandardScaler()
iris_scaled = scaler.fit_transform(iris[numerical_features])

iris_scaled_df = pd.DataFrame(
    iris_scaled,
    columns=numerical_features
)

# Combine encoded + scaled data
iris_final = pd.concat(
    [iris_scaled_df, iris["species_encoded"]],
    axis=1
)

print("\nProcessed Iris Dataset:")
print(iris_final.head())

# Save processed dataset
iris_final.to_csv("iris_processed.csv", index=False)
print("\n✅ iris_processed.csv saved successfully")

# ===============================
# PART 2: NETFLIX DATASET
# ===============================

print("\n===== TASK 4: NETFLIX DATASET =====\n")

# Load Netflix dataset
netflix = pd.read_csv("task3/netflix_titles.csv")
print("Original Netflix Dataset:")
print(netflix.head())

# Select relevant features
netflix_features = netflix[["type", "rating", "release_year"]].dropna()

# Identify categorical & numerical features
categorical_features = ["type", "rating"]
numerical_features = ["release_year"]

print("\nCategorical Features:", categorical_features)
print("Numerical Features:", numerical_features)

# One-Hot Encoding (no order exists)
netflix_encoded = pd.get_dummies(
    netflix_features,
    columns=categorical_features
)

# Scaling numerical features
scaler = StandardScaler()
netflix_encoded[numerical_features] = scaler.fit_transform(
    netflix_encoded[numerical_features]
)

print("\nProcessed Netflix Dataset:")
print(netflix_encoded.head())

# Save processed dataset
netflix_encoded.to_csv("netflix_processed.csv", index=False)
print("\n✅ netflix_processed.csv saved successfully")

# ===============================
# MODEL READINESS COMPARISON
# ===============================

print("\n===== MODEL READINESS COMPARISON =====")
print("Before Scaling:")
print(netflix_features.describe())

print("\nAfter Scaling:")
print(netflix_encoded.describe())

# ===============================
# IMPACT OF SCALING (PRINTED EXPLANATION)
# ===============================

print("\n===== IMPACT OF SCALING ON ML MODELS =====")
print("- Scaling ensures features contribute equally")
print("- Improves convergence speed of Gradient Descent")
print("- Essential for KNN, SVM, Logistic Regression")
print("- Not mandatory for tree-based models")

print("\n===== TASK 4 COMPLETED SUCCESSFULLY =====")
