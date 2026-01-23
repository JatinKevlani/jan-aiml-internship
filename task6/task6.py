# ===============================
# Task 6: Linear Regression – House Price Prediction
# Dataset: California Housing
# ===============================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error

# ===============================
# 1. LOAD DATASET
# ===============================

print("\n===== TASK 6: LINEAR REGRESSION =====\n")

housing = fetch_california_housing()

# Convert to DataFrame
df = pd.DataFrame(housing.data, columns=housing.feature_names)
df["HousePrice"] = housing.target   # Target variable

print("Dataset Loaded Successfully\n")

# ===============================
# 2. BASIC INSPECTION
# ===============================

print("HEAD:")
print(df.head())

print("\nINFO:")
print(df.info())

print("\nDESCRIBE:")
print(df.describe())

# ===============================
# 3. SPLIT FEATURES & TARGET
# ===============================

X = df.drop(columns=["HousePrice"])
y = df["HousePrice"]   # Continuous numeric target

print("\nTarget Variable Type:", y.dtype)

# ===============================
# 4. TRAIN-TEST SPLIT
# ===============================

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42
)

print("\nTraining Samples:", X_train.shape[0])
print("Testing Samples :", X_test.shape[0])

# ===============================
# 5. TRAIN LINEAR REGRESSION MODEL
# ===============================

model = LinearRegression()
model.fit(X_train, y_train)

print("\nLinear Regression Model Trained Successfully")

# ===============================
# 6. PREDICTION
# ===============================

y_pred = model.predict(X_test)

# Actual vs Predicted comparison
comparison = pd.DataFrame({
    "Actual Price": y_test.values[:10],
    "Predicted Price": y_pred[:10]
})

print("\nActual vs Predicted (Sample):")
print(comparison)

# ===============================
# 7. MODEL EVALUATION (MAE & RMSE)
# ===============================

mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))

print("\n===== EVALUATION METRICS =====")
print(f"MAE  : {mae:.3f}")
print(f"RMSE : {rmse:.3f}")

# ===============================
# 8. PREDICTED vs ACTUAL PLOT
# ===============================

plt.figure(figsize=(7, 5))
plt.scatter(y_test, y_pred, alpha=0.5)
plt.xlabel("Actual House Price")
plt.ylabel("Predicted House Price")
plt.title("Predicted vs Actual House Prices")
plt.show()

# ===============================
# 9. MODEL COEFFICIENT INTERPRETATION
# ===============================

coefficients = pd.DataFrame({
    "Feature": X.columns,
    "Coefficient": model.coef_
}).sort_values(by="Coefficient", ascending=False)

print("\n===== FEATURE IMPACT (COEFFICIENTS) =====")
print(coefficients)

print("\n===== TASK 6 COMPLETED SUCCESSFULLY =====")
