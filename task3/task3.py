# ===============================
# Task 3: Exploratory Data Analysis (EDA)
# Fixed Version (Ready to Submit)
# ===============================

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sns.set(style="whitegrid")

# ===============================
# PART 1: IRIS DATASET EDA
# ===============================

print("\n===== IRIS DATASET EDA =====\n")

# Load Iris dataset
iris = sns.load_dataset("iris")
print(iris.head())

# 1. Distribution of numerical features
iris.hist(figsize=(10, 6))
plt.suptitle("Iris Dataset: Distribution of Numerical Features")
plt.show()

# 2. Categorical feature analysis
plt.figure(figsize=(6, 4))
sns.countplot(x="species", data=iris)
plt.title("Iris Species Count")
plt.show()

# 3. Box plot for outlier detection
plt.figure(figsize=(10, 6))
sns.boxplot(data=iris.select_dtypes(include="number"))
plt.title("Iris Dataset: Box Plot (Outlier Detection)")
plt.show()

# 4. Correlation heatmap (ONLY numerical columns)
plt.figure(figsize=(8, 6))
numeric_iris = iris.select_dtypes(include="number")
sns.heatmap(numeric_iris.corr(), annot=True, cmap="coolwarm")
plt.title("Iris Dataset: Correlation Heatmap")
plt.show()

print("\nIris Dataset Insights:")
print("- Petal length and petal width are highly correlated")
print("- Petal features are most important for prediction")
print("- Sepal width shows presence of outliers")

# ===============================
# PART 2: NETFLIX DATASET EDA
# ===============================

print("\n===== NETFLIX DATASET EDA =====\n")

# Load Netflix dataset
netflix = pd.read_csv("netflix_titles.csv")
print(netflix.head())

# 1. Movies vs TV Shows
plt.figure(figsize=(6, 4))
sns.countplot(x="type", data=netflix)
plt.title("Netflix Content Type Distribution")
plt.show()

# 2. Content release year distribution
plt.figure(figsize=(8, 5))
netflix["release_year"].hist(bins=20)
plt.title("Netflix Content Release Year Distribution")
plt.xlabel("Release Year")
plt.ylabel("Count")
plt.show()

# 3. Rating distribution
plt.figure(figsize=(8, 6))
sns.countplot(
    y="rating",
    data=netflix,
    order=netflix["rating"].value_counts().index
)
plt.title("Netflix Content Rating Distribution")
plt.show()

# 4. Duration analysis
print("\nTop 10 Content Durations:")
print(netflix["duration"].value_counts().head(10))

print("\nNetflix Dataset Insights:")
print("- Movies dominate Netflix content")
print("- Major increase in content after 2015")
print("- TV-MA is the most common rating")
print("- Duration and release year are important features")

# ===============================
# FINAL SUMMARY
# ===============================

print("\n===== FINAL SUMMARY =====")
print("Iris Dataset:")
print("- Clean and balanced dataset")
print("- Petal length & width are best predictors")

print("\nNetflix Dataset:")
print("- Movies are more than TV Shows")
print("- Content surged after 2015")
print("- Adult-rated content dominates")
