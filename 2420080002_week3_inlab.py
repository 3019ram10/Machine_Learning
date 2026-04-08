import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA

# Load Iris Dataset
iris = load_iris()
X = iris.data
y = iris.target

# Standardization
scaler = StandardScaler()
X_standard = scaler.fit_transform(X)

# Normalization
normalizer = MinMaxScaler()
X_normalized = normalizer.fit_transform(X)

# Apply PCA (2 Components)
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_standard)

# Explained Variance Ratio
print("Explained Variance Ratio:", pca.explained_variance_ratio_)

# 2D Visualization of PCA
plt.figure()
for i in range(3):
    plt.scatter(X_pca[y == i, 0], X_pca[y == i, 1], label=iris.target_names[i])
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.title("PCA - 2D Visualization")
plt.legend()
plt.show()

# Scree Plot (Cumulative Explained Variance)
pca_full = PCA()
pca_full.fit(X_standard)
cumulative_variance = np.cumsum(pca_full.explained_variance_ratio_)

plt.figure()
plt.plot(cumulative_variance, marker='o')
plt.xlabel("Number of Components")
plt.ylabel("Cumulative Explained Variance")
plt.title("Scree Plot")
plt.show()

# Original Data Visualization
plt.figure()
plt.scatter(X[:,0], X[:,1], c=y)
plt.xlabel("Sepal Length")
plt.ylabel("Sepal Width")
plt.title("Original Data Visualization")
plt.show()