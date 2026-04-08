import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import make_regression, load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

# ---------------------------
# PART A: LINEAR REGRESSION
# ---------------------------

# Create synthetic regression dataset
X, y = make_regression(n_samples=100, n_features=1, noise=10, random_state=42)

# Train Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Model
lr = LinearRegression()
lr.fit(X_train, y_train)

# Predictions
y_pred = lr.predict(X_test)

# Performance Metrics
print("MSE:", mean_squared_error(y_test, y_pred))
print("MAE:", mean_absolute_error(y_test, y_pred))
print("R2 Score:", r2_score(y_test, y_pred))

# Coefficients
print("Coefficient:", lr.coef_)
print("Intercept:", lr.intercept_)

# Plot Regression Line
plt.scatter(X_test, y_test)
plt.plot(X_test, y_pred)
plt.title("Linear Regression")
plt.xlabel("Feature")
plt.ylabel("Target")
plt.show()

# ---------------------------
# PART B: LOGISTIC REGRESSION
# ---------------------------

# Load Breast Cancer Dataset
data = load_breast_cancer()
X = data.data
y = data.target

# Train Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Model
log_model = LogisticRegression(max_iter=5000)
log_model.fit(X_train, y_train)

# Predictions
y_pred = log_model.predict(X_test)

# Performance Metrics
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Precision:", precision_score(y_test, y_pred))
print("Recall:", recall_score(y_test, y_pred))
print("F1 Score:", f1_score(y_test, y_pred))

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)

sns.heatmap(cm, annot=True, fmt='d')
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

# Model Coefficients
print("Logistic Regression Coefficients:")
print(log_model.coef_)