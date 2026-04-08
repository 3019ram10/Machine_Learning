import numpy as np
import matplotlib.pyplot as plt

from sklearn import svm
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score


# -------------------------------
# Load Real Dataset
# -------------------------------

data = load_iris()
X = data.data
y = data.target

# Feature Scaling
scaler = StandardScaler()
X = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)


# -------------------------------
# Multi-class SVM (One-vs-Rest)
# -------------------------------

svm_model = svm.SVC(kernel='rbf', decision_function_shape='ovr')
svm_model.fit(X_train, y_train)

pred = svm_model.predict(X_test)

print("OVR Accuracy:", accuracy_score(y_test, pred))


# -------------------------------
# Multi-class SVM (One-vs-One)
# -------------------------------

svm_model2 = svm.SVC(kernel='rbf', decision_function_shape='ovo')
svm_model2.fit(X_train, y_train)

pred2 = svm_model2.predict(X_test)

print("OVO Accuracy:", accuracy_score(y_test, pred2))


# -------------------------------
# Hyperparameter Tuning
# -------------------------------

param_grid = {
    'C':[0.1,1,10],
    'gamma':[0.01,0.1,1],
    'kernel':['rbf']
}

grid = GridSearchCV(svm.SVC(), param_grid, cv=5)

grid.fit(X_train, y_train)

print("Best Parameters:", grid.best_params_)
print("Best Score:", grid.best_score_)


# -------------------------------
# Custom Kernel Function
# -------------------------------

def custom_kernel(X, Y):
    return np.dot(X, Y.T) + 1

svm_custom = svm.SVC(kernel=custom_kernel)

svm_custom.fit(X_train, y_train)

pred3 = svm_custom.predict(X_test)

print("Custom Kernel Accuracy:", accuracy_score(y_test, pred3))