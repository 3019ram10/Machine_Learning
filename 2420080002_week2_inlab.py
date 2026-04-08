import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# Load Titanic Dataset
data = pd.read_csv("/content/Titanic-Dataset.csv")

# Basic Information
print("Rows and Columns:", data.shape)
print("\nMissing Values:")
print(data.isnull().sum())

# Handle Missing Values
data['Age'] = data['Age'].fillna(data['Age'].median())
data['Embarked'] = data['Embarked'].fillna(data['Embarked'].mode()[0])

# Drop Irrelevant Columns
data.drop(['Cabin', 'Name', 'Ticket', 'PassengerId'], axis=1, inplace=True)

# Detect Outliers in Fare using IQR
Q1 = data['Fare'].quantile(0.25)
Q3 = data['Fare'].quantile(0.75)
IQR = Q3 - Q1

lower = Q1 - 1.5 * IQR
upper = Q3 + 1.5 * IQR

data = data[(data['Fare'] >= lower) & (data['Fare'] <= upper)]

# One Hot Encoding
data = pd.get_dummies(data, columns=['Sex', 'Embarked'])

# Label Encoding (if any remaining categorical columns)
le = LabelEncoder()
for col in data.select_dtypes(include=['object']).columns:
    data[col] = le.fit_transform(data[col])

# Split Features and Target
X = data.drop('Survived', axis=1)
y = data['Survived']

# Train (70%) and Temp (30%)
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)

# Validation (15%) and Test (15%)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

# Print Shapes
print("Training set:", X_train.shape)
print("Validation set:", X_val.shape)
print("Testing set:", X_test.shape)
