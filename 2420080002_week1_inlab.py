import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 1. Hello World and Name
print("Hello World")
print("My Name is JITHIN SAI")

# 2. Arithmetic Operations
a = int(input("Enter first number: "))
b = int(input("Enter second number: "))

print("Addition:", a + b)
print("Subtraction:", a - b)
print("Multiplication:", a * b)
print("Division:", a / b)

# 3. Load Dataset
data = pd.read_csv("/content/Titanic-Dataset.csv")

# 4. Display First 5 Rows
print("\nFirst 5 Rows:")
print(data.head())

# 5. Dataset Information
print("\nDataset Info:")
print(data.info())

# 6. Statistical Summary
print("\nStatistical Summary:")
print(data.describe())

# 7. Visualization using Matplotlib
data.plot()
plt.title("Dataset Visualization")
plt.xlabel("Index")
plt.ylabel("Values")
plt.legend()
plt.show()

# 8. Visualization using Seaborn
sns.histplot(data=data, x=data.columns[0])
plt.title("Seaborn Histogram")
plt.xlabel("Values")
plt.ylabel("Frequency")
plt.show()
