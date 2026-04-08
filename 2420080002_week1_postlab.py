
import matplotlib.pyplot as plt
import pandas as pd
import time

# 1. Sum of Digits
n = int(input("Enter a number: "))
temp = n
sum_digits = 0

while temp > 0:
    digit = temp % 10
    sum_digits += digit
    temp //= 10

print("Sum of digits:", sum_digits)

# 2. Multiplication Table
num = int(input("Enter number for multiplication table: "))
for i in range(1, 11):
    print(num, "x", i, "=", num * i)

# 3. Pie Chart
labels = ['Category A', 'Category B', 'Category C', 'Category D']
values = [25, 30, 20, 25]

plt.pie(values, labels=labels, autopct='%1.1f%%')
plt.title("Category Distribution")
plt.show()

# 4. Compare Data Loading Methods
start = time.time()
data1 = pd.read_csv("/content/Titanic-Dataset.csv")
print("read_csv time:", time.time() - start)

start = time.time()
data2 = pd.read_table("/content/Titanic-Dataset.csv", delimiter=",")
print("read_table time:", time.time() - start)

# 5. Error Handling for File Operations
try:
    data = pd.read_csv("/content/Titanic-Dataset.csv")
    print("File loaded successfully")
    print(data.head())
except FileNotFoundError:
    print("File not found")
except Exception as e:
    print("Error:", e)