#Program using Perceptron Neural Network to recognize even and odd numbers. Given numbers are in ASCII from 0 to 9

import numpy as np

# 4-bit binary representation of digits 0–9
X = np.array([
    [0, 0, 0, 0],   # 0
    [0, 0, 0, 1],   # 1
    [0, 0, 1, 0],   # 2
    [0, 0, 1, 1],   # 3
    [0, 1, 0, 0],   # 4
    [0, 1, 0, 1],   # 5
    [0, 1, 1, 0],   # 6
    [0, 1, 1, 1],   # 7
    [1, 0, 0, 0],   # 8
    [1, 0, 0, 1]    # 9
])

# Target (Even = 1, Odd = 0)
T = np.array([1, 0, 1, 0, 1, 0, 1, 0, 1, 0])

# Initialize weights and bias
w = np.zeros(4)
b = 0
lr = 0.2

# Training
for epoch in range(25):

    for i in range(len(X)):

        net = np.dot(X[i], w) + b

        y = 1 if net >= 0 else 0

        error = T[i] - y

        w = w + lr * error * X[i]

        b = b + lr * error

print("Training Finished")
print("Weights =", w)
print("Bias =", b)

# Testing
print("\nDigit  Output")

for i in range(len(X)):

    net = np.dot(X[i], w) + b

    y = 1 if net >= 0 else 0

    print(i, "   Even" if y == 1 else "   Odd")
