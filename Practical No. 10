#Write a python program to design a Hopfield Network which stores 4 vectors

import numpy as np

# Convert binary (0,1) to bipolar (-1,1)
def to_bipolar(x):

    return np.where(x == 0, -1, 1)

# Convert bipolar back to binary
def to_binary(x):

    return np.where(x == -1, 0, 1)

# Training patterns (4 vectors)
patterns = np.array([
    [1, 0, 1, 0],
    [1, 1, 0, 0],
    [0, 1, 0, 1],
    [0, 0, 1, 1]
])

# Convert to bipolar
patterns = np.array([
    to_bipolar(p) for p in patterns
])

# Number of neurons
n = patterns.shape[1]

# Initialize weight matrix
W = np.zeros((n, n))

# Hebbian Learning Rule
for p in patterns:

    W += np.outer(p, p)

# Set diagonal to zero
np.fill_diagonal(W, 0)

print("Weight Matrix:\n", W)

# Test with a noisy pattern
test_pattern = np.array([1, 0, 1, 1])   # noisy version

test_pattern = to_bipolar(test_pattern)

print("\nInitial Input:", to_binary(test_pattern))

# Recall process
for _ in range(5):   # iterations

    for i in range(n):

        net = np.dot(W[i], test_pattern)

        test_pattern[i] = 1 if net >= 0 else -1

print("Recovered Pattern:", to_binary(test_pattern))
