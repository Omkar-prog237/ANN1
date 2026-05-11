#Write a python program to show Back Propagation Network for XOR function with Binary Input and Output

import numpy as np

# Sigmoid activation function
def sigmoid(x):

    return 1 / (1 + np.exp(-x))

# Derivative of sigmoid
def sigmoid_derivative(x):

    return x * (1 - x)

# Input dataset (XOR)
X = np.array([
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1]
])

# Output dataset
y = np.array([
    [0],
    [1],
    [1],
    [0]
])

# Seed for reproducibility
np.random.seed(1)

# Initialize weights
input_neurons = 2
hidden_neurons = 2
output_neurons = 1

# Weights
weights_input_hidden = np.random.uniform(
    size=(input_neurons, hidden_neurons)
)

weights_hidden_output = np.random.uniform(
    size=(hidden_neurons, output_neurons)
)

# Biases
bias_hidden = np.random.uniform(size=(1, hidden_neurons))

bias_output = np.random.uniform(size=(1, output_neurons))

# Learning rate
lr = 0.5

# Training
epochs = 10000

for epoch in range(epochs):

    # Forward propagation
    hidden_input = np.dot(X, weights_input_hidden) + bias_hidden

    hidden_output = sigmoid(hidden_input)

    final_input = np.dot(hidden_output, weights_hidden_output) + bias_output

    final_output = sigmoid(final_input)

    # Error
    error = y - final_output

    # Backpropagation
    d_output = error * sigmoid_derivative(final_output)

    error_hidden = d_output.dot(weights_hidden_output.T)

    d_hidden = error_hidden * sigmoid_derivative(hidden_output)

    # Update weights and biases
    weights_hidden_output += hidden_output.T.dot(d_output) * lr

    weights_input_hidden += X.T.dot(d_hidden) * lr

    bias_output += np.sum(d_output, axis=0, keepdims=True) * lr

    bias_hidden += np.sum(d_hidden, axis=0, keepdims=True) * lr

# Testing
print("Final Output after Training:\n")

for i in range(len(X)):

    hidden = sigmoid(
        np.dot(X[i], weights_input_hidden) + bias_hidden
    )

    output = sigmoid(
        np.dot(hidden, weights_hidden_output) + bias_output
    )

    print(
        f"Input: {X[i]} -> Output: {output[0][0]:.4f} "
        f"-> Rounded: {np.round(output[0][0])}"
    )
