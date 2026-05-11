#Write a python program in python program for creating a Back Propagation Feed-forward neural network.

import numpy as np

# Activation Function (Sigmoid)
def sigmoid(x):

    return 1 / (1 + np.exp(-x))

# Derivative of Sigmoid
def sigmoid_derivative(x):

    return x * (1 - x)

# Sample Input Data (can be modified)
X = np.array([
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1]
])

# Expected Output (Example: AND function)
y = np.array([
    [0],
    [0],
    [0],
    [1]
])

# Network Structure
input_neurons = 2
hidden_neurons = 3
output_neurons = 1

# Initialize weights and biases
np.random.seed(42)

weights_input_hidden = np.random.rand(
    input_neurons,
    hidden_neurons
)

weights_hidden_output = np.random.rand(
    hidden_neurons,
    output_neurons
)

bias_hidden = np.random.rand(1, hidden_neurons)

bias_output = np.random.rand(1, output_neurons)

# Learning rate
learning_rate = 0.5

# Training process
epochs = 10000

for epoch in range(epochs):

    # Forward Pass
    hidden_input = (
        np.dot(X, weights_input_hidden)
        + bias_hidden
    )

    hidden_output = sigmoid(hidden_input)

    final_input = (
        np.dot(hidden_output, weights_hidden_output)
        + bias_output
    )

    final_output = sigmoid(final_input)

    # Error Calculation
    error = y - final_output

    # Backpropagation
    d_output = error * sigmoid_derivative(final_output)

    error_hidden = d_output.dot(
        weights_hidden_output.T
    )

    d_hidden = (
        error_hidden
        * sigmoid_derivative(hidden_output)
    )

    # Update weights and biases
    weights_hidden_output += (
        hidden_output.T.dot(d_output)
        * learning_rate
    )

    weights_input_hidden += (
        X.T.dot(d_hidden)
        * learning_rate
    )

    bias_output += (
        np.sum(d_output, axis=0, keepdims=True)
        * learning_rate
    )

    bias_hidden += (
        np.sum(d_hidden, axis=0, keepdims=True)
        * learning_rate
    )

# Testing the network
print("Final Outputs:\n")

for i in range(len(X)):

    hidden = sigmoid(
        np.dot(X[i], weights_input_hidden)
        + bias_hidden
    )

    output = sigmoid(
        np.dot(hidden, weights_hidden_output)
        + bias_output
    )

    print(
        f"Input: {X[i]} -> Output: {output[0][0]:.2f}"
    )
