#Implement Artificial Neural Network training process in Python by using Forward Propagation, Back Propagation.

import numpy as np

# activation function
def sigmoid(x):

    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):

    return x * (1 - x)

# -------- Training Data --------
X = np.array([[1, 0, 1]])     # input

y = np.array([[1]])           # output

# -------- Network Structure --------
np.random.seed(2)

input_layer_neurons = 3
hidden_layer_neurons = 4
output_neurons = 1

W1 = np.random.uniform(size=(input_layer_neurons, hidden_layer_neurons))

b1 = np.random.uniform(size=(1, hidden_layer_neurons))

W2 = np.random.uniform(size=(hidden_layer_neurons, output_neurons))

b2 = np.random.uniform(size=(1, output_neurons))

learning_rate = 0.1
epochs = 5000

# -------- Training --------
for i in range(epochs):

    # forward
    hidden_layer_input = np.dot(X, W1) + b1

    hidden_layer_output = sigmoid(hidden_layer_input)

    final_input = np.dot(hidden_layer_output, W2) + b2

    predicted_output = sigmoid(final_input)

    # error
    error = y - predicted_output

    # backprop
    d_output = error * sigmoid_derivative(predicted_output)

    error_hidden = d_output.dot(W2.T)

    d_hidden = error_hidden * sigmoid_derivative(hidden_layer_output)

    # update
    W2 += hidden_layer_output.T.dot(d_output) * learning_rate

    b2 += np.sum(d_output, axis=0, keepdims=True) * learning_rate

    W1 += X.T.dot(d_hidden) * learning_rate

    b1 += np.sum(d_hidden, axis=0, keepdims=True) * learning_rate

# -------- Testing --------
test = np.array([[1, 0, 1]])

hidden = sigmoid(np.dot(test, W1) + b1)

output = sigmoid(np.dot(hidden, W2) + b2)

print("Predicted Output:")
print(output)
