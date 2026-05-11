#Write a Python program to plot a few activation functions that are being used in neural networks.

import numpy as np
import matplotlib.pyplot as plt

# Input range
x = np.linspace(-10, 10, 400)

# Activation Functions
linear = x

step = np.where(x >= 0, 1, 0)

sigmoid = 1 / (1 + np.exp(-x))

tanh = np.tanh(x)

relu = np.maximum(0, x)

leaky_relu = np.where(x > 0, x, 0.01 * x)

# Softmax (applied on vector)
softmax = np.exp(x) / np.sum(np.exp(x))

# Plotting
plt.figure(figsize=(12, 10))

plt.subplot(3, 3, 1)
plt.plot(x, linear)
plt.title("Linear")

plt.subplot(3, 3, 2)
plt.plot(x, step)
plt.title("Step")

plt.subplot(3, 3, 3)
plt.plot(x, sigmoid)
plt.title("Sigmoid")

plt.subplot(3, 3, 4)
plt.plot(x, tanh)
plt.title("Tanh")

plt.subplot(3, 3, 5)
plt.plot(x, relu)
plt.title("ReLU")

plt.subplot(3, 3, 6)
plt.plot(x, leaky_relu)
plt.title("Leaky ReLU")

plt.subplot(3, 3, 7)
plt.plot(x, softmax)
plt.title("Softmax")

plt.tight_layout()
plt.show()
