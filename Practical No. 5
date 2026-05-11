#Write a python Program for Bidirectional Associative Memory with two pairs of vectors.

import numpy as np

# --------- Training Pairs (Bipolar Form) ---------

# Pair 1
X1 = np.array([1, -1, 1])
Y1 = np.array([1, -1])

# Pair 2
X2 = np.array([-1, 1, -1])
Y2 = np.array([-1, 1])

# --------- Weight Matrix Formation ---------
W = np.outer(X1, Y1) + np.outer(X2, Y2)

print("Weight Matrix W:")
print(W)

# --------- Activation Function (Sign) ---------
def sign(v):

    return np.where(v >= 0, 1, -1)

# --------- Recall Y from X ---------
def recall_Y(X):

    Yin = np.dot(X, W)

    Y = sign(Yin)

    return Y

# --------- Recall X from Y ---------
def recall_X(Y):

    Xin = np.dot(Y, W.T)

    X = sign(Xin)

    return X

# --------- Testing ---------
print("\nRecall Process")

test_X = np.array([1, -1, 1])

Y_out = recall_Y(test_X)

print("Input X:", test_X)
print("Recalled Y:", Y_out)

test_Y = np.array([-1, 1])

X_out = recall_X(test_Y)

print("\nInput Y:", test_Y)
print("Recalled X:", X_out)
