#How to Train a Neural Network with TensorFlow/Pytorch and evaluation of logistic regression using Tensorflow

import tensorflow as tf
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# 1. Load MNIST dataset
(X_train, y_train), (X_test, y_test) = (
    tf.keras.datasets.mnist.load_data()
)

# 2. Flatten images (28x28 → 784)
X_train = X_train.reshape(-1, 28 * 28)

X_test = X_test.reshape(-1, 28 * 28)

# 3. Normalize (0–255 → 0–1)
X_train = X_train / 255.0

X_test = X_test / 255.0

# 4. Build Model (Multiclass Logistic Regression)
model = tf.keras.models.Sequential([
    tf.keras.layers.Dense(
        10,
        activation='softmax',
        input_shape=(784,)
    )
])

# 5. Compile Model
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# 6. Train Model
model.fit(X_train, y_train, epochs=5)

# 7. Predict
y_pred = model.predict(X_test)

# Convert probabilities → class labels
y_pred_classes = np.argmax(y_pred, axis=1)

# 8. Evaluate
test_loss, test_accuracy = model.evaluate(
    X_test,
    y_test
)

print("Test Accuracy:", test_accuracy)
