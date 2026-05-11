#Implementation of MNIST Handwritten Character Detection using PyTorch, Keras and Tensorflow

import tensorflow as tf

from tensorflow.keras.datasets import mnist

from tensorflow.keras.models import Sequential

from tensorflow.keras.layers import Dense, Flatten

from tensorflow.keras.optimizers import Adam

# 1. Load dataset
(X_train, y_train), (X_test, y_test) = mnist.load_data()

# 2. Normalize data (0–255 → 0–1)
X_train = X_train / 255.0

X_test = X_test / 255.0

# 3. Define model architecture
model = Sequential([

    Flatten(input_shape=(28, 28)),

    Dense(
        128,
        activation='relu'
    ),

    Dense(
        10,
        activation='softmax'
    )
])

# 4. Compile model
model.compile(
    optimizer=Adam(learning_rate=0.001),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# 5. Train model
model.fit(
    X_train,
    y_train,
    batch_size=64,
    epochs=10,
    verbose=1
)

# 6. Evaluate model
loss, accuracy = model.evaluate(
    X_test,
    y_test
)

# 7. Print results
print("Test Loss:", loss)

print("Test Accuracy:", accuracy)
