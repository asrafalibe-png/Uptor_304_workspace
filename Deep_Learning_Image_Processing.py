import tensorflow as tf
from tensorflow.keras import datasets, layers, models

"""pip install tensorflow"""

import matplotlib.pyplot as plt

# 1. Load MNIST dataset
(x_train, y_train), (x_test, y_test) = datasets.mnist.load_data()

# 2. Normalize pixel values (0–255 → 0–1)
x_train, x_test = x_train / 255.0, x_test / 255.0


# 3. Build the deep learning model
model = models.Sequential([
    layers.Flatten(input_shape=(28, 28)),          # Flatten 28x28 images → 784 vector
    layers.Dense(128, activation='relu'),          # Hidden layer
    layers.Dropout(0.2),                           # Dropout for regularization
    layers.Dense(64, activation='relu'),           # Another hidden layer
    layers.Dense(10, activation='softmax')         # Output layer (10 classes)
])

# 4. Compile the model
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# 5. Train the model
history = model.fit(x_train, y_train, epochs=5, validation_data=(x_test, y_test))

# 6. Evaluate the model
test_loss, test_acc = model.evaluate(x_test, y_test, verbose=2)
print(f"\nTest accuracy: {test_acc:.4f}")

print(history.history['accuracy'])
print(history.history['val_accuracy'])

# 7. Plot training history
plt.plot(history.history['accuracy'], label='train acc')
plt.plot(history.history['val_accuracy'], label='val acc')
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()
plt.show()

# 8. Make predictions on first 5 test samples
predictions = model.predict(x_test[:5])
print("Predicted labels:", predictions.argmax(axis=1))
print("True labels:", y_test[:5])
