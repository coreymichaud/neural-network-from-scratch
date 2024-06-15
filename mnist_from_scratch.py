# Import libraries
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder

# Load MNIST dataset from sklearn
mnist = fetch_openml('mnist_784', version=1)
X, y = mnist['data'], mnist['target'].astype(int)

# Convert y to numpy array
y = np.array(y)

# Normalize pixel values to be between 0 and 1
X = X / 255.0

# One-hot encode labels
encoder = OneHotEncoder(categories='auto')
y = encoder.fit_transform(y.reshape(-1, 1)).toarray()

# Split data into training and test sets using 80/20 split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Multilayer Perceptron (MLP) Class
class MLP:
    def __init__(self, input_size, hidden_size, output_size):
        self.W1 = np.random.randn(input_size, hidden_size)
        self.b1 = np.zeros((1, hidden_size))
        self.W2 = np.random.randn(hidden_size, output_size)
        self.b2 = np.zeros((1, output_size))

    def sigmoid(self, x):
        return 1.0 / (1.0 + np.exp(-x))

    def softmax(self, x):
        exp_scores = np.exp(x)
        return exp_scores / np.sum(exp_scores, axis=1, keepdims=True)

    def forward(self, X):
        self.z1 = np.dot(X, self.W1) + self.b1
        self.a1 = self.sigmoid(self.z1)
        self.z2 = np.dot(self.a1, self.W2) + self.b2
        self.a2 = self.softmax(self.z2)
        return self.a2

    def backward(self, X, y, learning_rate=0.1):
        m = X.shape[0]
        delta3 = self.a2 - y
        dW2 = np.dot(self.a1.T, delta3)
        db2 = np.sum(delta3, axis=0, keepdims=True)
        delta2 = np.dot(delta3, self.W2.T) * self.a1 * (1 - self.a1)
        dW1 = np.dot(X.T, delta2)
        db1 = np.sum(delta2, axis=0)

        self.W1 -= learning_rate * dW1
        self.b1 -= learning_rate * db1
        self.W2 -= learning_rate * dW2
        self.b2 -= learning_rate * db2

    def predict(self, X):
        return np.argmax(self.forward(X), axis=1)

# Initialize MLP
input_size = X_train.shape[1]
hidden_size = 128
output_size = 10
mlp = MLP(input_size, hidden_size, output_size)

# Training parameters
epochs = 20
batch_size = 64
learning_rate = 0.1

# Training loop
num_batches = X_train.shape[0] // batch_size
accuracy_history = []

for epoch in range(epochs):
    for i in range(num_batches):
        start = i * batch_size
        end = start + batch_size
        X_batch = X_train[start:end]
        y_batch = y_train[start:end]

        # Forward and backward pass
        mlp.forward(X_batch)
        mlp.backward(X_batch, y_batch, learning_rate)

    # Calculate accuracy on the training set
    y_pred = mlp.predict(X_train)
    accuracy = np.mean(y_pred == np.argmax(y_train, axis=1))
    accuracy_history.append(accuracy)
    print(f"Epoch {epoch + 1}/{epochs}, Accuracy: {accuracy}")

# Calculate accuracy on the test set
y_pred_test = mlp.predict(X_test)
test_accuracy = np.mean(y_pred_test == np.argmax(y_test, axis=1))
print(f"Test Accuracy: {test_accuracy}")

# Plot accuracy over epochs
plt.figure(figsize=(10, 6))
plt.plot(range(1, epochs + 1), accuracy_history, marker='o', linestyle='-', color='b', label='Training Accuracy')
plt.title('MLP Training Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.xticks(range(1, epochs + 1))
plt.legend()
plt.grid(True)
plt.show()
