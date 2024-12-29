import numpy as np
from sklearn.model_selection import train_test_split

class SimpleANN:
    def __init__(self, input_size, hidden_size, output_size):
        self.weights1 = np.random.randn(input_size, hidden_size) * 0.01
        self.bias1 = np.zeros((1, hidden_size))
        self.weights2 = np.random.randn(hidden_size, output_size) * 0.01
        self.bias2 = np.zeros((1, output_size))
    
    def relu(self, x):
        return np.maximum(0, x)
    
    def softmax(self, x):
        exps = np.exp(x - np.max(x, axis=1, keepdims=True))
        return exps / np.sum(exps, axis=1, keepdims=True)
    
    def forward(self, X):
        self.z1 = np.dot(X, self.weights1) + self.bias1
        self.a1 = self.relu(self.z1)
        self.z2 = np.dot(self.a1, self.weights2) + self.bias2
        self.a2 = self.softmax(self.z2)
        return self.a2
    
    def backward(self, X, y, output, learning_rate):
        m = X.shape[0]
        one_hot_y = np.zeros((m, output.shape[1]))
        one_hot_y[np.arange(m), y] = 1
        
        dz2 = output - one_hot_y
        dw2 = np.dot(self.a1.T, dz2) / m
        db2 = np.sum(dz2, axis=0, keepdims=True) / m
        
        dz1 = np.dot(dz2, self.weights2.T) * (self.z1 > 0)
        dw1 = np.dot(X.T, dz1) / m
        db1 = np.sum(dz1, axis=0, keepdims=True) / m
        
        self.weights1 -= learning_rate * dw1
        self.bias1 -= learning_rate * db1
        self.weights2 -= learning_rate * dw2
        self.bias2 -= learning_rate * db2
    
    def train(self, X, y, epochs, learning_rate):
        for epoch in range(epochs):
            output = self.forward(X)
            loss = -np.mean(np.log(output[np.arange(len(y)), y]))
            self.backward(X, y, output, learning_rate)
            if epoch % 10 == 0:
                print(f"Epoch {epoch}, Loss: {loss}")
    
    def predict(self, X):
        output = self.forward(X)
        return np.argmax(output, axis=1)

# Load the dataset
data = np.load("features.npy")
labels = np.load("labels.npy")

# Encode labels to integers
unique_labels = np.unique(labels)
label_map = {label: idx for idx, label in enumerate(unique_labels)}
encoded_labels = np.array([label_map[label] for label in labels])


# Stratified split to ensure balanced label distribution
train_data, test_data, train_labels, test_labels = train_test_split(
    data, encoded_labels, test_size=0.2, stratify=encoded_labels, random_state=42
)

# Check the shapes
print(f"Training data shape: {train_data.shape}, Training labels shape: {train_labels.shape}")
print(f"Test data shape: {test_data.shape}, Test labels shape: {test_labels.shape}")


# Train the ANN
input_size = train_data.shape[1]
hidden_size = 64
output_size = len(unique_labels)
print("Number of classes", output_size)

ann = SimpleANN(input_size, hidden_size, output_size)
ann.train(train_data, train_labels, epochs=100, learning_rate=0.1)

# Test the ANN
predictions = ann.predict(test_data)
accuracy = np.mean(predictions == test_labels)
print(f"Test Accuracy: {accuracy * 100:.2f}%")
