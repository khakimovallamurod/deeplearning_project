import csv 
import os
import numpy as np
import random
from sklearn.metrics import f1_score, accuracy_score

class SimplePerceptron:
    def __init__(self, margin = 0.1, learning_rate = 0.01, max_iter = 1000):
        self.margin = margin
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.weights=None
        self.bias=0

    def load_data(self, data_file):
        # Implement the method to load data from a CSV file
        if not os.path.isfile(data_file):
            return NotImplementedError("Load data method not implemented. ")
        
        data = []
        with open(data_file, mode='r', newline='', encoding='utf-8') as file:
            reader = csv.reader(file)
            header = next(reader)  # Skip the header line
            for row in reader:
                # Convert features (first 10 columns) to floats
                x = np.array([float(val) for val in row[:-1]])  # Features (f1 to f10)
                
                # Convert the label (last column) to 0 or 1
                y = 1 if row[-1] == 'Y' else 0  # Label ('Y' -> 1, 'N' -> 0)
                
                data.append((x, y))
        return data
    
    def fit (self, train_file):
        # Implement the algorithm to fit the model to the training data
        if not os.path.isfile(train_file):
            return NotImplementedError('Fit method not implemented. ')
        
        # Load data
        data = self.load_data(train_file)
        
        # Initialize weight vector w and bias b to zero
        num_features = len(data[0][0])  # Assume the first example contains all features
        self.weights = np.zeros(num_features)
        self.bias = 0
        
        # Training loop for max_iter iterations
        for t in range(self.max_iter):
            random.shuffle(data)  # Randomly shuffle training examples
            
            # Iterate over each training example (xi, yi)
            for xi, yi in data:
                # Compute f = w · xi + b
                f = np.dot(self.weights, xi) + self.bias
                
                # Update the weight and bias based on the conditions
                if yi == 1 and f < self.margin:
                    self.weights += self.learning_rate * xi
                    self.bias += self.learning_rate
                
                elif yi == 0 and f >= -self.margin:
                    self.weights -= self.learning_rate * xi
                    self.bias -= self.learning_rate
    
    def predict(self, X):
        # Implement the method to make predictions based on the learned weights and bias
        
        f = np.dot(X, self.weights) + self.bias
        return 1 if f>=0 else 0  # Return 1 if score >= 0, else 0
    
    def calculate_scores(self, test_file):
        # Implement the method to calculate accuracy , weighted F1 , and macro F1 scores
        # calls the predict() method to generate predictions
        if not os.path.isfile(test_file):
            return NotImplementedError('Calculate scores method not implemented. ')
        
        data = self.load_data(test_file)
        # Initialize variables for true labels and predicted labels
        true_labels = []
        predicted_labels = []
        
        # Loop through each test example and make predictions
        for xi, yi in data:
            true_labels.append(yi)
            predicted_labels.append(self.predict(xi))
        
        # Calculate accuracy
        accuracy = accuracy_score(true_labels, predicted_labels)
        
        # Calculate weighted F1 score (supports binary classification)
        weighted_f1 = f1_score(true_labels, predicted_labels, average='weighted')
        
        # Calculate macro F1 score
        macro_f1 = f1_score(true_labels, predicted_labels, average='macro')
        
        return accuracy, weighted_f1, macro_f1
        
# Create a SimplePerceptron instance
perceptron = SimplePerceptron(margin=0.1, learning_rate=0.01, max_iter=1000)

# Fit the model on training data from a CSV file
perceptron.fit('q1_train_data1.csv')

# Calculate the performance metrics on test data from a CSV file
accuracy, weighted_f1, macro_f1 = perceptron.calculate_scores('q1_test_data1.csv')

# Print the results
print(f"Accuracy: {accuracy}")
print(f"Weighted F1: {weighted_f1}")
print(f"Macro F1: {macro_f1}")
