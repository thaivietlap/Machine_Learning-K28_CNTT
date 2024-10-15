# Import libraries
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from DecisionTree import DecisionTreeClass  # Custom implementation
from RandomForest import RandomForest  # Custom implementation
from sklearn.metrics import accuracy_score

# Load dataset
file_path = 'drug200.csv'
data = pd.read_csv(file_path)

# Preprocessing: Map categorical variables to numerical ones
data['Sex'] = data['Sex'].map({'M': 0, 'F': 1})
data['BP'] = data['BP'].map({'HIGH': 2, 'NORMAL': 1, 'LOW': 0})
data['Cholesterol'] = data['Cholesterol'].map({'HIGH': 1, 'NORMAL': 0})
data['Drug'] = data['Drug'].map({'drugA': 0, 'drugB': 1, 'drugC': 2, 'drugX': 3, 'DrugY': 4})

# Create features (X) and target (y)
X = data[['Age', 'Sex', 'BP', 'Cholesterol', 'Na_to_K']]
y = data['Drug']

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define and fit the DecisionTree model
decisionTree = DecisionTreeClass(min_samples_split=2, max_depth=10)
decisionTree.fit(X_train, y_train)
y_pred_dt = decisionTree.predict(X_test)

# Calculate accuracy for DecisionTree
accuracy_dt = np.mean(y_test == y_pred_dt)
print(f'Accuracy of Decision Tree: {accuracy_dt:.2f}')

# Define and fit the RandomForest model
randomForest = RandomForest(n_trees=3, n_features=4)  # You can adjust n_trees and n_features
randomForest.fit(X_train, y_train)
y_pred_rf = randomForest.predict(X_test)

# Calculate accuracy for RandomForest
accuracy_rf = np.mean(y_test == y_pred_rf)
print(f'Accuracy of Random Forest: {accuracy_rf:.2f}')
