# Linear Support Vector Machine

import numpy as np 
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, classification_report
import time


banknote_data = pd.read_csv("data_banknote.csv")

print(banknote_data.isnull().sum())

X = banknote_data.iloc[:, :-1] # All columns except the last one
y = banknote_data.iloc[:, -1] # Last column

print(X.shape)
print(y.shape)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


svm_classifier = SVC(kernel='linear')
svm_classifier.fit(X_train_scaled, y_train)

y_pred = svm_classifier.predict(X_test_scaled)

# Confusion Matrix
conf_matrix = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:")
print(conf_matrix)

classification_rep = classification_report(y_test, y_pred)
print("Classification Report:")
print(classification_rep)

# training time
start_time = time.time()
svm_classifier.fit(X_train_scaled, y_train)
training_time = time.time() - start_time
print(f"Training Time: {training_time} seconds")

# testing time
start_time = time.time()
y_pred = svm_classifier.predict(X_test_scaled)
testing_time = time.time() - start_time
print(f"Testing Time: {testing_time} seconds")



# The confusion matrix is a table that shows the number of true positives, true negatives, false positives, and false negatives. In your case, the matrix indicates that:

# There are 185 true negatives (TN): Instances that are correctly classified as class 0 (authentic banknotes).
# There are 152 true positives (TP): Instances that are correctly classified as class 1 (counterfeit banknotes).
# There are 6 false positives (FP): Instances that are incorrectly classified as class 1 but are actually class 0.
# There are 0 false negatives (FN): Instances that are incorrectly classified as class 0 but are actually class 1.

# The classification report provides various performance metrics for each class and overall:

# - Precision: The ratio of true positive predictions to the total predicted positives. A high precision indicates few false positives.
# - Recall (Sensitivity): The ratio of true positive predictions to the total actual positives. A high recall indicates few false negatives.
# - F1-score: The harmonic mean of precision and recall. It provides a balanced measure between precision and recall.
# - Support: The number of instances in each class.
# - Accuracy: The ratio of correct predictions to the total number of instances.
# - Macro avg: The average of precision, recall, and F1-score for both classes, with equal weight.
# - Weighted avg: The weighted average of precision, recall, and F1-score considering the number of instances in each class.

