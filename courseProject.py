import numpy as np 
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import confusion_matrix, classification_report
import time

def PrintConfusionMatrix(y_test, y_pred):
    conf_matrix = confusion_matrix(y_test, y_pred)
    print("Confusion Matrix:")
    print(conf_matrix)

if __name__ == "__main__":
  banknote_data = pd.read_csv("data_banknote.csv")

  X = banknote_data.iloc[:, :-1] # All columns except the last one
  y = banknote_data.iloc[:, -1] # Last column

  # Split the data into training and testing sets
  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

  scaler = StandardScaler()
  X_train_scaled = scaler.fit_transform(X_train)
  X_test_scaled = scaler.transform(X_test)

  clf = LinearDiscriminantAnalysis()
  svm_classifier = SVC(kernel='linear')

  #--------------------------------------- Fisher Linear Discriminant---------------------------------------
  # training time
  print("Fisher's Linear Discriminant")
  start_time = time.time()
  clf.fit(X_train_scaled, y_train)
  training_time = time.time() - start_time
  print(f"Training Time: {training_time} seconds")

  # testing time
  start_time = time.time()
  y_pred = clf.predict(X_test_scaled)
  testing_time = time.time() - start_time
  print(f"Testing Time: {testing_time} seconds")

  PrintConfusionMatrix(y_test, y_pred)

  #--------------------------------------- Linear support vector machine------------------------------------
  # training time
  print("\n\nSupport Vector Machine")
  start_time = time.time()
  svm_classifier.fit(X_train_scaled, y_train)
  training_time = time.time() - start_time
  print(f"Training Time: {training_time} seconds")

  # testing time
  start_time = time.time()
  y_pred = svm_classifier.predict(X_test_scaled)
  testing_time = time.time() - start_time
  print(f"Testing Time: {testing_time} seconds")

  PrintConfusionMatrix(y_test, y_pred)