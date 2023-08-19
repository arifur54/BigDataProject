import numpy as np 
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import confusion_matrix, classification_report
import time

df = pd.read_csv('training.csv',header=None)

X = np.array(df.iloc[:, :-1].values)  # All columns except the last one
y = np.array(df.iloc[:, -1].values)   # Last column (class labels)
clf = LinearDiscriminantAnalysis()
start_t = time.time() # Start recording the time
clf.fit(X,y)
LinearDiscriminantAnalysis()
end_t = time.time() # End record the time
elapsed_time = end_t-start_t
print("Time taken for training: ", elapsed_time, "seconds")

test_df = pd.read_csv('test.csv',header=None) # Read the csv file
test_X = np.array(test_df.iloc[:,:-1].values) # All columns except the last one
test_y = np.array(test_df.iloc[:, -1].values)   # Last column (class labels)
output =np.array([])
error=0
false_p=0
false_n = 0
for num in range(len(test_X)):
    out_y=clf.predict([[test_X[num,0],test_X[num,1], test_X[num,2], test_X[num,3]]])
    if(out_y!=test_y[num]):
        error+=1
        if (out_y==1):
            false_p+=1
        else:
            false_n+=1
    output=np.append(output,y)

print("error= ",(error/len(test_y))*100,"%")
print("Number of false Positives : ", false_p)
print("Number of false Negatives: ", false_n)