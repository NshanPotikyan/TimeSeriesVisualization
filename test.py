import numpy as np 
import pandas as pd
import time
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, accuracy_score
from dtw import DTW
from one_nn import OneNN

train = pd.read_csv('datasets/ECG200/ECG200_TRAIN.tsv', 
                   header=None,
                   sep='\t')

test = pd.read_csv('datasets/ECG200/ECG200_TEST.tsv', 
                   header=None,
                   sep='\t')

print('Training set', train.shape)
print('Testing set', test.shape)

train.head() 
# first column is the label
# each row is time series

X_train = train.iloc[:, 1:]
y_train = train.iloc[:, 0]
X_test = test.iloc[:, 1:]
y_test = test.iloc[:, 0]

"""### Visualizing with DTW"""

print('The series have different labels')
DTW(X_train.iloc[0, :], X_train.iloc[1, :]).plot()
print('The series have the same labels')
DTW(X_train.iloc[0, :], X_train.iloc[2, :]).plot()


