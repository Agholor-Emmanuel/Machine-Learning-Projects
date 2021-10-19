# -*- coding: utf-8 -*-
"""
Created on Tue Mar  9 17:26:31 2021

@author: EMMANUEL
"""

#import libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#Import datasets
dataset = pd.read_csv('Customer List.csv')
X = dataset.iloc[:,2:4].values
Y = dataset.iloc[:,4].values

#Split into train and test sets
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X,Y,test_size=0.25, random_state=0)

#Feature scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.fit_transform(X_test)

#Fit K-NN to Training data
from sklearn.neighbors import KNeighborsClassifier
classifier = KNeighborsClassifier(n_neighbors=5, p=2, metric='minkowski')
classifier.fit(X_train,Y_train)

#Predict results
Y_pred = classifier.predict(X_test)

#Check individual Probability
var_prob = classifier.predict_proba(X_test)

#See probability of individual line
var_prob[0,:]

#calculate accuracy by building confusion matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(Y_test,Y_pred)
#Accurac rate = correct results / total results
#Error rate = wrong results / total results

#Type1 error: when we predict an event will occur and it doesnt occur
#Type2 error: when we predict an event will not occur and it occurs
#Type2 errors are more dangerous than type1 errors
