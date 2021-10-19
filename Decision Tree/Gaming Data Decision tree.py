# -*- coding: utf-8 -*-
"""
Created on Sun Mar  7 23:41:31 2021

@author: EMMANUEL
"""

#Decision Tree
#Import library
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#import dataset and create variables
dataset = pd.read_csv('Gaming_data.csv')
X = dataset.iloc[:,0:1].values
Y = dataset.iloc[:,1:2].values

#Fit Decision tree regressor to dataset
from sklearn.tree import DecisionTreeRegressor
regressor = DecisionTreeRegressor(random_state=0)
regressor.fit(X,Y)

#Visualize results
plt.scatter(X,Y)
plt.plot(X, regressor.predict(X), color='red')
plt.title('Gaming Data(Decision Tree)')
plt.xlabel('Gaming steps')
plt.ylabel('Gaming points')
plt.show()

#Predict values
Y_pred = regressor.predict([[7.5]])

#Visualize in Higher Resolution
X_grid = np.arange(min(X), max(X), 0.01)
X_grid = X_grid.reshape(len(X_grid),1)
plt.scatter(X,Y)
plt.plot(X_grid, regressor.predict(X_grid), color='red')
plt.title('Gaming Data(Decision Tree)')
plt.xlabel('Gaming steps')
plt.ylabel('Gaming points')
plt.show()

#Note: Decision tree algorithm is a non linear and non continuous algorithm
#hence it takes averages of points and gives them the same results
#example 6.6-7.5 same result, 7.6-8.5 another result etc 
#that is why the graph is different and not a continuous graph like Linear, polynomial, svr etc 


         