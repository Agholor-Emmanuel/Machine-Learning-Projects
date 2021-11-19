# -*- coding: utf-8 -*-
"""
Created on Fri Mar 12 13:15:54 2021

@author: EMMANUEL
"""

#Import Libraries

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


#import dataset and create variable

dataset = pd.read_csv('Shopping_center.csv')
X = dataset.iloc[:,3:5].values


#Using Elbow method to find optimal value of K

from sklearn.cluster import KMeans
WCSS = []
for i in range(1,11):
    kmeans = KMeans(n_clusters=i, init='k-means++', n_init=10, max_iter=300, random_state=0)
    kmeans.fit(X)
    WCSS.append(kmeans.inertia_)
    

#Visualize optimal K  
  
plt.plot(range(1,11), WCSS)
plt.title('Elbow method')
plt.xlabel('number of cluster')
plt.ylabel('WCSS')
plt.show() 


#fit k means to dataset  

kmeans = KMeans(n_clusters=5, init='k-means++', n_init=10, max_iter=300, random_state=0)
Y_kmeans = kmeans.fit_predict(X)


#Visualize

plt.figure(figsize=(8,6))
plt.scatter(X[Y_kmeans==0,0], X[Y_kmeans==0,1], s=100, color='red', label='cluster 1' )
plt.scatter(X[Y_kmeans==1,0], X[Y_kmeans==1,1], s=100, color='blue', label='cluster 2' )
plt.scatter(X[Y_kmeans==2,0], X[Y_kmeans==2,1], s=100, color='green', label='cluster 3' )
plt.scatter(X[Y_kmeans==3,0], X[Y_kmeans==3,1], s=100, color='orange', label='cluster 4' )
plt.scatter(X[Y_kmeans==4,0], X[Y_kmeans==4,1], s=100, color='purple', label='cluster 5' )
plt.scatter(kmeans.cluster_centers_[:,0],kmeans.cluster_centers_[:,1],s=300,color='yellow', label='center')
plt.title('Clustering of customers')
plt.xlabel('Annual Income')
plt.ylabel('Spending points')
plt.legend()
plt.show()


#With these we've been able to divide our customers into four (4) clusters
#High Earners and High Spenders
#High Earners and Low Spenders
#Low Earners and High Spenders
#Low Earners and Low Spenders

#This will enable us give each cluster different discounts, benefits etc.

    
    