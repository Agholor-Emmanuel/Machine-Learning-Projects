# -*- coding: utf-8 -*-
"""
Created on Mon Oct 18 22:21:21 2021

@author: Emmanuel
"""

#Import Libraries

import math
import pandas_datareader as web
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras import Sequential
from keras.layers import Dense, LSTM
import matplotlib.pyplot as plt
plt.style.use('fivethirtyeight')


#Set our display to show all columns

pd.set_option('display.max_columns', None)


#Get the stock quote

df = web.DataReader('AAPL', data_source='yahoo', start='2012-01-01', end='2021-09-30')
print(df)
print(df.shape) #shows the shape of the dataframe to know how many enteries we have


#Visualize the closing prices of the stock over the period

plt.figure(figsize=(16,8))
plt.title('Apple Stock Closing Price History')
plt.plot(df['Close'])
plt.xlabel('Date', fontsize=18)
plt.ylabel('Closing Price (USD)', fontsize=18)
plt.show()


#Create a new datafrane with only the closing price and convert new dataframe to numpy array

data = df.filter(['Close'])
cp = data.values
print(cp)


#Scale our data

scaler = MinMaxScaler(feature_range=(0,1))
scaled_cp = scaler.fit_transform(cp)
print(scaled_cp)


#Get number of rows to train the model on

len_training_data = math.ceil(len(cp) * 0.8)
print(len_training_data)


#create training dataset

train_data = scaled_cp[:len_training_data, :]


#Split train_data into x_train and y_train datasets

x_train = []
y_train = []

for i in range(70, len(train_data)):
    x_train.append(train_data[i-70:i, 0])
    y_train.append(train_data[i,0])
    if i <=71:
        print(x_train)
        print('\n', y_train)
        print()
 
        
 #Convert x_train and y_train to numpy array
 
x_train, y_train = np.array(x_train), np.array(y_train)


#The LSTM model requires our train data to have 3 dimensions and our train data currently has 2d we have to reshape
#The dimensions for the LSTM model are No of Rows, No of Columns and No of features
 
x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
print(x_train.shape)


#Build the LSTM Model

model = Sequential()
model.add(LSTM(50, return_sequences= True, input_shape = (x_train.shape[1], 1)))
model.add(LSTM(50, return_sequences= False))
model.add(Dense(25))
model.add(Dense(1))


#compile the model

model.compile(optimizer='adam', loss=('mean_squared_error'))


#Train the model

model.fit(x_train, y_train, batch_size=(1), epochs=1)



#Create Testing dataset

test_data = scaled_cp[len_training_data-70: , :]

#create x_test and y_test

x_test = []
y_test = cp[len_training_data: , : ]
for i in range(70, len(test_data)):
    x_test.append(test_data[i-70:i, :])
    

#Convert x_test to numpy array

x_test = np.array(x_test)


#The LSTM model requires our train data to have 3 dimensions and our train data currently has 2d we have to reshape
#The dimensions for the LSTM model are No of Rows, No of Columns and No of features
#x_test = np.reshape(x_test(x_test.shape[0], x_test.shape[1], 1))

print(x_test.shape)


#Use our model to predict the scaled closing price for x_test
#Inverse transform our predicted scaled cp to get actual cp

predicted_cp = model.predict(x_test)
predicted_cp = scaler.inverse_transform(predicted_cp)   


#check how well our model did by getting the root mean squared error(RMSE)

rmse =np.sqrt(np.mean(((predicted_cp- y_test)**2)))
print(rmse)


#Plot the data

train = data[: len_training_data]
valid = data[len_training_data:]
valid['Predicted'] = predicted_cp

 
plt.figure(figsize=(16,8))
plt.title('ML Stock Price Prediction')
plt.plot(train['Close'], color='blue')
plt.plot(valid['Close'], color = 'green')
plt.plot(valid['Predicted'], color = 'red')
plt.xlabel('Date', fontsize=18)
plt.ylabel('Closing Price (USD)', fontsize=18)
plt.legend(['Training Data', 'Actual Price', 'Predicted Price'], loc= 'lower right')
plt.show()

print(valid) #To see both actual and predicted stock prices in a table form



#use the model predict the stock price for October 1, 2021 which is not part of our dataset
    
#Create new dataframe
cp2 = df.filter(['Close'])


#Get the last 70 days closing price value and convert the dataframe to an array

last_70_days = cp2[-70:].values


#Scale our data

scaler = MinMaxScaler(feature_range=(0,1))
scaled_cp2 = scaler.fit_transform(last_70_days)
print(scaled_cp2)


#create an empty x_test2 list and append the values of scaled_cp2 into it

x_test2 = []
x_test2.append(scaled_cp2)


#convert x_test2 to numpy array

x_test2 = np.array(x_test2)


#Get predicted scaled price for oct 1 and then undo the scaling

pred_price = model.predict(x_test2)
pred_price = scaler.inverse_transform(pred_price)
print(pred_price)


#Get actual price to compare with predicted price

df2 = web.DataReader('AAPL', data_source='yahoo', start='2021-10-01', end='2021-10-01')
print(df2['Close'])



