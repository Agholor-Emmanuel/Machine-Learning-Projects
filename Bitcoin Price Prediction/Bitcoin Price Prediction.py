# -*- coding: utf-8 -*-
"""
Created on Wed Oct 20 07:22:57 2021

@author: Emmanuel
"""


#Import Libraries

import math
import pandas_datareader as web
import numpy as np
import pandas as pd
from pandas import DataFrame
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras import Sequential
from keras.layers import Dense, LSTM
import matplotlib.pyplot as plt
plt.style.use('fivethirtyeight')


#Set our display to show all columns

pd.set_option('display.max_columns', None)


#Get Bitcoin stock quote

df = web.DataReader('BTC-USD', data_source='yahoo', start='2012-01-01', end='2021-10-18')
print(df)



#I noticed that our datasource did not have price entries for 2021-10-18 and 2021-10-19
#so I went ahead to create a dataframe containing the missing data for those 2 days and then append it to our original dataframe

df1 = pd.DataFrame({
    'High' : [62614.66, 64434.53],
    'Low'  : [60012.76, 61622.93],
    'Open' : [61548.80, 62043.17],
    'Close': [62026.08, 64261.99],
    'Volume':[38055562075, 40471196346],
    'Adj Close':[62026.08, 64261.99]
    }, index= ['2021-10-18', '2021-10-19'])

df1.index = pd.to_datetime(df1.index) #convert our index to datetime format so we can append it easily to df

df = df.append(df1) #append the missing data to our original df


#Visualize Bitcoin closing stock price over the selected period

plt.figure(figsize=(16,8))
plt.title('Bitcoin Stock Price History')
plt.plot(df['Close'])
plt.xlabel('Date', fontsize=18)
plt.ylabel('Closing Price (USD)', fontsize=18)
plt.show()


#Create a new datafrane with only bitcoin's closing price and convert new dataframe to numpy array

btc = df.filter(['Close']).values
print(btc)


#Scale our data

scaler = MinMaxScaler(feature_range=(0,1))
scaled_btc = scaler.fit_transform(btc)
print(scaled_btc)



#Get number of rows to train the model on

len_training_data = math.ceil(len(btc)*0.8)
print(len_training_data)


#create training dataset

train_data = scaled_btc[: len_training_data, :]


#split train_data into x_train and y_train

x_train = []
y_train = []

for i in range(60, len(train_data)):
    x_train.append(train_data[i-60:i, 0])
    y_train.append(train_data[i, 0])


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


#create test dataset

test_data = scaled_btc[len_training_data-60: , :]


#split test_data into x_test and y_test

x_test = []
y_test = btc[len_training_data: , :]

for i in range(60, len(test_data)):
    x_test.append(test_data[i-60:i, 0])
  

#Convert x_test to numpy array

x_test = np.array(x_test)


#The LSTM model requires our test data to have 3 dimensions and our test data currently has 2d we have to reshape
#The dimensions for the LSTM model are No of Rows, No of Columns and No of features

x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))
print(x_test.shape)

#Use our model to predict the scaled closing price for x_test
#Inverse transform our predicted scaled cp to get actual cp

predicted_btc = model.predict(x_test)
predicted_btc = scaler.inverse_transform(predicted_btc) 


#check how well our model did by getting the root mean squared error(RMSE)

rmse =np.sqrt(np.mean(((predicted_btc- y_test)**2)))
print(rmse)


#Plot the data
data = df.filter(['Close'])
train = data[: len_training_data]
valid = data[len_training_data:]
valid['Predicted'] = predicted_btc

 
plt.figure(figsize=(16,8))
plt.title('BTC Stock Price Prediction')
plt.plot(train['Close'], color='blue')
plt.plot(valid['Close'], color = 'green')
plt.plot(valid['Predicted'], color = 'red')
plt.xlabel('Date', fontsize=18)
plt.ylabel('Closing Price (USD)', fontsize=18)
plt.legend(['Training Data', 'Actual Price', 'Predicted Price'], loc= 'lower right')
plt.show()

print(valid) #To see both actual and predicted stock prices in a table form


#use the model to predict btc closing price for today 2021-10-20 
    
#Create new dataframe
btc2 = df.filter(['Close'])


#Get the last 70 days closing price value and convert the dataframe to an array

last_60_days = btc2[-60:].values


#Scale our data

scaler = MinMaxScaler(feature_range=(0,1))
scaled_btc2 = scaler.fit_transform(last_60_days)
print(scaled_btc2)


#create an empty x_test2 list and append the values of scaled_cp2 into it

x_test2 = []
x_test2.append(scaled_btc2)


#convert x_test2 to numpy array

x_test2 = np.array(x_test2)


#Get predicted scaled price for oct 1 and then undo the scaling

pred_price = model.predict(x_test2)
pred_price = scaler.inverse_transform(pred_price)
print(pred_price)





