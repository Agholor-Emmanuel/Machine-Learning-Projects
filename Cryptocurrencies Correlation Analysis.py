# -*- coding: utf-8 -*-
"""
Created on Sun Oct 24 15:16:52 2021

@author: Emmanuel
"""

#Import libraries

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
plt.style.use('ggplot')
from matplotlib.pyplot import figure



#Set our display to show all columns when printed

pd.set_option('display.max_columns', None) 


#Import Historical Datasets for selected coins, create new datasets containing only the closing prices for the coins 
#and change their indexes to a dateTime stamp

#BNB

df_bnb = pd.read_csv(r'C:/Users/Emmanuel/Documents/Data science/PROJECTS/SPYDER PROJECTS/Cryptocurrencies correlation analysis/Datasets/coin_BinanceCoin.csv')
print(df_bnb)
print(df_bnb.dtypes)
df_bnb1 = df_bnb['Close']
df_bnb1.index = pd.to_datetime(df_bnb['Date'])


#BTC

df_btc = pd.read_csv(r'C:/Users/Emmanuel/Documents/Data science/PROJECTS/SPYDER PROJECTS/Cryptocurrencies correlation analysis/Datasets/coin_Bitcoin.csv')
print(df_btc)
df_btc1 = df_btc['Close']
df_btc1.index = pd.to_datetime(df_btc['Date'])


#Dogecoin

df_doge = pd.read_csv(r'C:/Users/Emmanuel/Documents/Data science/PROJECTS/SPYDER PROJECTS/Cryptocurrencies correlation analysis/Datasets/coin_Dogecoin.csv')
print(df_doge)
df_doge1 = df_doge['Close']
df_doge1.index = pd.to_datetime(df_doge['Date'])


#Ethereum

df_eth = pd.read_csv(r'C:/Users/Emmanuel/Documents/Data science/PROJECTS/SPYDER PROJECTS/Cryptocurrencies correlation analysis/Datasets/coin_Ethereum.csv')
print(df_eth)
df_eth1 = df_eth['Close']
df_eth1.index = pd.to_datetime(df_eth['Date'])


#Iota

df_iota = pd.read_csv(r'C:/Users/Emmanuel/Documents/Data science/PROJECTS/SPYDER PROJECTS/Cryptocurrencies correlation analysis/Datasets/coin_Iota.csv')
print(df_iota)
df_iota1 = df_iota['Close']
df_iota1.index = pd.to_datetime(df_iota['Date'])


#Litecoin

df_lite = pd.read_csv(r'C:/Users/Emmanuel/Documents/Data science/PROJECTS/SPYDER PROJECTS/Cryptocurrencies correlation analysis/Datasets/coin_Litecoin.csv')
print(df_lite)
df_lite1 = df_lite['Close']
df_lite1.index = pd.to_datetime(df_lite['Date'])


#Solana

df_sol = pd.read_csv(r'C:/Users/Emmanuel/Documents/Data science/PROJECTS/SPYDER PROJECTS/Cryptocurrencies correlation analysis/Datasets/coin_Solana.csv')
print(df_sol)
df_sol1 = df_sol['Close']
df_sol1.index = pd.to_datetime(df_sol['Date'])


#Tron

df_tron = pd.read_csv(r'C:/Users/Emmanuel/Documents/Data science/PROJECTS/SPYDER PROJECTS/Cryptocurrencies correlation analysis/Datasets/coin_Tron.csv')
print(df_tron)
df_tron1 = df_tron['Close']
df_tron1.index = pd.to_datetime(df_tron['Date'])


#USDT

df_usd = pd.read_csv(r'C:/Users/Emmanuel/Documents/Data science/PROJECTS/SPYDER PROJECTS/Cryptocurrencies correlation analysis/Datasets/coin_USDCoin.csv')
print(df_usd)
df_usd1 = df_usd['Close']
df_usd1.index = pd.to_datetime(df_usd['Date'])


#XRP

df_xrp = pd.read_csv(r'C:/Users/Emmanuel/Documents/Data science/PROJECTS/SPYDER PROJECTS/Cryptocurrencies correlation analysis/Datasets/coin_XRP.csv')
print(df_xrp)
df_xrp1 = df_xrp['Close']
df_xrp1.index = pd.to_datetime(df_xrp['Date'])



#create a new dataset to by using Concat to join all the closing prices of the 10 cryptocoins

df_coins = pd.concat(
    [df_btc1, df_bnb1, df_doge1, df_eth1, df_iota1, df_lite1, df_sol1, df_tron1, df_usd1, df_xrp1], 
    axis=1,
    keys=(['BTC', 'BNB', 'DOGE', 'ETH', 'IOTA', 'LITE', 'SOL', 'TRON', 'USDT', 'XRP']))


print(df_coins)


#Fill all missing data with 0

df_coins['BNB'] = df_coins['BNB'].fillna(0)
df_coins['BTC'] = df_coins['BTC'].fillna(0)
df_coins['DOGE'] = df_coins['DOGE'].fillna(0)
df_coins['ETH'] = df_coins['ETH'].fillna(0)
df_coins['IOTA'] = df_coins['IOTA'].fillna(0)
df_coins['LITE'] = df_coins['LITE'].fillna(0)
df_coins['SOL'] = df_coins['SOL'].fillna(0)
df_coins['TRON'] = df_coins['TRON'].fillna(0)
df_coins['USDT'] = df_coins['USDT'].fillna(0)
df_coins['XRP'] = df_coins['XRP'].fillna(0)

print(df_coins)




#Visualize closing prices for the various coins with BTC included 

plt.figure(figsize=(16,8))
plt.title('Crypto-coins Price History')
plt.plot(df_coins['BTC'])
plt.plot(df_coins['BNB'])
plt.plot(df_coins['DOGE'])
plt.plot(df_coins['ETH'])
plt.plot(df_coins['IOTA'])
plt.plot(df_coins['LITE'])
plt.plot(df_coins['SOL'])
plt.plot(df_coins['TRON'])
plt.plot(df_coins['USDT'])
plt.plot(df_coins['XRP'])
plt.xlabel('Date', fontsize=18)
plt.ylabel('Closing Price (USD)', fontsize=18)
plt.legend(['BTC', 'BNB', 'DOGE', 'ETH', 'IOTA', 'LITE', 'SOL', 'TRON', 'USDT', 'XRP'], loc= 'lower right')
plt.show()



#Visualize closing prices for the various coins without BTC 

plt.figure(figsize=(16,8))
plt.title('Crypto-coins Price History')
plt.plot(df_coins['BNB'])
plt.plot(df_coins['DOGE'])
plt.plot(df_coins['ETH'])
plt.plot(df_coins['IOTA'])
plt.plot(df_coins['LITE'])
plt.plot(df_coins['SOL'])
plt.plot(df_coins['TRON'])
plt.plot(df_coins['USDT'])
plt.plot(df_coins['XRP'])
plt.xlabel('Date', fontsize=18)
plt.ylabel('Closing Price (USD)', fontsize=18)
plt.legend(['BNB', 'DOGE', 'ETH', 'IOTA', 'LITE', 'SOL', 'TRON', 'USDT', 'XRP'], loc= 'lower right')
plt.show()



#Check for Correlation among crypto-coins

df_coins.corr(method = 'pearson')


#Visualize Correlations with Seaborn Heatmap 

correlation_matrix = df_coins.corr(method = 'pearson')
sns.heatmap(correlation_matrix, annot=True)
plt.title('Crypto-currency Correlation Matrix ')
plt.xlabel('Crypto-currencies')
plt.ylabel('Crypto-currencies')
plt.show()

print(correlation_matrix)



#Sort Correlation insights by BTC

sorted_correlation_matrix = correlation_matrix.sort_values(by='BTC', ascending=False)
print((sorted_correlation_matrix))


#Visualize sorted BTC correlation insights


sns.heatmap(sorted_correlation_matrix, annot=True)
plt.title('Crypto-currency Correlation Matrix By BTC ')
plt.xlabel('Crypto-currencies')
plt.ylabel('Crypto-currencies')
plt.show()


#In summary we can conclude that all the 9 coins have good correlation matrixes with BTC so we can expect them to increase whenever BTC increases
#we can conclude that ETH and BNB have the highest correlation with BTC 
#We can also say that Sol and BNB have the highest correlation among all the coins so we can expect an increase in sol when BNB increases
#SOL and Doge is a close second so we can also expect Doge to increase whenever SOL increases
#There are many other insights to get from this project depending on what you are looking for
 



