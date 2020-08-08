#this is the stock market prediction code using a very simple concept of moving average

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
tcs=pd.read_csv('Tcs.csv',index_col="Date")#this is the data of 1 year of tcs company

print(tcs.head())#printing and checking the data 
print(tcs.tail())#printing and checking the data 

#tcs.shape
print(tcs.describe())

tcs_2019=tcs.loc['2019-01-01':'2019-12-30']#format should be same as dataset

print(tcs_2019.loc['2019-07-23'])#for a particular point

print(tcs.iloc[0,0])

print(tcs.iloc[1,1])#access by location 
#and loc is accessing full row

plt.figure(figsize=(12,10))
tcs['Close'].plot()#if nothing given so since index col is date so plot will be plotted against date 
plt.show()

tcs.head()

tcs['Price diff']=tcs['Close'].shift(-1)-tcs['Close']
#shifting the close column one day up and one dat after 
#now this will give me per day profit 

tcs.head()
#now if we want profit over a few days take cumulative sum

tcs.tail()

#now we are gonna create a profit percentage column
tcs['Return']=tcs['Price diff']/tcs['Close']#this is percentage change 
tcs.head()

tcs['ma50']=tcs['Close'].rolling(50).mean()#this is moving average
plt.figure(figsize=(12,8))
tcs['ma50'].plot(label='MA50')
tcs['Close'].plot(label='Close')
plt.legend()
plt.show()



#below is the 1 year data 

hdfc=pd.read_csv('hdfc.csv',index_col="Date")
hdfc['MA10']=hdfc['Close'].rolling(10).mean()
hdfc['MA50']=hdfc['Close'].rolling(50).mean()
hdfc=hdfc.dropna()#all null entries will be removed
hdfc.head()

plt.figure(figsize=(10,8))
hdfc['MA10'].plot(label='MA10')
hdfc['MA50'].plot(label='MA50')
hdfc['Close'].plot(label='Close')
plt.legend()
plt.show()

#if moving average of 10 days is more than the moving average of 50 days then 1
#else do nothing
hdfc['Shares']=[1 if hdfc.loc[x,'MA10']>hdfc.loc[x,'MA50'] else 0 for x in hdfc.index]

hdfc['Close1']=hdfc['Close'].shift(-1)
hdfc['Profit']=[hdfc.loc[x,'Close1']-hdfc.loc[x,'Close'] if hdfc.loc[x,"Shares"]==1 else 0 for x in hdfc.index]

hdfc['wealth']=hdfc['Profit'].cumsum()

hdfc.tail()

hdfc['wealth'].plot()
plt.title('Total money u gain is {}'.format)
plt.show()


# below is the data for microsoft 5 years

microsoft=pd.read_csv("microsoft.csv",index_col="Date")
microsoft['MA200']=microsoft['Close'].rolling(200).mean()
microsoft=microsoft.dropna()#all null entries will be removed
microsoft.head()


plt.figure(figsize=(10,8))
microsoft['MA200'].plot(label='MA200')
microsoft['Close'].plot(label='Close')
plt.legend()
plt.show()

#if moving average of 100 days is more than the moving average of 200 days then 1 means sell 
microsoft['Shares']=[1 if microsoft.loc[x,'Close']>microsoft.loc[x,'MA200'] else 0 for x in microsoft.index]

microsoft['Close1']=microsoft['Close'].shift(-1)
microsoft['Profit']=[microsoft.loc[x,'Close1']-microsoft.loc[x,'Close'] if microsoft.loc[x,"Shares"]==1 else 0 for x in microsoft.index]

microsoft['wealth']=microsoft['Profit'].cumsum()
microsoft.tail()

microsoft['wealth'].plot()
plt.title('Total money u gain is {}'.format(microsoft.loc[microsoft.index[-2],'wealth']))
plt.show()

