#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime as dt
from scipy.stats import norm
import yfinance as yf


# In[4]:


years=15
endDate= dt.datetime.now()
startDate=endDate-dt.timedelta(days=365*years)


# In[5]:


tickers=['SPY','BND','GLD','QQQ','VTI']


# In[10]:


adj_close_df=pd.DataFrame()

for ticker in tickers:
    data=yf.download(ticker,start=startDate,end=endDate)
    adj_close_df[ticker]=data['Adj Close']
    
print(adj_close_df)


# In[19]:


log_returns=np.log(adj_close_df/adj_close_df.shift(1))
log_returns=log_returns.dropna()

print(log_returns)


# In[12]:


def expected_return(weights,log_returns):
    return np.sum(log_returns.mean()*weights)


# In[14]:


def std_dev(weights,cov_matrix):
    variance=weights.T @ cov_matrix@ weights
    return np.sqrt(variance)


# In[17]:


cov_matrix=log_returns.cov()
print(cov_matrix)


# In[24]:


portfolio_value=1000000
weights=np.array([1/len(tickers)]*len(tickers))
portfolio_expected_return=expected_return(weights,log_returns)
portfolio_std_dev=std_dev(weights,cov_matrix)


# In[26]:


def random_z_score():
    return np.random.normal(0,1)


# In[39]:


days=20
def scenario_gain_loss(portfolio_value,portfolio_std_dev,random_z_score,days):
    return portfolio_value*portfolio_expected_return*days+portfolio_value*portfolio_std_dev*random_z_score*np.sqrt(days)


# In[40]:


simulations=10000
scenarioReturn=[]

for i in range(simulations):
    z_score=random_z_score()
    scenarioReturn.append(scenario_gain_loss(portfolio_value,portfolio_std_dev,z_score,days))


# In[42]:


confidence_interval=0.95
VaR=-np.percentile(scenarioReturn,100*(1-confidence_interval))
print(VaR)


# In[45]:


plt.hist(scenarioReturn,bins=50,density=True)
plt.xlabel('Scenario Gain/Loss($)')
plt.ylabel('Frequency')
plt.title(f'Distribution of Porfolio gains/Loss over {days} Days')
plt.axvline(-VaR,color='r',linestyle='dashed',linewidth=2,label=f'VaR at{confidence_interval:.0%} confidence level')
plt.legend()
plt.show()

