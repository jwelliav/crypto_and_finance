#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np 
import os
import sys
import matplotlib.pyplot as plt
import datetime
from tqdm.notebook import tqdm
import plotly.express as px
import plotly.graph_objects as go
import statistics 
from statistics import mode 
import math


# In[2]:


def load_df(path):
    """
    input : 
           path - string which indicates where csv file is located. 
           
    output :
           df - convert csv to df and counts the total NaN in the dataframe. 
    """
    if path.endswith('csv'):
        df = pd.read_csv(path)
    elif path.endswith('parquet'):
         df = pd.read_parquet(path) 
        
    df.columns = ['timestamp','open','high','low','close','volume']
    df['date'] = pd.to_datetime(df['timestamp'], unit = 'ms')
    df = df.set_index(pd.DatetimeIndex(df['date']))
    df = df.drop(['date'],axis = 1)        
    df['returns'] = df['close'].pct_change()
    print('Total NaN : ' + "\n")
    print(df.isna().sum())
    print('')
    df = df.drop_duplicates(subset = 'timestamp')
    return(df)


# In[3]:


def check_for_missing_timestamps(df):
    """
    input : df - dataframe with the OHLCV data. 
           
    output : count - total number of holes in the data. By hole we mean an interval for which there is no data. 
             
             missing_indices - list of indices which are the starting points for missing data. 
    """
    
    m = len(df)
    X = df['timestamp'].values
    count = 0
    missing_indices = []
    
    
    l_temp = [X[i]-X[i-1] for i in range(1,len(X))]
    interval_length = mode(l_temp)    # We assume that we more or less have all the data. 
    
    
    for i in range(1,m):
        if X[i]-X[i-1] != interval_length:
            count += 1
            error = np.timedelta64(X[i]-X[i-1],'ms')
            missing_indices.append((i,error))
    
    t = (np.array(missing_indices).sum())
    t = t.astype('int')
    t = t/60000 
    
    print('Number of instances for which we have intervals of missing data is {}.'.format(count))
    print('')
    print('Total amount of missing time in dataframe is {} minutes'.format(t))
    
    return(count,missing_indices)


# In[4]:


def resample(df,interval):
    """
    Input : df -- dataframe OHLCV with pandas datetime as index.
            interval -- string eg. 1T,1H,1D,1m 
    
    Output : df resampled. 
    
    In the resample code, T -- minute eg 5T will resample for 5 min intervals,
    H -- hours, D -- days, m -- months.
    """
    
    ohlc_dict = {
        'open':'first',
        'high':'max',
        'low':'min',
        'close':'last',
        'volume':'sum'
        }
    
    for c in df.columns:
        if 'volume' in c:
            ohlc_dict[c] = 'sum'
    
    df = df.resample(interval).agg(ohlc_dict)
    
    df['returns'] = df['close'].pct_change()
    
    df = df.dropna(axis = 0)
    
    return(df)



def data_preprocessing(in_data, sample_rate='1min'):
    
    """
    Input: 
        - in_data: Dataframe with OHLCV data, indexed by datetime and with columns including 
          timestamp (unix time in lm), O, H, L, C, V. Indices should be with 1min frequency 
          without missing values.
        - sample_rate: the sampling rate used to down-sample. E.g '1h' for one hour.
    """
    
    # impute missing values
    out_data= in_data.loc[:,['timestamp', 'O', 'H', 'L', 'C', 'V']].copy()
    out_data.loc[:,'C']=out_data.loc[:,'C'].fillna(method='ffill')
    out_data.loc[:,'V']=out_data.loc[:,'V'].fillna(0)
    out_data=out_data.fillna(method='bfill',axis=0)
    
    # resample
    out_data=out_data.resample(sample_rate,origin='start',label='left', closed='left').agg(
                                                            {'timestamp':'first',
                                                             'O':'first', 
                                                             'H':'max', 
                                                             'L':'min', 
                                                             'C':'last', 
                                                             'V':'sum'})
    out_data.drop(index=out_data.index[0], inplace=True)
    time_delta= (out_data.timestamp[1]-out_data.timestamp[0])/60000
    ### CMNT : What if (out_data.timestamp[1]-out_data.timestamp[0])/60000 is 
    ### different for the rest of the data. 
    
            
    # add normalized volume
    days_30=int(43200/time_delta)
    log_volume=out_data.V.map(lambda x:log(x+1e-10))
    rolling_25=log_volume.rolling(days_30).quantile(0.25).fillna(method='bfill')
    rolling_50=log_volume.rolling(days_30).median().fillna(method='bfill')
    rolling_75=log_volume.rolling(days_30).quantile(0.75).fillna(method='bfill')
    log_volume_normal=(log_volume-rolling_50)/(rolling_75-rolling_25)
    out_data.loc[:,'normalized_volume']=log_volume_normal
    
    # add log returns (of closing prices)
    out_data.loc[:,'C_log_return']=out_data.C.map(log).diff()
    
    # normalize open, high, and low with respect to previous closing price
    log_C=out_data.C.map(log).shift(1)
    for col in ['O','H','L']:        
        out_data.loc[:,'log_'+col]= out_data.loc[:,col].map(log)-log_C
        
    # add two columns encoding the time of day
    day= 60000*60*24
    daytime= (out_data.timestamp % day)/day*2*pi
    out_data.loc[:,'time_of_day_1']= daytime.map(cos)
    out_data.loc[:,'time_of_day_2']= daytime.map(sin)   
    
    # add columns with rolling means
    for time in [3*60,12*60,24*60,3*24*60, 7*24*60 ]:
        if time>=2*time_delta:
            window=int(time/time_delta)
            out_data.loc[:,'rolling_returns_mean_'+str(time)]=out_data.C_log_return.rolling(window).mean()
        
    # add the target. At the moment this is the log_H column one time-step ahead.
    out_data.loc[:,'target']=out_data.log_H.shift(-1)
    
    # delete rows with missing entries and the O, H, L, C, V columns
    out_data.dropna(inplace=True, axis=0)
    out_data.drop(columns=['O','H','L','C','V','timestamp'], inplace=True)
    
    return out_data

