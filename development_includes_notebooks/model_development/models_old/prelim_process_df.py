#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import torch 
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sklearn.model_selection import TimeSeriesSplit
import pickle


def save_list(d,file_name):
    with open(file_name,"wb") as fp:
        pickle.dump(d,fp)
        
        
def load_list(file_name):
    with open(file_name,"rb") as fp:
        b = pickle.load(fp)
    return(b)    


def read_process_df(path,col_specific = ''):
    
    """
    Input is a path to a csv file of data. Output will be a dataframe with
    columns - 'open', 'high', 'low', 'close', 'volume', 'pct_change', 'log_ret', 'vol_usdt_K', 'datetime'
    col_specific - suffix to attach to columns
    
    """
        
    df = pd.read_csv(path, names = ['timestamp' ,'open','high','low','close','volume'])
    df = df.drop_duplicates(subset = 'timestamp')
    
    tmp = [x/1000 for x in (df.timestamp)] #timestamps are in milliseconds - change if not
    l = list(map(datetime.fromtimestamp,tmp))
    df['datetime'] = l
    
    df.drop('timestamp', inplace=True, axis=1)

    df = df.set_index('datetime')
    df['returns'] = df.close.pct_change()
    df['log_returns'] = np.log(df.close) - np.log(df.close.shift(1))
    df['vol_usdt_K'] = (((df['open'] + df['close'])/2)*df['volume'])/1000
    df = df.iloc[1:-1]
    
    if len(col_specific) > 0:
        df.columns = [x + '_' + col_specific for x in df.columns]

    return(df)




def add_datetime(df):
    
    """
    
    Input is a dataframe with timestamp as index. Returns a dataframe with a datetime column for readability.
    
    """
    
    tmp = [x/1000 for x in (df.index)]
    l = list(map(datetime.fromtimestamp,tmp))
    
    df['datetime'] = l
    
    return(df)


def resample(df,interval):
    """
    Input : df -- dataframe OHLCV with pandas datetime as index.
            interval -- string eg. 1T,1H,1D,1m 
    
    Output : df resampled. 
    
    In the resample code, T -- minute eg 5T will resample for 5 min intervals,
    H -- hours, D -- days, m -- months.
    """
    
    ohlc_dict = {}
    
    for x in df.columns:
        if x[:3] == 'ope':
            ohlc_dict[x] = 'first'
        if x[:3] == 'hig':
            ohlc_dict[x] = 'max'
        if x[:3] == 'low':
            ohlc_dict[x] = 'min'
        if x[:3] == 'clo':
            ohlc_dict[x] = 'last'
        if x[:3] == 'vol':
            ohlc_dict[x] = 'sum'    
    
    df = df.resample(interval).agg(ohlc_dict)
          
    return(df)


def merge_list_dfs(l):
    """
    merges the dataframes in the list l one at a time assuming there share an index.
    """
    df_temp = l[0].copy()
    for x in l[1:]:
        df_temp = pd.merge(df_temp,x,left_index = True, right_index = True)
    return(df_temp)  

def get_nan_counts(df):
    """
    gets columns with counts of the nan values in each col
    """
    return(df.isna().sum())
    
    
    
    

