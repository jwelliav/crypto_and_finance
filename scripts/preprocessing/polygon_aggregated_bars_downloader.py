from polygon import RESTClient
from local_settings import polygon as settings
from tqdm.notebook import tqdm
from joblib import Parallel,delayed
from polygon import RESTClient
import pandas as pd
import numpy as np
import os

# change if necessary - 
apikey = "wBkWOy90budzobSlu_fSbwiTrf9JVb20"

client = RESTClient(apikey)

import pickle

def save(l,filename):
    with open(filename, 'wb') as f:
         pickle.dump(l, f)
            
            
def load(filename):
    with open(filename, 'rb') as g:
         l = pickle.load(g)
    return l

# current plan is limited to 10 years of data - but we query from 2010 onwards. 

def parse_agg(x):
    """
    x is an object made by polygon with aggregated information over a time period of a stocks        
    performance
    """
    e = {}
    e['open'] = x.open
    e['high'] = x.high
    e['low'] = x.low
    e['close'] = x.close
    e['volume'] = x.volume
    e['vwap'] = x.vwap
    e['timestamp'] = x.timestamp
    e['transactions'] = x.transactions
    e['otc'] = x.otc
    return e

def parse_as_df(agg_list: list):
    
    d = {}
    x = agg_list[0]
    e = parse_agg(x)
    
    for k in e.keys():
        d[k] = []
        
    for x in (agg_list):
        e = parse_agg(x)
        for k in e.keys():
            d[k].append(e[k])
            
    df = pd.DataFrame.from_dict(d)
    
    df.index = df.timestamp
    df.drop('timestamp',axis = 1,inplace = True)
    
    return df

def get_all_data(ticker: str,
                 output_dir:str,
                 granularity: str = 'minute'): # granularity hasn't been implemented yet
    
    start_date = 1262304000000
    end_date = 1680001367000
    
    interval = 60000*60*24*30  #adjust this if you choose granularity greater than 1 minute
    
    tmp = start_date
    l = []
    
    holes = []
    
    for tmp in tqdm(range(start_date,end_date,interval)):
        try:
            aggs = client.get_aggs(
                ticker,
                1,
                "minute",
                tmp,
                tmp+interval,limit = 50000)
            l += aggs
        except:
            holes.append((tmp,tmp+interval))
        
    source_dir = output_dir + '/{}'.format(ticker)
    
    if os.path.exists(source_dir) == False:
        os.mkdir(source_dir) 
    
    df = parse_as_df(l)
    save(df,source_dir + '/{}.parquet'.format(ticker))
    save(holes,source_dir + '/holes')
    
        


