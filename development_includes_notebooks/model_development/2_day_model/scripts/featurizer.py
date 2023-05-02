import pandas as pd
import numpy as np
import os
import os.path as osp
from tqdm.notebook import tqdm
import sys
sys.path.append('/home/jwelliav/Documents/crypto_and_finance/scripts/preprocessing')
import preprocessing


def featurize(df:pd.DataFrame,
             resample_granularity: str = '120min'):
    df['dollar_volume'] = df['volume']* ((df['open'] + df['close'])/2)
     
    tmp = preprocessing.resample(df,resample_granularity)
    
    
    a0 = tmp.returns.values.T.tolist()
    a1 = tmp.dollar_volume.values.T.tolist()
    a2 = [tmp.returns.std()] # risk measured over the data 
    a3 = [tmp.dollar_volume.std()] 
    a4 = [(-df.open.iloc[0] + df.high.max())/(df.open.iloc[0])]
    a5 = [(df.high.max() - df.low.min())/(df.low.min())]
    
    l = np.array(a0 + a1 + a2 + a3 + a4 + a5)
    
    return l


def get_y(df: pd.DataFrame,
         threshold: float = 0.01,
         stop_loss: float = 0.1):
    
    b = pd.Timedelta('5m')
    buy_price = df.iloc[0].close
    tmp = df.loc[df.index[0] + b:]
    
    t  = (tmp.close > ((1 + threshold) * buy_price))
    s  = (tmp.close < ((1 - stop_loss) * buy_price))
    
    if t.sum() > 0:
        sell_index = np.where(t)[0][0]
        no_collapse_before_sell = (s[:sell_index + 1].sum() == 0)
        
        return no_collapse_before_sell
        
    else:
        
        return False
    
    
def load_and_featurize(path:str):
    """
    Restrict to just the two day strategy
    """
    
    df = preprocessing.load_df(path)
    keys = df[(df.index.hour == 23) & (df.index.minute == 59)].index.tolist()
    
    e = {}

    for x in tqdm(keys):
        a = pd.Timedelta('5D') 
        b = pd.Timedelta('2D')
        c = pd.Timedelta('10m')
        X_chunk = df.loc[(df.index >= x-a) & (df.index <= x-c)]
        y_chunk = df.loc[(df.index >= x) & (df.index <= x+b)]
        e[x] = (X_chunk,y_chunk)
        
    mode1,mode2 = pd.Series([(len(e[k][0]),len(e[k][1])) for k in e.keys()]).value_counts().index[0]    
        
    # restrict to some of the data where we have same size data

    A = [((len(e[k][0]) == mode1) and (len(e[k][1]) == mode2)) for k in keys]

    print("We make use of {}% of the data".format(np.array(A).sum()*100/len(keys))) 
    
    
    featurized = {}

    for i,k in tqdm(enumerate(keys)):
        if A[i]:
            
            A0= featurize(e[k][0])
            B0= get_y(e[k][1])
            
            featurized[k] = (A0,B0)
        
    feat_keys = list(featurized.keys())
    feat_keys.sort()
    
    
    X = []
    y = []
    
    for f in feat_keys:
        X.append(featurized[f][0])
        y.append(featurized[f][1])
    
    tmp_dict = {}
    tmp_dict['X'] = X
    tmp_dict['y'] = y
    tmp_dict['timestamp'] = feat_keys
    
    return pd.DataFrame.from_dict(tmp_dict)
    
    
    #X = np.array(X)
    #y = np.array(y).astype(int)    
    #    
    #
    #return X,y
    
    