from torch.utils.data import Dataset
from torch import Tensor
import pandas as pd
import numpy as np

class twoday_basic_dset(Dataset):
    
    def __init__(self,df):
        self.X = torch.tensor(df.X.tolist(),dtype = torch.float)
        self.y = torch.tensor(df.y.astype(int).tolist(),dtype = torch.float)
        
    def __getitem__(self,
                    index: int):
        return (self.X[index],self.y[index])
    
    def __len__(self):
        return len(self.X)
    

def get_dloader(df,
                batch_size,
                num_workers = 64,
                shuffle = True):
    
    dset = twoday_basic_dset(df)
    dloader = DataLoader(dset,batch_size = batch_size,num_workers = num_workers, shuffle = shuffle)
    
    return dloader