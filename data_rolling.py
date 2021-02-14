#%%
import os
import sys
import pandas as pd
import numpy as np
from tqdm.auto import tqdm

HOME = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = HOME+'/models/'
DATA_DIR = HOME+'/data/'

from utils import *
from utils_js import *
# %%
'''
1. Using the past day mean as fillna
2. For certain features use EWM (maybe too slow?)

Past day mean
Reference: Lucas Morin's notebook
https://www.kaggle.com/lucasmorin/running-algos-fe-for-fast-inference?scriptVersionId=50754012
'''

class RunningPDA:
    '''
    https://www.kaggle.com/lucasmorin/running-algos-fe-for-fast-inference?scriptVersionId=50754012
    '''
    def __init__(self, past_mean=0):
        self.day = -1
        self.past_mean = past_mean # past day mean, initialized as the mean
        self.cum_sum = 0
        self.day_instances = 0 # current day instances
        self.past_value = past_mean # the previous row's value, initialized as the mean
        self.past_instances =0 # instances in the past day

    def clear(self):
        self.n = 0
        self.windows.clear()

    def push(self, x, date):
        
        x = fast_fillna(x, self.past_value)
        self.past_value = x
        
        # change of day
        if date > self.day:
            self.day = date
            if self.day_instances > 0:
                self.past_mean = self.cum_sum/self.day_instances
            self.past_instances = self.day_instances
            self.day_instances = 1
            self.cum_sum = x
            
        else:
            self.day_instances += 1
            self.cum_sum += x

    def get_mean(self):
        return self.cum_sum/self.day_instances

    def get_past_mean(self):
        return self.past_mean

    def get_past_trade(self):
        return self.past_instances



#%%
def load_train(drop_days=None):
    with timer("Loading train parquet"):
        train_parquet = os.path.join(DATA_DIR, 'train.parquet')
        train = pd.read_parquet(train_parquet)
        if drop_days:
            train = train.query(f'date not in {drop_days}').reset_index (drop = True)
    return train


def process_train_rolling(train, debug=False):
    TRAIN_ROWS = 50_000
    if debug:
        train = train[:TRAIN_ROWS]

    f_mean = train.mean().values

    train_dtypes = {'date': np.int32,
                    'ts_id': np.int64,
                    'resp': np.float64,
                    'weight': np.float64,
                    }
    for c in range(1,5):
        train_dtypes['resp_'+str(c)] = np.float64
    for c in range(130):
        train_dtypes['feature_'+str(c)] = np.float32

    pdm = RunningPDA(past_mean=f_mean)

    with tqdm(total=len(train)) as pbar:
        row_vals = []
        for _, row in train.iterrows(): 
            date = row['date']
            pdm.push(np.array(row), date)
            
            past_day_mean = pdm.get_past_mean()

            x_tt = row.values
            if np.isnan(x_tt.sum()):
                x_tt = np.nan_to_num(x_tt) + np.isnan(x_tt) * past_day_mean

            row_vals.append(x_tt)
            pbar.update()
    
    train_pdm = pd.DataFrame(row_vals, columns=train.columns, index=train.index).astype(train_dtypes)
    train_pdm.to_parquet('train_pdm.parquet', index=False)

  
# %%

if __name__ == '__main__':
    get_system()
    train = load_train(drop_days=[2, 36, 270, 294])
    process_train_rolling(train, debug=False)