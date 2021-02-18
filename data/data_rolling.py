#%%
import os
import sys
import pandas as pd
import numpy as np
from tqdm.auto import tqdm
from collections import deque
import collections

current_path = os.path.dirname(os.path.abspath(__file__))
HOME = os.path.dirname(current_path)
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
    Past day mean
    Reference: Lucas Morin
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

class RunningEWMean:
    '''
    Reference: Lucas Morin
    https://www.kaggle.com/lucasmorin/running-algos-fe-for-fast-inference?scriptVersionId=50754012
    '''
    def __init__(self, win_size=20, n_size = 1):
        self.s = np.zeros(n_size)
        self.past_value = 0
        self.alpha = 2 /(win_size + 1)

    def clear(self):
        self.s = 0

    def push(self, x):
        
        x = fast_fillna(x, self.past_value)
        self.past_value = x
        self.s = self.alpha * x + (1 - self.alpha) * self.s
        
    def get_mean(self):
        return self.s


class RunningMean:
    def __init__(self, win_size=20, n_size = 1):
        self.n = 0
        self.mean = np.zeros(n_size)
        self.n_size=n_size
        self.cum_sum = 0
        self.past_value = 0
        self.win_size = win_size
        self.windows = collections.deque(maxlen=win_size+1)
        
    def clear(self):
        self.n = 0
        self.windows.clear()

    def push(self, x):
        
        x = fast_fillna(x, self.past_value)
        self.past_value = x
        
        self.windows.append(x)
        self.cum_sum += x
        
        if self.n < self.win_size:
            self.n += 1
            self.mean = self.cum_sum / float(self.n)
            
        else:
            self.cum_sum -= self.windows.popleft()
            self.mean = self.cum_sum / float(self.win_size)

    def get_mean(self):
        return self.mean if self.n else np.zeros(self.n_size)

    def __str__(self):
        return f"Current window values: {list(self.windows)}"


#%%
def load_train(drop_days=None, zero_weight=True):
    with timer("Loading train parquet"):
        train_parquet = os.path.join(DATA_DIR, 'train.parquet')
        train = pd.read_parquet(train_parquet)
        if drop_days:
            train = train.query(f'date not in {drop_days}').reset_index (drop = True)
        
        if not zero_weight:
            train = train.query('weight > 0').reset_index (drop = True)
        
        feat_cols = [f'feature_{i}' for i in range(130)]
        # train[feat_cols].mean().to_csv(os.path.join(DATA_DIR, 'f_mean_final.csv'), 
        #                                index_label=['features'], header=['mean'])
        f_mean = train[feat_cols].mean().values.reshape(1,-1)
        if zero_weight:
            np.save(DATA_DIR+'f_mean_after_85_include_zero_weight.npy', f_mean)
        else:
            np.save(DATA_DIR+'f_mean_after_85_positive_weight.npy', f_mean)
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

    if not debug:
        train_pdm.to_parquet(os.path.join(DATA_DIR, 'train_pdm.parquet'), index=False)

  
# %%

if __name__ == '__main__':
    get_system()
    # train = load_train(drop_days=[2, 36, 270, 294])
    train = load_train(drop_days=list(range(0,86))+[270, 294])
    process_train_rolling(train, debug=True)
