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

Modified by Shuhao Cao and Ethan Zheng to 
1. able to return past day trading numbers.
2. able to use feature 64 to predict whether a day is ''busy''

'''


class RunningPDA:
    '''
    https://www.kaggle.com/lucasmorin/running-algos-fe-for-fast-inference?scriptVersionId=50754012
    '''
    def __init__(self, past_mean=0, start=1000, end=2500, slope=0.00116):
        self.day = -1
        self.past_mean = past_mean # past day mean, initialized as the mean
        self.cum_sum = 0
        self.day_instances = 0 # current day instances
        self.past_value = past_mean # the previous row's value, initialized as the mean
        self.past_instances = 0 # instances in the past day
        
        self.start = start
        self.end = end
        self.slope = slope
        self.start_value = None
        self.end_value = None

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
            
            self.start_value, self.end_value = None, None
            
        else:
            self.day_instances += 1
            self.cum_sum += x
        
        if self.day_instances == self.start:
            self.start_value = x[:, 64]
        if self.day_instances == self.end:
            self.end_value = x[:, 64]

    def get_mean(self):
        return self.cum_sum/self.day_instances

    def get_past_mean(self):
        return self.past_mean

    def get_past_trade(self):
        return self.past_instances
    
    def predict_today_busy(self):
        if self.start_value is None or self.end_value is None:
            return False
        return (self.end_value - self.start_value) / (self.end - self.start) < self.slope

class RunningEWMeanDay:
    '''
    Reference: Lucas Morin
    https://www.kaggle.com/lucasmorin/running-algos-fe-for-fast-inference?scriptVersionId=50754012
    Modified to do the rolling mean only intraday
    '''
    def __init__(self, window=20, num_feat = 1, lt_mean = None):
        if lt_mean is not None:
            self.s = lt_mean
        else:
            self.s = np.zeros(num_feat)
        self.past_value = np.zeros(num_feat)
        self.alpha = 2 /(window + 1)
        self.day = -1

    def clear(self):
        self.s = 0

    def push(self, x, date):
        
        x = fast_fillna(x, self.past_value)
        self.past_value = x

        if date > self.day:
            self.day = date
            self.clear()
            self.s = x
        else:
            self.s = self.alpha * x + (1 - self.alpha) * self.s
        
    def get_mean(self):
        return self.s


class RunningMeanDay:
    '''
    Reference: Lucas Morin
    https://www.kaggle.com/lucasmorin/running-algos-fe-for-fast-inference?scriptVersionId=50754012
    Modified to do the rolling mean only intraday
    '''
    def __init__(self, window=1000, num_feat = 1):
        self.day = -1
        self.n = 0
        self.mean = 0
        self.run_var = 0
        self.window = window
        self.past_value = 0
        self.windows = deque(maxlen=window+1)
        self.num_feat=num_feat

    def clear(self):
        self.n = 0
        self.windows.clear()

    def push(self, x, date):
        
        x = fast_fillna(x, self.past_value)
        self.past_value = x

        if date > self.day:
            self.day = date
            self.clear()
            self.windows.append(x)
            self.n = 1
            self.mean = x
            self.run_var = 0
        else:
            self.windows.append(x)

            if self.n < self.window:
                # Calculating first variance
                self.n += 1
                delta = x - self.mean
                self.mean += delta / self.n
                self.run_var += delta * (x - self.mean)
            else:
                # Adjusting variance
                x_removed = self.windows.popleft()
                old_m = self.mean
                self.mean += (x - x_removed) / self.window
                self.run_var += (x + x_removed - old_m - self.mean) * (x - x_removed)

    def get_mean(self):
        return self.mean if self.n else np.zeros(self.num_feat)

    def get_var(self):
        return self.run_var / (self.n) if self.n > 1 else np.zeros(self.num_feat)

    def get_std(self):
        return math.sqrt(self.get_var())

    def get_all(self):
        return list(self.windows)

    def __str__(self):
        return "Current window values: {}".format(list(self.windows))


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
