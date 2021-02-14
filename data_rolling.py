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
get_system()
# %%
'''
Using the past day mean as fillna, for certain features use EWM

Past day mean
Reference: Lucas Morin's notebook
https://www.kaggle.com/lucasmorin/running-algos-fe-for-fast-inference?scriptVersionId=50754012
'''

class RunningPDA:
    '''
    https://www.kaggle.com/lucasmorin/running-algos-fe-for-fast-inference?scriptVersionId=50754012
    '''
    def __init__(self):
        self.day = -1
        self.past_mean = 0
        self.cum_sum = 0
        self.day_instances = 0
        self.past_value = 0

    def clear(self):
        self.n = 0
        self.windows.clear()

    def push(self, x, date):
        
        x = fast_fillna(x, self.past_value)
        self.past_value = x
        
        # change of day
        if date>self.day:
            self.day = date
            if self.day_instances > 0:
                self.past_mean = self.cum_sum/self.day_instances
            else:
                self.past_mean = 0
            self.day_instances = 1
            self.cum_sum = x
            
        else:
            self.day_instances += 1
            self.cum_sum += x

    def get_mean(self):
        return self.cum_sum/self.day_instances

    def get_past_mean(self):
        return self.past_mean

class RunningEWMean:
    def __init__(self, WIN_SIZE=20, n_size = 1):
        self.s = np.zeros(n_size)
        self.past_value = 0
        self.alpha = 2 /(WIN_SIZE + 1)

    def clear(self):
        self.s = 0

    def push(self, x):
        
        x = fast_fillna(x, self.past_value)
        self.past_value = x
        self.s = self.alpha * x + (1 - self.alpha) * self.s
        
    def get_mean(self):
        return self.s

#%%
def load_train():
    with timer("Loading train parquet"):
        train_parquet = os.path.join(DATA_DIR, 'train.parquet')
        train = pd.read_parquet(train_parquet)
        train = train.query('date > 85').reset_index (drop = True)
    return train
    
train = load_train()

#%%

TRAIN_ROWS = 50_000

train = train[:TRAIN_ROWS]
train_past_day_mean = [] 
pdm = RunningPDA()

pbar = tqdm(total=TRAIN_ROWS)
for index, row in train.iterrows(): 
    date=row['date']
    pdm.push(np.array(row),date)
    pbar.update()
# %%
