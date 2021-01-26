#%%
import os
import sys
import pandas as pd
import numpy as np
from tqdm import tqdm
from sklearn.metrics import roc_auc_score
import torch

from numba import njit

HOME = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = HOME+'/model/'
DATA_DIR = HOME+'/data/'
from utils import *
from utils_js import *

#%%
'''
The mock test set is taken from the Purged Time series CV split last fold's test set:

Reference:
https://www.kaggle.com/jorijnsmit/found-the-holy-grail-grouptimeseriessplit
https://www.kaggle.com/tomwarrens/purgedgrouptimeseriessplit-stacking-ensemble-mode
'''

with timer("Loading train parquet"):
    train_parquet = os.path.join(DATA_DIR, 'train.parquet')
    train = pd.read_parquet(train_parquet)
# print(train.info())
simu_test = train[train['date'] > 480].reset_index(drop = True) 
print(f"Simulated public test file length: {simu_test}")

#%%
# for tr_idx, val_idx in PurgedGroupTimeSeriesSplit().split(train, groups=train['date']):
#     print(train.loc[tr_idx, 'date'].unique())
#     print(train.loc[val_idx, 'date'].unique(), '\n\n')
# %%
class Iter_Valid(object):
    def __init__(self, df):
        df = df.reset_index(drop=True)
        self.df = df
        self.weight = df['weight'].astype(float).values
        self.action = df['action'].astype(int).values
        self.pred_df = df
        self.pred_df['action'] = 0
        self.len = len(df)
        self.current = 0

    def __iter__(self):
        return self
    
    def yield_df(self, pre_start):
        df = self.df[pre_start:self.current].copy()
        sample_df = self.sample_df[pre_start:self.current].copy()
        return df, sample_df

    def __next__(self):
        pre_start = self.current
        while self.current < self.len:
            self.current += 1
        if pre_start < self.current:
            return self.df, self.pred_df
        else:
            raise StopIteration()
try:
    iter_test = Iter_Valid(simu_test)
except:
    pass

predicted = []

def set_predict(df):
    predicted.append(df)


# %%

