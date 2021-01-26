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
# %%
'''
Simulate the inference env of Kaggle

Utility function taken from https://www.kaggle.com/gogo827jz/jane-street-super-fast-utility-score-function
'''

def utility_score_loop(date, weight, resp, action):
    count_i = len(np.unique(date))
    Pi = np.zeros(count_i)
    for i, day in enumerate(np.unique(date)):
        Pi[i] = np.sum(weight[date == day] * resp[date == day] * action[date == day])
    t = np.sum(Pi) / np.sqrt(np.sum(Pi ** 2)) * np.sqrt(250 / count_i)
    u = np.clip(t, 0, 6) * np.sum(Pi)
    return u

def utility_score_bincount(date, weight, resp, action):
    count_i = len(np.unique(date))
    Pi = np.bincount(date, weight * resp * action)
    t = np.sum(Pi) / np.sqrt(np.sum(Pi ** 2)) * np.sqrt(250 / count_i)
    u = np.clip(t, 0, 6) * np.sum(Pi)
    return u

@njit(fastmath = True)
def utility_score_numba(date, weight, resp, action):
    Pi = np.bincount(date, weight * resp * action)
    t = np.sum(Pi) / np.sqrt(np.sum(Pi ** 2)) * np.sqrt(250 / len(Pi))
    u = min(max(t, 0), 6) * np.sum(Pi)
    return u

@njit(fastmath = True)
def decision_threshold_optimisation(preds, date, weight, resp, low = 0, high = 1, bins = 100, eps = 1):
    opt_threshold = low
    gap = (high - low) / bins
    action = np.where(preds >= opt_threshold, 1, 0)
    opt_utility = utility_score_numba(date, weight, resp, action)
    for threshold in np.arange(low, high, gap):
        action = np.where(preds >= threshold, 1, 0)
        utility = utility_score_numba(date, weight, resp, action)
        if utility - opt_utility > eps:
            opt_threshold = threshold
            opt_utility = utility
    print(f'Optimal Decision Threshold:   {opt_threshold}')
    print(f'Optimal Utility Score:        {opt_utility}')
    return opt_threshold, opt_utility

@njit
def fast_fillna(array, values):
    if np.isnan(array.sum()):
        array = np.where(np.isnan(array), values, array)
    return array
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

iter_test = Iter_Valid(simu_test_df)
predicted = []
def set_predict(df):
    predicted.append(df)