#%%
import os
import sys
current_path = os.path.dirname(os.path.abspath(__file__))
HOME = os.path.dirname(current_path)
sys.path.append(HOME)
# for f in ['/home/scao/anaconda3/lib/python3.8/lib-dynload', 
#           '/home/scao/anaconda3/lib/python3.8/site-packages']:
#     sys.path.append(f) 

import pandas as pd
pd.set_option('display.max_rows', 100)
pd.set_option('display.max_columns', 100)

import numpy as np
import datatable as dt
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(style="darkgrid", context="talk")
from jupyterthemes import jtplot
jtplot.style(theme='onedork', context='notebook', ticks=True, grid=False)


MODEL_DIR = HOME+'/models/'
DATA_DIR = HOME+'/data/'
from utils import *
from utils_js import *
# from data.data_rolling import RunningPDA, RunningEWMean, RunningMean

# %%
'''
data preparation for the final submission
1. Drop outliers [2, 294], low volume days [36, 270].
2. fillna() uses past day mean including all weight zero rows,
3. all data, only drop the two partial days and the two <2k ts_id days.
4. smoother data, aside from 1, query day > 85, drop ts_id > 8700 days.
5. Final training uses only weight > 0 rows, but with a randomly
selected 40% of weight zero rows' weight being replaced by 1e-7 to
reduce overfitting.
6. a new denoised target is generated with all five targets.

Reference: Carl McBride Ellis
https://www.kaggle.com/carlmcbrideellis/semper-augustus-pre-process-training-data

Past day mean/EW mean push
Reference: Lucas Morin's notebook
https://www.kaggle.com/lucasmorin/running-algos-fe-for-fast-inference?scriptVersionId=50754012
'''
# %%
with timer("Loading train"):
    train_csv = os.path.join(DATA_DIR, 'train.csv')
    train = dt.fread(train_csv).to_pandas()
    train = train.set_index('ts_id')
# %%
feat_reg_index = [0, 17, 18, 37, 39, 40, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 55, 57, 58]
feat_reg_index += list(range(60,69))
feat_reg_index += [89, 101, 108, 113, 119, 120, 121, 122, 124, 125, 126, 128]
feat_spike_index = list(set(range(130)).difference(feat_reg_index))
features_reg = [f'feature_{i}' for i in feat_reg_index]
features_spike = [f'feature_{i}' for i in feat_spike_index]
# %%
