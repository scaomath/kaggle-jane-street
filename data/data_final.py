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
from data.data_rolling import RunningPDA, RunningEWMean, RunningMean

# %%
'''
data preparation for the final submission (in order)
1. fillna() uses past day mean including all weight zero rows. 
2. Most common values fillna for spike features rows.
3. Drop outliers [2, 294], low volume days [36, 270].
4. all data, only drop the two partial days and the two <2k ts_id days.
5. smoother data, aside from 1, query day > 85, drop ts_id > 8700 days.
6. Final training uses only weight > 0 rows, but with a randomly
selected 40% of weight zero rows' weight being replaced by 1e-7 to
reduce overfitting.
7. a new denoised target is generated with all five targets.

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
    # train = train.set_index('ts_id')
# %%
feat_spike_index = [1, 2, 3, 4, 5, 6, 10, 14, 16, 69, 70, 71, 73, 74, 75, 76, 79, 80, 81, 82, 85,
                    86, 87, 88, 91, 92, 93, 94, 97, 98, 99, 100, 103, 104, 105, 106, 109, 111, 112, 115, 117, 118]
feat_reg_index = list(set(range(130)).difference(feat_spike_index))
features_reg = [f'feature_{i}' for i in feat_reg_index]
features_spike = [f'feature_{i}' for i in feat_spike_index]
resp_cols = ['resp_1', 'resp_2', 'resp_3', 'resp', 'resp_4', ]

feat_cols = [f'feature_{i}' for i in range(130)]
# feat_cols = features_reg
feat_cols_c = feat_cols + [f+'_c' for f in features_spike]
print(f"Number of features: {len(feat_cols)}")

# %%
feat_mean = train[feat_cols].mean().values.reshape(1,-1)
np.save(DATA_DIR+'f_mean_all_days_include_zero_weight.npy', feat_mean)

#%%
all_mean = train.mean().values.reshape(1,-1)
# %%
most_common_vals = []
# most_common_vals = np.load(DATA_DIR+'spike_common_vals_42.npy').reshape(-1)

for i, feat in enumerate(features_spike):
    sorted_counts = train[feat].value_counts().sort_values(ascending=False)
    # print(sorted_counts.head(5), '\n\n')
    # if sorted_counts.iloc[0]/sorted_counts.iloc[1] > 30 and sorted_counts.iloc[0] > 5000:
    # feat_spike_index.append(sorted_counts.name.split('_')[-1])
    most_common_val = sorted_counts.index[0]
    most_common_vals.append(most_common_val)

spike_fillna_val = np.zeros((1, len(feat_cols)))
spike_fillna_val[...,feat_spike_index] = np.array(most_common_vals)
# np.save(DATA_DIR+'fillna_val_spike_feats.npy', spike_fillna_val)

#%%
class RunningPDAFinal():
    '''
    The subclass only for data-preparation, not for final submission pipeline
    '''
    def __init__(self, past_mean=0):
        self.day = -1
        self.past_mean = past_mean # past day mean, initialized as the mean
        self.cum_sum = 0
        self.day_instances = 0 # current day instances
        self.past_value = past_mean # the previous row's value, initialized as the mean
        self.past_instances = 0 # instances in the past day
        self.past_day_data = np.zeros_like(past_mean)
        self.current_day_data = np.zeros_like(past_mean)
    
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
            self.past_day_data = np.array(self.current_day_data)
            self.current_day_data = [x]
            
        else:
            self.day_instances += 1
            self.cum_sum += x
            self.current_day_data.append(x)

    def get_mean(self):
        return self.cum_sum/self.day_instances

    def get_past_mean(self):
        return self.past_mean

    def get_past_mean_numpy(self):
        return np.mean(self.past_day_data, axis=0)

    def get_past_std(self):
        return np.std(self.past_day_data, axis=0)


pdm = RunningPDAFinal(past_mean=all_mean)


#%%
train_debug = train.query('date in [8, 9, 10, 11]').copy().reset_index(drop=True)
train_debug = train_debug[:-1]
row_vals = []
nonfeat_cols = ['date', 'weight', 'resp', 'resp_1', 'resp_2', 'resp_3', 'resp_4',]
with tqdm(total=len(train_debug)) as pbar:
    
    for _, row in train_debug.iterrows(): 
        date = row['date']
        pdm.push(np.array(row), date)
        
        past_day_mean = pdm.get_past_mean()

        x_tt = row.values
        if np.isnan(x_tt.sum()):
            x_tt = np.nan_to_num(x_tt) + np.isnan(x_tt) * past_day_mean 

        row_vals.append(x_tt)
        pbar.update()

#%%
train_dtypes = {'date': np.int32,
                'ts_id': np.int64,
                'resp': np.float64,
                'weight': np.float64,
                }
for c in range(1,5):
    train_dtypes['resp_'+str(c)] = np.float64
for c in range(130):
    train_dtypes['feature_'+str(c)] = np.float32
train_pdm = pd.DataFrame(row_vals, columns=nonfeat_cols+feat_cols, index=train.index).astype(train_dtypes)

#%%    
train_pdm.to_parquet(os.path.join(DATA_DIR, 'train_final.parquet'), index=False)