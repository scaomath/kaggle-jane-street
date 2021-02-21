#%%
import os
import sys
current_path = os.path.dirname(os.path.abspath(__file__))
HOME = os.path.dirname(current_path)
sys.path.append(HOME)

import pandas as pd
pd.set_option('display.max_rows', 100)
pd.set_option('display.max_columns', 100)

import numpy as np
import datatable as dt
from tqdm.auto import tqdm
from collections import deque
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(style="darkgrid", context="talk")
from jupyterthemes import jtplot
jtplot.style(theme='onedork', context='notebook', ticks=True, grid=False)


MODEL_DIR = HOME+'/models/'
DATA_DIR = HOME+'/data/'
from utils import *
from utils_js import *
from data.data_rolling import RunningPDA, RunningEWMeanDay, RunningMeanDay

# %%
'''
data preparation for the final submission (in order)

1. Drop outliers [2, 294], low volume days [36, 270].
2. fillna() uses past day mean including all weight zero rows. 
3. Most common values fillna for spike features rows (a small random noise added).
4. all data, only drop the two partial days and the two <2k ts_id days.
5. smoother data, aside from 1, query day > 85, drop ts_id > 8700 days.
6. Final training uses only weight > 0 rows, but with a randomly
selected 40% of weight zero rows' weight being replaced by 1e-7 to
reduce overfitting.
7. a new denoised target is generated with all five targets.

testing out new features
- ewm for feature_0
- moving average for feature_0

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
    
    # train_parquet = os.path.join(DATA_DIR, 'train.parquet')
    # train = pd.read_parquet(train_parquet)

# train = train.set_index('ts_id')
train = train.query('date not in [2, 36, 270, 294]').reset_index(drop=True)
# %%
# the first one is used for model
# feat_spike_index = [1, 2, 3, 4, 5, 6, 10, 14, 16, 69, 70, 71, 73, 74, 75, 76, 79, 80, 81, 82, 85,
#                     86, 87, 88, 91, 92, 93, 94, 97, 98, 99, 100, 103, 104, 105, 106, 109, 111, 112, 115, 117, 118]

# this one is used for fillna
feat_spike_index = [1, 2, 69, 71, 85, 87, 88, 91, 93, 94, 97, 99, 100, 103, 105, 106]

noisy_index = [3, 4, 5, 6, 8, 10, 12, 14, 16, 37, 38, 39, 40, 72, 73, 74, 75, 76,
                78, 79, 80, 81, 82, 83]
negative_index = [73, 75, 76, 77, 79, 81, 82]
hybrid_index = [55, 56, 57, 58, 59]
running_indices = sorted([0]+noisy_index+negative_index+hybrid_index)
features_running = [f'feature_{i}' for i in running_indices]

feat_reg_index = list(set(range(130)).difference(feat_spike_index))
features_reg = [f'feature_{i}' for i in feat_reg_index]
features_spike = [f'feature_{i}' for i in feat_spike_index]
resp_cols = ['resp_1', 'resp_2', 'resp_3', 'resp', 'resp_4', ]

feat_cols = [f'feature_{i}' for i in range(130)]
# feat_cols = features_reg
feat_cols_c = feat_cols + [f+'_c' for f in features_spike]
print(f"Number of features: {len(feat_cols)}")
print(f"Number of spike fillna features: {len(features_spike)}")
# %%
try:
    feat_mean = np.load(DATA_DIR+'f_mean_all_days_include_zero_weight.npy')
except:
    feat_mean = train[feat_cols].mean().values.reshape(1,-1)
    np.save(DATA_DIR+'f_mean_all_days_include_zero_weight.npy', feat_mean)
all_mean = train.mean().values
#%%
# %%
try:
    spike_fillna_val = np.load(DATA_DIR+'fillna_val_spike_feats.npy')
except:
    most_common_vals = []
    # most_common_vals = np.load(DATA_DIR+'spike_common_vals_42.npy').reshape(-1)

    for i, feat in enumerate(features_spike):
        sorted_counts = train[feat].value_counts().sort_values(ascending=False)
        print(sorted_counts.head(5), '\n\n')
        # if sorted_counts.iloc[0]/sorted_counts.iloc[1] > 30 and sorted_counts.iloc[0] > 5000:
        # feat_spike_index.append(sorted_counts.name.split('_')[-1])
        most_common_val = sorted_counts.index[0]
        most_common_vals.append(most_common_val)

    spike_fillna_val = np.zeros((len(feat_cols), ))
    spike_fillna_val[feat_spike_index] = np.array(most_common_vals)
    np.save(DATA_DIR+'fillna_val_spike_feats.npy', spike_fillna_val)

#%%

class RunningPDAFinal():
    '''
    The subclass only for data-preparation, not for final submission pipeline
    '''
    def __init__(self, past_mean=all_mean):
        self.day = -1
        self.past_mean = past_mean # past day mean, initialized as the mean
        self.cum_sum = 0
        self.day_instances = 0 # current day instances
        self.past_value = past_mean # the previous row's value, initialized as the mean
        self.past_instances = 0 # instances in the past day
        self.past_day_data = np.zeros_like(past_mean)
        self.current_day_data = past_mean
    
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
            # print(self.past_day_data[0])
            self.current_day_data = []
            self.current_day_data.append(list(x))
            # print(self.current_day_data)
            # print(x[0])

        else:
            self.day_instances += 1
            self.cum_sum += x
            self.current_day_data.append(list(x))
       

    def get_mean(self):
        return self.cum_sum/self.day_instances

    def get_past_mean(self):
        return self.past_mean

    def get_past_mean_numpy(self):
        return np.mean(self.past_day_data, axis=0)

    def get_past_std(self):
        return np.std(self.past_day_data, axis=0)
#%%
feat_mean = feat_mean.reshape(-1)
pdm = RunningPDAFinal(past_mean=feat_mean)

feat_vals = []
# nonfeat_cols = ['date', 'weight', 'resp', 'resp_1', 'resp_2', 'resp_3', 'resp_4',]
n_feats = len(feat_cols)
spike_fillna_val = spike_fillna_val.reshape(-1)

with tqdm(total=len(train)) as pbar:
    
    for _, row in train.iterrows(): 
        date = row['date']
        x_tt = row.values[7:-1]
        assert x_tt[0] == 1 or x_tt[0] == -1
        pdm.push(x_tt, date)
        
        past_day_mean = pdm.get_past_mean().reshape(-1)
        past_day_mean[feat_spike_index] = 0
        fillna_val = past_day_mean + spike_fillna_val
        if np.isnan(x_tt.sum()):
            # x_tt = np.nan_to_num(x_tt) + np.isnan(x_tt)*spike_fillna_val # bug!!!!!!
            x_tt = np.nan_to_num(x_tt) + np.isnan(x_tt)*fillna_val

        feat_vals.append(x_tt)
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

#%%    
feature_df = pd.DataFrame(feat_vals, columns=feat_cols, index=train.index)

# %%
train_final = train.copy()
train_final[feat_cols] = feature_df
train_final = train_final.astype(train_dtypes)
# %%
# train_final = train_final.astype(train_dtypes)
train_final.to_parquet(os.path.join(DATA_DIR, 'train_final.parquet'), index=False)
# %%
train_final.to_feather(os.path.join(DATA_DIR, 'train_final.feather'))
# %%
trades_per_day = train_final.groupby(['date'])['ts_id'].count()
volatile_days  = pd.DataFrame(trades_per_day[trades_per_day > 8600])
print("Number of volatile days",volatile_days.count())
filter_list    = volatile_days.index.to_list()

#%%
filter_list = [1,  4,  5,  12,  16,  18,  24,  37,  38,  43,  44,  45,  47,
             59,  63,  80,  85, 161, 168, 452, 459, 462]
train_final_regular = train_final.query('date != @filter_list').reset_index(drop = True)
train_final_regular = train_final.query('date >85').reset_index(drop = True)
# %%
train_final_regular.to_parquet(os.path.join(DATA_DIR, 'train_final_regular.parquet'), index=False)
# %%
