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
#%%
train_parquet = os.path.join(DATA_DIR, 'train_final.parquet')
train_final = pd.read_parquet(train_parquet)

train_parquet = os.path.join(DATA_DIR, 'train_final_ver1.parquet')
train_final_ver1 = pd.read_parquet(train_parquet)

train_parquet = os.path.join(DATA_DIR, 'train.parquet')
train_orig = pd.read_parquet(train_parquet)

#%%
features_csv = os.path.join(DATA_DIR, 'features.csv')
features = pd.read_csv(features_csv)
tags = [t for t in list(features.iloc[:,1:])]
tags_dict = {}
for tag in tags:
    tags_dict[tag] = features[features[tag] == True]['feature'].to_list()
    # print(tag)
    feat_num = " ".join([t.split('_')[-1] for t in tags_dict[tag]])
    # print(f"Features: {feat_num}")


def plot_features(feats, train, scatter=False, num_days=3, start_day=None):
    if not start_day:
        start_day = np.random.randint(0, 500-num_days, 1)[0]
    days = [start_day+i for i in range(num_days)]
    days_str = " ".join([str(d) for d in days])

    num_feat = len(feats)
    _, axes = plt.subplots(num_feat, 1, figsize=(15,num_feat*2), constrained_layout=True)
    cmap = get_cmap(num_feat*2, cmap='RdYlGn')
    for i in range(num_feat):
        feat = feats[i]
        feat_vals = train[train['date'].isin(days)][feat].reset_index(drop=True)
        if scatter:
            axes[i].scatter(pd.Series(feat_vals).index, pd.Series(feat_vals), s=5, color=cmap(i))
        else:
            axes[i].plot(pd.Series(feat_vals).index, pd.Series(feat_vals), lw=1, color=cmap(i))
        axes[i].set_title (feat+" at "+days_str, fontsize=15);
        axes[i].set_xlim(xmin=0)
# %%
plot_features(tags_dict['tag_22'], train_final, scatter=True)


# %%
plot_features(tags_dict['tag_2'], train_final)
# %%
# feats = ['feature_74', 'feature_80', 'feature_86', 'feature_92', 'feature_98', 'feature_104']
# feats = ['feature_106', 'feature_118']
feats = ['feature_98', 'feature_104']
plot_features(feats, train_final, start_day=320, num_days=2)
# plt.savefig(DATA_DIR+'feat_98_104_fillna_pdm.png')
plot_features(feats, train_final_ver1, start_day=320, num_days=2)
plot_features(feats, train_orig, start_day=320,num_days=2)
# plt.savefig(DATA_DIR+'feat_98_104.png')
# %%
train_final['feature_92'].value_counts().sort_values(ascending=False)
train_final.query('date in [320]')['feature_92'].value_counts().sort_values(ascending=False)
# %%
feats = ['feature_1', 'feature_69']
start_day = np.random.randint(0, 500-3, 1)[0]
plot_features(feats, train_final, start_day=start_day)
plot_features(feats, train, start_day=start_day)

#%%
feat_spike_index = [1, 2, 69, 71, 85, 87, 88, 91, 93, 94, 97, 99, 100, 103, 105, 106]
# feats = ['feature_100', 'feature_106']
feats = ['feature_1', 'feature_2', 'feature_69']
start_day = np.random.randint(0, 500-3, 1)[0]
plot_features(feats, train_final, start_day=start_day, scatter=True)
plot_features(feats, train, start_day=start_day,  scatter=True)
# %%
train[['feature_85','feature_91']].value_counts()