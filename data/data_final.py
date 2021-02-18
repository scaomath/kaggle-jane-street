#%%
import os
import sys
import pandas as pd
import numpy as np
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(style="darkgrid", context="talk")
from jupyterthemes import jtplot
jtplot.style(theme='onedork', context='notebook', ticks=True, grid=False)


current_path = os.path.dirname(os.path.abspath(__file__))
HOME = os.path.dirname(current_path)
sys.path.append(HOME)
MODEL_DIR = HOME+'/models/'
DATA_DIR = HOME+'/data/'

from utils import *
from utils_js import *
from data.data_rolling import RunningPDA
get_system()
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
    train_parquet = os.path.join(DATA_DIR, 'train.parquet')
    train = pd.read_parquet(train_parquet)
# %%
plt.figure(figsize = (8,5))
sns.distplot(train['feature_3'], bins=200,
                  kde_kws={"clip":(-0.1,-0.02)}, 
                  hist_kws={"range":(-0.1,-0.02)},
                  color='darkcyan', kde=True);

#%%
# %%
plt.figure(figsize = (8,5))
sns.distplot(train['feature_3'], bins=200,
                  kde_kws={"clip":(0, 16)}, 
                  hist_kws={"range":(0, 16)},
                  color='darkcyan', kde=True);
# %%
plt.figure(figsize = (8,5))
train['feature_3_log'] = np.log(train['feature_3'].abs())*np.sign(train['feature_3'].values)
sns.distplot(train['feature_3_log'], bins=200,
                  kde_kws={"clip":(2,3)}, 
                  hist_kws={"range":(2,3)},
                  color='darkcyan', kde=True);
# %%
