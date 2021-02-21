#%%
import datetime
import gc
import os
import random
import sys

import datatable as dt
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from numba import njit
from sklearn.metrics import roc_auc_score
from tqdm import tqdm

HOME = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = HOME+'/models/'
DATA_DIR = HOME+'/data/'
from mlp.mlp import *
from utils import *
from utils_js import *

get_system()

import warnings

from tqdm.auto import tqdm

warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=pd.core.common.SettingWithCopyWarning)
    
plt.style.use('bmh')
plt.rcParams['figure.figsize'] = [14, 8]  # width, height

#%%
from matplotlib.colors import ListedColormap
import numpy as np
import matplotlib.pyplot as plt
    
# this is code slightly modified from the sklearn docs here:
# https://scikit-learn.org/stable/auto_examples/model_selection/plot_cv_indices.html#sphx-glr-auto-examples-model-selection-plot-cv-indices-py
def plot_cv_indices(cv, X, y, group, ax, n_splits, lw=10):
    """Create a sample plot for indices of a cross-validation object."""
    
    cmap_cv = plt.cm.coolwarm

    jet = plt.cm.get_cmap('jet', 256)
    seq = np.linspace(0, 1, 256)
    _ = np.random.shuffle(seq)   # inplace
    cmap_data = ListedColormap(jet(seq))

    # Generate the training/testing visualizations for each CV split
    for ii, (tr, tt) in enumerate(cv.split(X=X, y=y, groups=group)):
        # Fill in indices with the training/test groups
        indices = np.array([np.nan] * len(X))
        indices[tt] = 1
        indices[tr] = 0

        # Visualize the results
        ax.scatter(range(len(indices)), [ii + .5] * len(indices),
                   c=indices, marker='_', lw=lw, cmap=cmap_cv,
                   vmin=-.2, vmax=1.2)

    # Plot the data classes and groups at the end
    ax.scatter(range(len(X)), [ii + 1.5] * len(X),
               c=y, marker='_', lw=lw, cmap=plt.cm.Set3)

    ax.scatter(range(len(X)), [ii + 2.5] * len(X),
               c=group, marker='_', lw=lw, cmap=cmap_data)

    # Formatting
    yticklabels = list(range(n_splits)) + ['target', 'day']
    ax.set(yticks=np.arange(n_splits+2) + .5, yticklabels=yticklabels,
           xlabel='Sample index', ylabel="CV iteration",
           ylim=[n_splits+2.2, -.2], xlim=[0, len(y)])
    ax.set_title('{}'.format(type(cv).__name__), fontsize=15)
    return ax
# %%
n_samples = 20000
n_groups = 500
assert n_samples % n_groups == 0

idx = np.linspace(0, n_samples-1, num=n_samples)
X_train = np.random.random(size=(n_samples, 5))
y_train = np.random.choice([0, 1], n_samples)
groups = np.repeat(np.linspace(0, n_groups-1, num=n_groups), n_samples/n_groups)

fig, ax = plt.subplots()

cv = PurgedGroupTimeSeriesSplit(
    n_splits=5,
    max_train_group_size=300,
    group_gap=5,
    max_test_group_size=40
)

plot_cv_indices(cv, X_train, y_train, groups, ax, 5, lw=20);
# %%
train_parquet = os.path.join(DATA_DIR, 'train_final.parquet')
train_final = pd.read_parquet(train_parquet)
# %%
fig, ax = plt.subplots()

cv = PurgedGroupTimeSeriesSplit(
    n_splits=5,
    max_train_group_size=15,
    group_gap=5,
    max_test_group_size=5
)

plot_cv_indices(
    cv,
    train_final.query('date < 50')[
        train_final.columns[train_final.columns.str.contains('feature')]
    ].values,
    (train_final.query('date < 50')['resp'] > 0).astype(int).values,
    train_final.query('date < 50')['date'].values,
    ax,
    5,
    lw=20
);