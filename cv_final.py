#%%
import datetime
import gc
import os
HOME = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = HOME+'/models/'
DATA_DIR = HOME+'/data/'
# from mlp.mlp import *
from utils import *
from utils_js import *
from mlp.tf_models import *

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


# %%
'''
Loading model trained in tf and verify their utility scores
'''
# %%
