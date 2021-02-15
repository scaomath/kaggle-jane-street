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
data preparation for the final submission
1. Drop outliers [2, 294], low volume days [36, 270].


Reference: Carl McBride Ellis
https://www.kaggle.com/carlmcbrideellis/semper-augustus-pre-process-training-data

Past day mean/EW mean push
Reference: Lucas Morin's notebook
https://www.kaggle.com/lucasmorin/running-algos-fe-for-fast-inference?scriptVersionId=50754012
'''