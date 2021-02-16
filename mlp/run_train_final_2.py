#%%
from tensorflow.keras.layers import Input, Dense, BatchNormalization, Dropout, GaussianNoise, Concatenate, Lambda, Activation
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.losses import BinaryCrossentropy
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, Callback
from tensorflow.keras.layers.experimental.preprocessing import Normalization
import tensorflow as tf
import tensorflow_addons as tfa
import kerastuner as kt
from tensorflow.keras import backend as K

import numpy as np
import pandas as pd
from tqdm.auto import tqdm
from random import choices

current_path = os.path.dirname(os.path.abspath(__file__))
HOME = os.path.dirname(current_path)
MODEL_DIR = os.path.join(HOME,  'models')
DATA_DIR = os.path.join(HOME,  'data')
sys.path.append(HOME)

from utils import *
# %%
'''
Final model 2: 
1. data: including the volatile day but excluding the outlier days (2, 294, 36, 270)
2. data: the fillna is using the past day mean (after excluding the days above)
3. data: target is only resp_{0,3,4}
3. Denoised auto-encoder
4. simple MLP tf model
'''