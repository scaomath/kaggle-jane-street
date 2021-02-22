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
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
tf.config.optimizer.set_jit(True)
# %%
'''
Loading model trained in tf and verify their utility scores
'''
train_parquet = os.path.join(DATA_DIR, 'train_pdm.parquet')
train = pd.read_parquet(train_parquet)
# %%
'''
Final model resnet
'''

features = [f'feature_{i}' for i in range(130)]

resp_cols = ['resp_1', 'resp_2', 'resp_3', 'resp', 'resp_4']
resp_cols_vol = ['resp_3', 'resp', 'resp_4']
# split features for a ResNet feature 2 is more important
features_2_index = [0, 1, 2, 3, 4, 5, 6, 15, 16, 25, 26, 35, 
             36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 
             49, 50, 51, 52, 53, 54, 59, 60, 61, 62, 63, 64, 65, 
             66, 67, 68, 69, 70, 71, 76, 77, 82, 83, 88, 89, 94, 
             95, 100, 101, 106, 107, 112, 113, 118, 119, 128, 129]

features_1_index = [0] + list(set(range(130)).difference(features_2_index))

features_1 = [f'feature_{i}' for i in features_1_index]

features_2 = [f'feature_{i}' for i in features_2_index]


# split features for a ResNet feature 2 is more important
features_1_index_v = [0,
                   7, 8, 17, 18, 27, 28, 55, 72, 78, 84, 90, 96, 102, 108, 114, 120, 121,
                   11, 12, 21, 22, 31, 32, 57, 74, 80, 86, 92, 98, 104, 110, 116, 124, 125] 
                # resp_1 resp_2 feat
    
features_2_index_v = [0] + list(set(range(130)).difference(features_1_index_v))

features_1_v = [f'feature_{i}' for i in features_1_index_v]

features_2_v = [f'feature_{i}' for i in features_2_index_v]


feat_spike_index = [1, 2, 3, 4, 5, 6, 10, 14, 16, 69, 70, 71, 73, 74, 75, 76, 79, 80, 81, 82, 85,
                    86, 87, 88, 91, 92, 93, 94, 97, 98, 99, 100, 103, 104, 105, 106, 109, 111, 112, 115, 117, 118]
features_spike = [f'feature_{i}' for i in feat_spike_index]

cat_cols = [f+'_c' for f in features_spike]


#%%
model_files = ['resnet_reg_fold_0_seed_1127802.h5', 
                'resnet_reg_fold_1_seed_1127802.h5',
                'resnet_reg_fold_2_seed_1127802.h5']
# model_files = ['tf2_fold_1_res_seed_792734.h5',
#                'tf2_fold_2_res_seed_97275.h5']

for _fold, model_file in enumerate(model_files):
    print(f"Model {model_file}")
    tf.keras.backend.clear_session()
    tf_model = create_resnet_reg(len(features_1), len(features_2), len(resp_cols), 
                                hidden_size=256, label_smoothing=5e-03)

    tf_model.load_weights(os.path.join(MODEL_DIR, model_file))
    # tf_model.call = tf.function(tf_model.call, experimental_relax_shapes=True)

    print_valid_score_tf(train, tf_model, start_day=400, num_days=33, 
                            f=median_avg, threshold=0.5, 
                            feature_indices=(features, features_1_index, features_2_index))

#%%
# model_files = ['resnet_volatile_fold_0_seed_1127802.h5', 
#                 'resnet_volatile_fold_1_seed_1127802.h5',
#                 'resnet_volatile_fold_2_seed_1127802.h5']
model_files = ['resnet_volatile_fold_0_seed_157157.h5', 
                'resnet_volatile_fold_1_seed_157157.h5',
                'resnet_volatile_fold_2_seed_157157.h5']

model_files = ['resnet_volatile_fold_0_seed_745273.h5', 
            'resnet_volatile_fold_2_seed_962656.h5']

for _fold, model_file in enumerate(model_files):
    print(f"Model {model_file}")
    tf.keras.backend.clear_session()
    tf_model = create_resnet(len(features_1_v), len(features_2_v), len(resp_cols_vol), 
                                hidden_size=256, label_smoothing=5e-03)

    tf_model.load_weights(os.path.join(MODEL_DIR, model_file))
    # tf_model.call = tf.function(tf_model.call, experimental_relax_shapes=True)

    print_valid_score_tf(train, tf_model, start_day=400, num_days=33, 
                            f=median_avg, threshold=0.5, 
                            feature_indices=(features, features_1_index_v, features_2_index_v))
# %%
# encoder_file = 'encoder_reg.hdf5'
# model_files = ['ae_reg_fold_0.hdf5', 
#                 'ae_reg_fold_1.hdf5',
#                 'ae_reg_fold_2.hdf5']
# hp_file = 'hp_ae_reg.pkl'


encoder_file = 'encoder_692874.hdf5'
model_files = ['model_692874_0.hdf5', 
                'model_692874_1.hdf5',
                'model_692874_2.hdf5']
hp_file = 'best_hp_692874.pkl'

_, encoder = create_autoencoder(len(features), len(resp_cols), noise=0.1)

encoder.load_weights(os.path.join(MODEL_DIR, encoder_file))
encoder.trainable = False

model_fn = lambda hp: create_model(hp, len(features), len(resp_cols), encoder)

hp = pd.read_pickle(os.path.join(MODEL_DIR, hp_file))
for _fold, model_file in enumerate(model_files):
    tf.keras.backend.clear_session()
    print(f"Model {model_file}")
    model = model_fn(hp)
    model.load_weights(os.path.join(MODEL_DIR, model_files[_fold]))

    print_valid_score_tf(train, model, start_day=400, num_days=33, 
                            f=median_avg, threshold=0.5, 
                            feature_indices=(features, features_1_index, features_2_index)
# %%
# volatile models, 3 targets
# encoder_file = 'encoder_volatile.hdf5'
# model_files = ['ae_volatile_fold_0.hdf5', 
#                 'ae_volatile_fold_1.hdf5',
#                 'ae_volatile_fold_2.hdf5']
# hp_file = 'hp_ae_volatile.pkl'


# encoder_file = 'v_encoder_969725.hdf5'
# model_files = ['v_model_969725_0.hdf5', 
#                 'v_model_969725_1.hdf5',
#                 'v_model_969725_2.hdf5']
# hp_file = 'v_best_hp_969725.pkl'

encoder_file = 'v_encoder_618734.hdf5'
model_files = ['v_model_618734_0.hdf5', 
                'v_model_618734_1.hdf5',
                'v_model_618734_2.hdf5']
hp_file = 'v_best_hp_618734.pkl'

_, encoder = create_autoencoder(len(features), len(resp_cols_vol), noise=0.1)

encoder.load_weights(os.path.join(MODEL_DIR, encoder_file))
encoder.trainable = False

model_fn = lambda hp: create_model(hp, len(features), len(resp_cols_vol), encoder)

hp = pd.read_pickle(os.path.join(MODEL_DIR, hp_file))
for _fold, model_file in enumerate(model_files):
    tf.keras.backend.clear_session()
    print(f"Model {model_file}")
    model = model_fn(hp)
    model.load_weights(os.path.join(MODEL_DIR, model_files[_fold]))

    print_valid_score_tf(train, model, start_day=400, num_days=33, 
                            f=median_avg, threshold=0.5, 
                            feature_indices=[features])
# %%

#%%
model_files = ['tf_spike_reg_seed_1127802_fold_0.h5', 
                'tf_spike_reg_seed_1127802_fold_1.h5',
                # 'tf_spike_reg_seed_1127802_fold_2.h5',
                'tf_spike_reg_seed_802_fold_2.h5'
                ]

for _fold, model_file in enumerate(model_files):
    print(f"Model {model_file}")
    tf.keras.backend.clear_session()
    tf_model = create_spikenet(len(features_1), len(features_2), len(cat_cols), len(resp_cols), 
                                hidden_size=256, label_smoothing=5e-03)

    tf_model.load_weights(os.path.join(MODEL_DIR, model_file))
    # tf_model.call = tf.function(tf_model.call, experimental_relax_shapes=True)

    print_valid_score_tf(train, tf_model, start_day=400, num_days=33, 
                            f=median_avg, threshold=0.5, 
                            feature_indices=(features, features_1_index, features_2_index, feat_spike_index))
# %%
