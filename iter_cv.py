#%%
import os
import sys
import pandas as pd
import numpy as np
from tqdm import tqdm
from sklearn.metrics import roc_auc_score
import torch
import tensorflow as tf
from numba import njit
import datetime

HOME = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = HOME+'/models/'
DATA_DIR = HOME+'/data/'
from utils import *
from utils_js import *

SEED = 1111
DAY_SIMU_TEST = 490 # this day to 499 as simulated test days
THRESHOLD = 0.5
TQDM_INT = 47
batch_size = 5000
label_smoothing = 1e-2
learning_rate = 1e-3
#%%
'''
The mock test set is taken from the Purged Time series CV split last fold's test set:

Reference:
https://www.kaggle.com/jorijnsmit/found-the-holy-grail-grouptimeseriessplit
https://www.kaggle.com/tomwarrens/purgedgrouptimeseriessplit-stacking-ensemble-mode
'''

with timer("Loading train parquet"):
    train_parquet = os.path.join(DATA_DIR, 'train.parquet')
    train = pd.read_parquet(train_parquet)
# print(train.info())

train['action'] = (train['resp'] > 0)
for c in range(1,5):
    train['action'] = train['action'] & ((train['resp_'+str(c)] > 0))
features = [c for c in train.columns if 'feature' in c]

resp_cols = ['resp', 'resp_1', 'resp_2', 'resp_3', 'resp_4']

# X = train[features].values
# y = np.stack([(train[c] > 0).astype('int') for c in resp_cols]).T #Multitarget

f_mean = np.mean(train[features[1:]].values,axis=0)

simu_test = train[train['date'] > DAY_SIMU_TEST].reset_index(drop = True) 

print(f"Simulated public test file length: {len(simu_test)}")

#%%
# for tr_idx, val_idx in PurgedGroupTimeSeriesSplit().split(train, groups=train['date']):
#     print(train.loc[tr_idx, 'date'].unique())
#     print(train.loc[val_idx, 'date'].unique(), '\n\n')
# %%
class Iter_Valid(object):
    def __init__(self, df):
        df = df.reset_index(drop=True)
        self.df = df
        self.weight = df['weight'].astype(float).values
        self.action = df['action'].astype(int).values
        self.pred_df = df
        self.pred_df['action'] = 0
        self.len = len(df)
        self.current = 0

    def __iter__(self):
        return self
    
    def yield_df(self, pre_start):
        df = self.df[pre_start:self.current].copy()
        sample_df = self.sample_df[pre_start:self.current].copy()
        return df, sample_df

    def __next__(self):
        pre_start = self.current
        while self.current < self.len:
            self.current += 1
        if pre_start < self.current:
            return self.df, self.pred_df
        else:
            raise StopIteration()
try:
    iter_test = Iter_Valid(simu_test)
except:
    pass



# %%

def create_mlp(
    num_columns, num_labels, hidden_units, dropout_rates, label_smoothing, learning_rate
):

    inp = tf.keras.layers.Input(shape=(num_columns,))
    x = tf.keras.layers.BatchNormalization()(inp)
    x = tf.keras.layers.Dropout(dropout_rates[0])(x)
    for i in range(len(hidden_units)):
        x = tf.keras.layers.Dense(hidden_units[i])(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Activation(tf.keras.activations.swish)(x)
        x = tf.keras.layers.Dropout(dropout_rates[i + 1])(x)

    x = tf.keras.layers.Dense(num_labels)(x)
    out = tf.keras.layers.Activation("sigmoid")(x)

    model = tf.keras.models.Model(inputs=inp, outputs=out)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss=tf.keras.losses.BinaryCrossentropy(label_smoothing=label_smoothing),
        metrics=tf.keras.metrics.AUC(name="AUC"),
    )

    return model


hidden_units = [150, 150, 150]
dropout_rates = [0.2, 0.2, 0.2, 0.2]


model = create_mlp(
    len(features), 5, hidden_units, dropout_rates, label_smoothing, learning_rate
    )


model.load_weights(os.path.join(MODEL_DIR,f'model_{SEED}.hdf5'))
model.summary()
models = []
models.append(model)

#%%
date = simu_test['date'].values
weight = simu_test['weight'].values
resp = simu_test['resp'].values
action = simu_test['action'].values

# %%
with timer("Numba", compact=True):
    print(utility_score_numba(date, weight, resp, action))

with timer("numpy", compact=True):
    print(utility_score_bincount(date, weight, resp, action))

with timer("loops", compact=True):
    print(utility_score_loop(date, weight, resp, action))

with timer("Pandas", compact=True):
    print(utility_score_pandas(simu_test, labels = ['action', 'resp', 'weight', 'date']))

with timer("Pandas2", compact=True):
    print(utility_score_pandas2(simu_test))
# %% inference
'''
Current inference just use iterrows()
'''
f = np.mean
test_columns = ['weight'] + features + ['date']
predicted = []
def set_predict(df):
    predicted.append(df)

start = time()

pbar = tqdm(total=len(simu_test))
for idx, row in simu_test.iterrows():
    
    row = pd.DataFrame(row.values.reshape(1,-1), columns=list(row.index) )
    test_df = row[test_columns].astype(float)
    pred_df = row[['action']].astype(int)

    if test_df['weight'].item() > 0:
        x_tt = test_df.loc[:, features].values
        if np.isnan(x_tt[:, 1:].sum()):
            x_tt[:, 1:] = np.nan_to_num(x_tt[:, 1:]) + np.isnan(x_tt[:, 1:]) * f_mean
        pred = np.mean([model(x_tt, training = False).numpy() for model in models],axis=0)
        pred = f(pred.squeeze())
        pred_df.action = np.where(pred >= THRESHOLD, 1, 0).astype(int)
    else:
        pred_df.action = 0

    set_predict(pred_df)

    if idx % TQDM_INT == 0:
        time_taken = time() - start
        total_time_est = int(time_taken /TQDM_INT * 1000000 / 60)
        pbar.set_description(f"Current speed = {total_time_est} minutes to complete inference")
        start = time()
        pbar.update(TQDM_INT)


# %%
y_true = simu_test['action']
y_pred = pd.concat(predicted)['action']
print('\nValidation auc:', roc_auc_score(y_true, y_pred))
score = utility_score_bincount(date, weight, resp, y_true)
score_pred = utility_score_bincount(date, weight, resp, y_pred)
print('\nMax possible utility score:', score)
print('\nModel utility score:       ', score_pred)