#%%
from tensorflow.keras.layers import Input, Dense, BatchNormalization, Dropout, Concatenate, Lambda, GaussianNoise, Activation
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.losses import BinaryCrossentropy
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers.experimental.preprocessing import Normalization
from tensorflow.keras.metrics import AUC
import tensorflow as tf
import kerastuner as kt
import numpy as np
import pandas as pd
import pickle
from sklearn.model_selection import GroupKFold

from tqdm import tqdm
from random import choices

import os, sys

HOME = os.path.abspath(os.path.join('.', os.pardir))
MODEL_DIR = os.path.join(HOME,  'models')
DATA_DIR = os.path.join(HOME,  'data')
sys.path.append(HOME) 
from utils import *
from utils_js import *
#%%
TRAINING = True
TRAINING_AE = True
HP_SEARCH = True
GPU = True
USE_FINETUNE = True
FOLDS = 5
SEED = 1127

if GPU:
    gpus = tf.config.experimental.list_physical_devices(device_type="GPU")
    tf.config.experimental.set_visible_devices(devices=gpus[0], device_type="GPU")
    tf.config.experimental.set_memory_growth(device=gpus[0], enable=True)

# %% loading data
with timer("Loading train parquet"):
    train_parquet = os.path.join(DATA_DIR, 'train.parquet')
    train = pd.read_parquet(train_parquet)
print(train.info())

# %%
with timer("preprocess train"):
    train = preprocess(train)

#%%
features = [c for c in train.columns if 'feature' in c]

resp_cols = ['resp_1', 'resp_2', 'resp_3', 'resp', 'resp_4']

X = train[features].values
y = np.stack([(train[c] > 0).astype('int') for c in resp_cols]).T #Multitarget

f_mean = np.mean(train[features[1:]].values,axis=0)
# %% AE

def create_autoencoder(input_dim, output_dim, noise=0.05, dropout=0.15):
    i = Input(input_dim)
    encoded = BatchNormalization()(i)
    encoded = GaussianNoise(noise)(encoded)
    encoded = Dense(64,activation='relu')(encoded)
    decoded = Dropout(dropout)(encoded)
    decoded = BatchNormalization()(decoded)
    decoded = Dense(input_dim,name='decoded')(decoded)
    x = Dense(32,activation='relu')(decoded)
    x = BatchNormalization()(x)
    x = Dropout(dropout)(x)
    x = Dense(32,activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(dropout)(x)    
    x = Dense(output_dim, activation='sigmoid', name='label_output')(x)
    
    encoder = Model(inputs=i,outputs=encoded)
    autoencoder = Model(inputs=i,outputs=[decoded,x])
    
    autoencoder.compile(optimizer=Adam(0.001),
                        loss={'decoded':'mse',
                              'label_output':'binary_crossentropy'})
    return autoencoder, encoder

def create_model(hp,input_dim,output_dim,encoder):
    inputs = Input(input_dim)
    
    x = encoder(inputs)
    x = Concatenate()([x,inputs, x]) #use both raw and encoded features
    x = BatchNormalization()(x)
    x = Dropout(hp.Float('init_dropout',0.0,0.5))(x)
    
    for i in range(hp.Int('num_layers',1,5)):
        x = Dense(hp.Int(f'num_units_{i}',64,256))(x)
        x = BatchNormalization()(x)
        x = Lambda(tf.keras.activations.swish)(x)
        x = Dropout(hp.Float(f'dropout_{i}',0.0,0.5))(x)
    x = Dense(output_dim,activation='sigmoid')(x)
    model = Model(inputs=inputs,outputs=x)
    model.compile(optimizer=Adam(hp.Float('lr',0.00001,0.1,
                                default=0.001)),
                                loss=BinaryCrossentropy(label_smoothing=hp.Float('label_smoothing',0.0,0.1)),
                                metrics=[AUC(name = 'auc')])
    return model
# %%
autoencoder, encoder = create_autoencoder(X.shape[-1],y.shape[-1],noise=0.1)
if TRAINING_AE:
    autoencoder.fit(X, (X,y),
                    epochs=1000,
                    batch_size=4096*2, 
                    validation_split=0.1,
                    callbacks=[EarlyStopping('val_loss',
                               patience=10,
                               restore_best_weights=True)])
    encoder.save_weights(MODEL_DIR+'/encoder.hdf5')
else:
    encoder.load_weights(MODEL_DIR+'/encoder.hdf5')

encoder.trainable = True

#%%

class CVTuner(kt.engine.tuner.Tuner):
    def run_trial(self, trial, X, y, splits, batch_size=32, verbose=2, epochs=1, callbacks=None):
        val_losses = []
        for idx_tr, idx_val in splits:
            X_train, X_val = [x[idx_tr] for x in X], [x[idx_val] for x in X]
            y_train, y_val = [a[idx_tr] for a in y], [a[idx_val] for a in y]
            if len(X_train) < 2:
                X_train = X_train[0]
                X_val = X_val[0]
            if len(y_train) < 2:
                y_train = y_train[0]
                y_val = y_val[0]
            
            model = self.hypermodel.build(trial.hyperparameters)
            hist = model.fit(X_train,y_train,
                      validation_data=(X_val,y_val),
                      epochs=epochs,
                      batch_size=batch_size,
                      callbacks=callbacks,
                      verbose=verbose)
            
            val_losses.append([hist.history[k][-1] for k in hist.history])

        val_losses = np.asarray(val_losses)
        self.oracle.update_trial(trial.trial_id, 
                {k:np.mean(val_losses[:,i]) for i,k in enumerate(hist.history.keys())})
        self.save_model(trial.trial_id, model)

model_fn = lambda hp: create_model(hp,X.shape[-1],y.shape[-1], encoder)

tuner = CVTuner(
        hypermodel=model_fn,
        directory=f'ae_mlp_{SEED}',
        oracle=kt.oracles.BayesianOptimization(
        objective= kt.Objective('val_auc', direction='max'),
        num_initial_points=10,
        max_trials=50))

gkf = PurgedGroupTimeSeriesSplit(n_splits = FOLDS, group_gap=5)
splits = list(gkf.split(y, groups=train['date'].values))
#%%
if HP_SEARCH:
    tuner.search((X,),(y,),
                 splits=splits,
                 batch_size=8192,
                 epochs=50,
                 verbose=2,
                 callbacks=[EarlyStopping('val_auc', 
                                          mode='max',
                                          patience=5)])
    hp  = tuner.get_best_hyperparameters(1)[0]

    with open(MODEL_DIR+f'/best_hp_{SEED}.pkl', 'wb') as f:
        pickle.dump(hp, f, protocol=pickle.HIGHEST_PROTOCOL)
    tuner.results_summary()
#%%
if TRAINING:
    with open(MODEL_DIR+f'/best_hp_{SEED}.pkl', 'rb') as f:
        hp = pickle.load(f)

    for fold, (idx_tr, idx_val) in enumerate(splits):
        model = model_fn(hp)
        X_train, X_val = X[idx_tr], X[idx_val]
        y_train, y_val = y[idx_tr], y[idx_val]
        model.fit(X_train,
                  y_train,
                  validation_data=(X_val,y_val),
                  epochs=100, 
                  batch_size=8192,
                  callbacks=[EarlyStopping('val_auc',
                                           mode='max',
                                           patience=10,
                                           restore_best_weights=True)])
        model.save_weights(MODEL_DIR + f'/model_{SEED}_{fold}.hdf5')
        model.compile(Adam(hp.get('lr')/100),loss='binary_crossentropy')

        model.fit(X_val, y_val, epochs=3, batch_size=8192)
        model.save_weights(MODEL_DIR+f'/model_{SEED}_{fold}_finetune.hdf5')
    
else:
    models = []
    hp = pd.read_pickle(MODEL_DIR+f'/best_hp_{SEED}.pkl')
    for f in range(FOLDS):
        model = model_fn(hp)
        if USE_FINETUNE:
            model.load_weights(MODEL_DIR+f'/model_{SEED}_{f}_finetune.hdf5')
        else:
            model.load_weights(MODEL_DIR+f'/model_{SEED}_{f}.hdf5')
        models.append(model)
# %%
