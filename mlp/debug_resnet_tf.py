#%%
from tensorflow.keras.layers import Input, Dense, BatchNormalization, Dropout, Concatenate, Lambda, GaussianNoise, Activation
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.losses import BinaryCrossentropy
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, Callback
from tensorflow.keras.layers.experimental.preprocessing import Normalization
import tensorflow as tf
import tensorflow_addons as tfa
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
from mlp import *

# %%
'''
baseline, dropped outlier days, fillna with mean, drop weight zero trades after. Using a feature split based on Carl's notebook. "Minor" features go through a linear layer block with high dropout rate first. Epoch = 50

Added a util score callback for keras fit API, epoch 80, the util score is for every 50 days after day 100. This model reaches 5k util in the last 50 days in under 50 epochs, too good to be true?
'''

SEED = 1127802
BETA = 0.7 # 5 preds then the middle 3

# split features for a ResNet feature 2 is more important
features_2_list = [0, 1, 2, 3, 4, 5, 6, 15, 16, 25, 26, 35, 
             36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 
             49, 50, 51, 52, 53, 54, 59, 60, 61, 62, 63, 64, 65, 
             66, 67, 68, 69, 70, 71, 76, 77, 82, 83, 88, 89, 94, 
             95, 100, 101, 106, 107, 112, 113, 118, 119, 128, 129]

features_1_list = [0] + list(set(range(130)).difference(features_2_list))

features_1 = [f'feature_{i}' for i in features_1_list]

features_2 = [f'feature_{i}' for i in features_2_list]

# %%
all_train = pd.read_parquet(DATA_DIR+'train.parquet')
all_train = all_train.query('date > 85').reset_index(drop = True) 
all_train = all_train.query('date not in [2, 36, 270, 294]').reset_index(drop=True)

all_train.fillna(all_train.mean(), inplace=True)

features = [f'feature_{i}' for i in range(130)]
f_mean = np.mean(all_train[features].values,axis=0)
# np.save('f_mean_after_85_include_zero_weight.npy', f_mean)

all_train = all_train[all_train['weight'] != 0].reset_index(drop=True)

all_train = all_train.astype({feat: np.float32 for feat in features})
#%%
_fold = 4
split = [('date > 450','date <= 450'),
         ('date <= 450 and date > 400','date <= 400 or date>450'),
         ('date <= 400 and date > 350','date <= 350 or date>400'),
         ('date <= 350 and date > 300','date <= 300 or date>350'),
         ('date <= 300 and date > 250','date <= 250 or date>300'),
         ('date <= 250 and date > 200','date <= 200 or date>250'),
         ('date <= 200 and date > 150','date <= 150 or date>200'),
         ('date <= 150 and date > 100','date <= 100 or date>150'),]

valid = all_train.query(split[_fold][0]).reset_index(drop = True)
train = all_train.query(split[_fold][1]).reset_index(drop = True)  

resp_cols = ['resp_1', 'resp_2', 'resp_3', 'resp', 'resp_4']

y_train = np.stack([(train[c] > 0).astype('int') for c in resp_cols]).T
y_val = np.stack([(valid[c] > 0).astype('int') for c in resp_cols]).T

X_train = [train.loc[:, features_1].values, 
           train.loc[:, features_2].values]
X_val = [valid.loc[:, features_1].values, 
           valid.loc[:, features_2].values]

print(len(train), len(valid))
# %%
class Mish(tf.keras.layers.Layer):

    def __init__(self, **kwargs):
        super(Mish, self).__init__(**kwargs)
        self.supports_masking = True

    def call(self, inputs):
        return inputs * K.tanh(K.softplus(inputs))

    def get_config(self):
        base_config = super(Mish, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def compute_output_shape(self, input_shape):
        return input_shape

def mish(x):
	return tf.keras.layers.Lambda(lambda x: x*K.tanh(K.softplus(x)))(x)


tf.keras.utils.get_custom_objects().update({'mish': tf.keras.layers.Activation(mish)})

def create_resnet(n_features, n_features_2, n_labels, hidden_size, 
                  learning_rate=1e-3, label_smoothing = 0.005):    
    input_1 = tf.keras.layers.Input(shape = (n_features,), name = 'Input1')
    input_2 = tf.keras.layers.Input(shape = (n_features_2,), name = 'Input2')

    head_1 = tf.keras.Sequential([
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.4),
        tf.keras.layers.Dense(hidden_size, activation="mish"), 
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.4),
        tf.keras.layers.Dense(hidden_size//2, activation = "mish")
        ],name='Head1') 

    input_3 = head_1(input_1)
    input_3_concat = tf.keras.layers.Concatenate()([input_2, input_3])

    head_2 = tf.keras.Sequential([
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(hidden_size, "mish"),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(hidden_size, "mish"),
        ],name='Head2')

    input_4 = head_2(input_3_concat)
    input_4_concat = tf.keras.layers.Concatenate()([input_3, input_4]) 

    head_3 = tf.keras.Sequential([
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dense(hidden_size, kernel_initializer='lecun_normal', activation='mish'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(hidden_size//2, kernel_initializer='lecun_normal', activation='mish'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(n_labels, activation="sigmoid")
        ],name='Head3')

    output = head_3(input_4_concat)


    model = tf.keras.models.Model(inputs = [input_1, input_2], outputs = output)
    model.compile(optimizer=tfa.optimizers.RectifiedAdam(learning_rate=learning_rate), 
                  loss=tf.keras.losses.BinaryCrossentropy(label_smoothing=label_smoothing), 
                  metrics=['AUC'])
    
    return model

class UtilEvaluation(Callback):
    def __init__(self, val_df=None, interval=3, start_day=100, end_day=500, num_days=50):
        super(UtilEvaluation, self).__init__()

        self.interval = interval
        self.val_df = val_df
        self.start_day = start_day
        self.end_day = end_day
        self.num_days = num_days

    def on_epoch_end(self, epoch, logs={}):
        if (epoch+1) % self.interval == 0:
            print("*"*40)
            print(f"Epoch [{epoch+1:d}/{EPOCHS}]:")
            all_score = []
            all_val_pred = self.val_df[['date', 'weight', 'resp']].copy()
            all_val_pred['action'] = 0

            for day in range(self.start_day, self.end_day, self.num_days):
                valid = self.val_df[self.val_df.date.isin(range(day, day+self.num_days))]
                valid = valid[valid.weight > 0]

                x_tt = valid.loc[:, features].values
                x_tt_1 = x_tt.take(features_1_list, axis=-1)
                x_tt_2 = x_tt.take(features_2_list, axis=-1)
                val_pred = self.model([x_tt_1, x_tt_2], training = False).numpy()
                val_pred = median_avg(val_pred)
                val_pred = np.where(val_pred >= 0.5, 1, 0).astype(int)
                valid_score = utility_score_bincount(date=valid.date.values, 
                                                 weight=valid.weight.values,
                                                 resp=valid.resp.values, 
                                                 action=val_pred)
                all_score.append(valid_score)
                all_val_pred.loc[self.val_df.date.isin(range(day, day+self.num_days)), 'action']=val_pred
                all_val_pred.to_csv(f'val_pred_fold_{_fold}.csv', index=False)
                print(f"Day {day:3d}-{day+self.num_days-1:3d} - util score: {valid_score:.2f}")
            
            print(f"Utility score mean with {self.num_days} span: {np.mean(all_score):.2f} ")
            print(f"Utility score std with {self.num_days} span: {np.std(all_score):.2f}")
            print("*"*40, '\n')

#%%
tf.keras.backend.clear_session()
SEED = 1127
get_seed(SEED)
tf_model = create_resnet(len(features_1), len(features_2), len(resp_cols), 
                         hidden_size=256, learning_rate=1e-4,label_smoothing=5e-03)
util_cb = UtilEvaluation(val_df=valid, start_day=valid.date.min(), end_day=valid.date.max())
tf_model.summary()
# %%
EPOCHS = 45
get_seed(SEED)
tf_model.fit(X_train, y_train, epochs=EPOCHS, batch_size=8192, 
             validation_data=(X_val, y_val),
             verbose=1,
             callbacks=[util_cb]
            )

# save model
tf_model.save(f'tf_res_fold_{_fold}_ep_{EPOCHS}.h5')
# %%
