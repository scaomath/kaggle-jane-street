#%%
from tensorflow.keras.layers import Input, Dense, BatchNormalization, Dropout, Concatenate, Lambda, GaussianNoise, Activation
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.losses import BinaryCrossentropy
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, Callback, ReduceLROnPlateau
from tensorflow.keras.layers.experimental.preprocessing import Normalization
import tensorflow as tf
import tensorflow_addons as tfa

from utils_js import *
# %%
from tensorflow.keras import backend as K

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
# %%

def create_resnet(n_features, n_features_2, n_labels, hidden_size, learning_rate=1e-3, 
                  label_smoothing = 0.005):    
    '''
    Unimportant features go through a block with high dropout during training
    '''
    input_1 = tf.keras.layers.Input(shape = (n_features,), name = 'Input1')
    input_2 = tf.keras.layers.Input(shape = (n_features_2,), name = 'Input2')

    head_1 = tf.keras.Sequential([
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(hidden_size, activation="mish"), 
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.5),
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
        tf.keras.layers.Dense(hidden_size//2, "mish"),
        ],name='Head2')

    input_4 = head_2(input_3_concat)
    input_4_concat = tf.keras.layers.Concatenate()([input_3_concat, input_4]) 

    head_3 = tf.keras.Sequential([
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dense(hidden_size, kernel_initializer='lecun_normal', activation='mish'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(n_labels, activation="sigmoid")
        ],name='Head3')

    output = head_3(input_4_concat)


    model = tf.keras.models.Model(inputs = [input_1, input_2], outputs = output)
    model.compile(optimizer=tfa.optimizers.RectifiedAdam(learning_rate=learning_rate), 
                  loss=tf.keras.losses.BinaryCrossentropy(label_smoothing=label_smoothing), 
                  metrics=['AUC'])
    
    return model


def create_spikenet(n_features, n_features_2, n_cat_features, n_labels, hidden_size, 
                  learning_rate=1e-3, label_smoothing = 0.005):    
    '''
    Cat features going through a small dense as an embedding block
    '''
    input_1 = tf.keras.layers.Input(shape = (n_features,), name = 'Input1')
    input_2 = tf.keras.layers.Input(shape = (n_features_2,), name = 'Input2')
    input_3 = tf.keras.layers.Input(shape = (n_cat_features,), name = 'Input3')
    
    head_1 = tf.keras.Sequential([
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(hidden_size, activation="mish"), 
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(hidden_size//2, activation = "mish")
        ],name='Head1') 
    
    head_c = tf.keras.Sequential([
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.1),
        tf.keras.layers.Dense(n_cat_features//2, activation = "mish")
        ],name='HeadC') 

    input_4 = head_1(input_1)
    input_5 = head_c(input_3)
    input_5_concat = tf.keras.layers.Concatenate()([input_2, input_4, input_5])

    head_2 = tf.keras.Sequential([
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(hidden_size, "mish"),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(hidden_size, "mish"),
        ],name='Head2')

    input_6 = head_2(input_5_concat)
    input_6_concat = tf.keras.layers.Concatenate()([input_5_concat, input_6]) 

    head_3 = tf.keras.Sequential([
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dense(hidden_size, kernel_initializer='lecun_normal', activation='mish'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(hidden_size//2, kernel_initializer='lecun_normal', activation='mish'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(n_labels, activation="sigmoid")
        ],name='Head3')

    output = head_3(input_6_concat)


    model = tf.keras.models.Model(inputs = [input_1, input_2, input_3], outputs = output)
    model.compile(optimizer=tfa.optimizers.RectifiedAdam(learning_rate=learning_rate), 
                  loss=tf.keras.losses.BinaryCrossentropy(label_smoothing=label_smoothing), 
                  metrics=['AUC'])
    
    return model
# %%
class UtilEvaluation(Callback):
    def __init__(self, epochs=100, val_df=None, interval=3, start_day=100, num_days=50, 
                       features_list=None):
        super(UtilEvaluation, self).__init__()

        self.interval = interval
        self.val_df = val_df
        self.start_day = start_day
        self.num_days = num_days
        self.features = features_list[0]
        self.features_1 = features_list[1]
        self.features_2 = features_list[2]
        self.epochs = epochs

    def on_epoch_end(self, epoch, logs={}):
        if (epoch+1) % self.interval == 0:
            print("*"*40)
            print(f"Epoch [{epoch+1:d}/{self.epochs}]:")
            all_score = []
            for day in range(self.start_day, 500-self.num_days, self.num_days):
                valid = self.val_df[self.val_df.date.isin(range(day, day+self.num_days))]
                valid = valid[valid.weight > 0]

                x_tt = valid.loc[:, self.features].values
                x_tt_1 = x_tt[...,self.features_1]
                x_tt_2 = x_tt[...,self.features_2]
                val_pred = self.model([x_tt_1, x_tt_2], training = False).numpy()
                val_pred = median_avg(val_pred)
                val_pred = np.where(val_pred >= 0.5, 1, 0).astype(int)
                valid_score = utility_score_bincount(date=valid.date.values, 
                                                 weight=valid.weight.values,
                                                 resp=valid.resp.values, 
                                                 action=val_pred)
                all_score.append(valid_score)
                print(f"Day {day:3d}-{day+self.num_days-1:3d} - util score: {valid_score:.2f}")
            
            print(f"Utility score mean with {self.num_days} span: {np.mean(all_score):.2f} ")
            print(f"Utility score std with {self.num_days} span: {np.std(all_score):.2f}")
            print("*"*40, '\n')