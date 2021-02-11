#%%
import tensorflow_addons as tfa
import tensorflow as tf
import tensorflow.keras.backend as K

#%%
resp_cols = ['resp','resp_1', 'resp_2', 'resp_3', 'resp_4']
target_cols = ['action','action_1', 'action_2', 'action_3', 'action_4']


#%%
def mish(x):
    return tf.keras.layers.Lambda(lambda x: x*K.tanh(K.softplus(x)))(x)

tf.keras.utils.get_custom_objects().update({'mish': tf.keras.layers.Activation(mish)})

def create_model(input_shape):
    
    inp = tf.keras.layers.Input(input_shape)
    tmp = tf.keras.layers.BatchNormalization()(inp)
    xs = [tmp]
    for _ in range(5):
        if len(xs) > 1:
            tmp = tf.keras.layers.Concatenate(axis=-1)(xs)
        else:
            tmp = xs[0]
        # tmp = tf.keras.layers.Dense(128, activation='mish')(tmp)
        tmp = tf.keras.layers.Dense(128, activation='swish')(tmp)
        tmp = tf.keras.layers.BatchNormalization()(tmp)
        tmp = tf.keras.layers.Dropout(0.2)(tmp)
        xs.append(tmp)
    
    output = tf.keras.layers.Dense(len(resp_cols),activation='sigmoid')(tf.keras.layers.Concatenate()(xs))
    model = tf.keras.models.Model(inp,output)
    optimizer = tfa.optimizers.RectifiedAdam(1e-3)
    model.compile(optimizer, loss=tf.keras.losses.BinaryCrossentropy(label_smoothing=0.001),
                    metrics=[tf.keras.metrics.BinaryAccuracy(name='binary_accuracy')])
    return model
# %%
model = create_model(132)
model.summary()
# %%
