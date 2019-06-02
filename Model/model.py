import tensorflow as tf
import keras 
from keras import backend as K
from model_utils import dense_layer


def Traffic_Demand_Model(input_feats):
    _input = tf.keras.layers.Input(shape=(input_feats,))
    net = tf.keras.layers.Dense(256)(_input)
    net = tf.keras.layers.Activation('relu')(net)
    net = tf.keras.layers.BatchNormalization()(net)
    net = tf.keras.layers.Dense(128)(net)
    net = tf.keras.layers.Activation('relu')(net)
    net = tf.keras.layers.BatchNormalization()(net)
    net = tf.keras.layers.Dense(64)(net)
    net = tf.keras.layers.Activation('relu')(net)
    net = tf.keras.layers.BatchNormalization()(net)
    net = tf.keras.layers.Dense(1, kernel_initializer='normal')(net)
    net = tf.keras.layers.Activation('sigmoid')(net)
    model = tf.keras.models.Model(inputs=_input, outputs=net)
    return model