import tensorflow as tf
import keras 
from keras import backend as K


def dense_layer(_input, output_units, activation, batch_norm):
    net = tf.keras.layers.Dense(output_units)(_input)
    if activation:
        net = tf.keras.layers.Activation(activation)(net)
    if batch_norm:
        net = tf.keras.layers.BatchNormalization()(net)
    return net
