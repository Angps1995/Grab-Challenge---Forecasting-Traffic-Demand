"""
Model Utility Functions

Author: Ang Peng Seng
Date: June 2019
"""

import tensorflow as tf
import keras
from keras import backend as K


def fc_layer(inputs, output_units, batch_norm):
    net = tf.keras.layers.Dense(output_units)(inputs)
    if batch_norm:
        net = tf.keras.layers.BatchNormalization()(net)
    net = tf.keras.layers.Activation('relu')(net)
    return net


def conv2d(inputs, filters, kernel_size,
           strides, padding):
    net = tf.keras.layers.Conv2D(filters=filters, kernel_size=kernel_size, strides=strides, padding=padding, 
                                 kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                 bias_initializer=tf.contrib.layers.xavier_initializer(),
                                 data_format='channels_last')(inputs)
    net = tf.keras.layers.BatchNormalization()(net)
    net = tf.keras.layers.Activation('relu')(net)
    return net