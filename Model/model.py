"""
Traffic Demand Model

Author: Ang Peng Seng
Date: June 2019
"""

import tensorflow as tf
import keras 
from keras import backend as K
from Model.model_utils import fc_layer


def rmse(y_true, y_pred):
    return K.sqrt(K.mean(K.square(y_pred - y_true), axis=-1))


def traffic_demand_model():
    _input = tf.keras.layers.Input(shape=(5, 5, 4))
    net = tf.keras.layers.Conv2D(filters=64, kernel_size=(2,2), strides=(1, 1), padding='same', 
                                 kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                 bias_initializer=tf.contrib.layers.xavier_initializer(),
                                    data_format='channels_last')(_input)
    #net = tf.keras.layers.BatchNormalization()(net)
    net = tf.keras.layers.Activation('relu')(net)
    net = tf.keras.layers.Conv2D(filters=64, kernel_size=(2,2), strides=(1, 1), padding='same', 
                                 kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                 bias_initializer=tf.contrib.layers.xavier_initializer(),
                                    data_format='channels_last')(net)
    #net = tf.keras.layers.BatchNormalization()(net)
    net = tf.keras.layers.Activation('relu')(net)
    net = tf.keras.layers.Conv2D(filters=32, kernel_size=(2,2), strides=(1, 1), padding='same', 
                                 kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                 bias_initializer=tf.contrib.layers.xavier_initializer(),
                                    data_format='channels_last')(net)
    #net = tf.keras.layers.BatchNormalization()(net)
    net = tf.keras.layers.Activation('relu')(net)
    net = tf.keras.layers.MaxPool2D(pool_size=(1,5))(net)
    net = tf.keras.layers.MaxPool2D(pool_size=(5,1))(net)
    net = tf.keras.layers.Flatten()(net)
    net = fc_layer(net, 128, batch_norm=True)
    net = fc_layer(net, 16, batch_norm=True)
    net = tf.keras.layers.Dense(5, kernel_initializer='normal')(net)
    net = tf.keras.layers.Activation('relu')(net)
    model = tf.keras.models.Model(inputs=_input, outputs=net)
    return model

