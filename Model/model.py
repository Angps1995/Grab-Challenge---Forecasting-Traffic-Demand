"""
Traffic Demand Model

Author: Ang Peng Seng
Date: June 2019
"""

import tensorflow as tf
import keras
from keras import backend as K
import sys
import os
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(ROOT_DIR)
from model_utils import fc_layer, conv2d


def rmse(y_true, y_pred):
    return K.sqrt(K.mean(K.square(y_pred - y_true), axis=-1))


def traffic_demand_model():
    _input = tf.keras.layers.Input(shape=(5, 5, 4))
    net = conv2d(inputs=_input, filters=16, kernel_size=(2, 2), strides=(1, 1),
                 padding='same')
    net = conv2d(inputs=net, filters=16, kernel_size=(2, 2), strides=(1, 1),
                 padding='valid')
    net = conv2d(inputs=net, filters=16, kernel_size=(2, 2), strides=(1, 1),
                 padding='valid')
    net = conv2d(inputs=net, filters=16, kernel_size=(2, 2), strides=(1, 1),
                 padding='valid')
    net = conv2d(inputs=net, filters=16, kernel_size=(2, 2), strides=(1, 1),
                 padding='valid')
    net = tf.keras.layers.Flatten()(net)
    net = fc_layer(net, 8, batch_norm=True)
    net = tf.keras.layers.Dense(1, kernel_initializer='normal')(net)
    net = tf.keras.layers.Activation('relu')(net)
    model = tf.keras.models.Model(inputs=_input, outputs=net)
    return model

if __name__ == '__main__':
    model = traffic_demand_model()
    model.summary()