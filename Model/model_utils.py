import tensorflow as tf
import keras 
from keras import backend as K


def fc_layer(inputs, output_units, batch_norm):
    net = tf.keras.layers.Dense(output_units)(inputs)
    if batch_norm:
        net = tf.keras.layers.BatchNormalization()(net)
    net = tf.keras.layers.Activation('relu')(net)
    return net


def conv2d(inputs, out_filters, kernel_size,
           strides, activation=tf.nn.relu):
    net = tf.keras.layers.Conv2D(out_filters,
            kernel_size, strides,
            kernel_initializer=tf.contrib.layers.xavier_initializer(),
            bias_initializer=tf.contrib.layers.xavier_initializer())(inputs)
    net = tf.keras.layers.BatchNormalization()(net)
    net = tf.keras.layers.Activation('relu')(net)
    return net