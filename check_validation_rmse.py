import tensorflow as tf
import keras 
from keras import backend as K
from Model.model_utils import fc_layer


def rmse(y_true, y_pred):
    return K.sqrt(K.mean(K.square(y_pred - y_true), axis=-1))

