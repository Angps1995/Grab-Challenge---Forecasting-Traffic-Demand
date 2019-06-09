import numpy as np
import pandas as pd
import tensorflow as tf
import keras
import os
import argparse
from Config import Config
from data_generator import data_gen
from Model.model import traffic_demand_model

base_lr = 0.001


def lr_linear_decay(epoch):
    return (base_lr * (1 - (epoch/num_epochs)))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-batch', default=4, type=int, help='Batch size')
    parser.add_argument('-epochs', default=5, type=int, help='Number of epochs to train')
    args = parser.parse_args()
    train_gen = data_gen('train', batch_size=args.batch)
    val_gen = data_gen('val', batch_size=args.batch)

    checkpoint_path = os.path.join(Config.MODEL_LOGS_DIR, Config.WEIGHTS)
    checkpoint_dir = os.path.dirname(checkpoint_path)

    cp_callback = tf.keras.callbacks.ModelCheckpoint(checkpoint_path,
                                                    save_weights_only=True,
                                                    save_best_only=True,
                                                    verbose=1)
    model = traffic_demand_model()
    history = model.fit_generator(generator=train_gen
                                  validation_data=val_gen,
                                  steps_per_epoch=50,
                                  validation_steps=50,
                                  max_queue_size=20,
                                  epochs=args.epochs,
                                  callbacks=[tf.keras.callbacks.LearningRateScheduler(lr_linear_decay), cp_callback],
                                  verbose=1)