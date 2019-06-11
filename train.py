import numpy as np
import pandas as pd
import tensorflow as tf
import keras
from sklearn.model_selection import train_test_split
import os
import argparse
from Config import Config
from data_generator import data_gen
from Model.model import traffic_demand_model, rmse


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-batch', default=64, type=int, help='Batch size')
    parser.add_argument('-epochs', default=10, type=int, help='Number of epochs to train')
    args = parser.parse_args()

    df = pd.read_csv(os.path.join(Config.DATA_DIR, Config.CLEANED_TRG_FILE))
    mor_df = df[df['Hour'] < 10]
    aft_df = df[(df['Hour'] >= 10) & (df['Hour'] < 20)]
    night_df = df[df['Hour'] >= 20]

    train_df, val_df = train_test_split(night_df, test_size=0.2, random_state=0)
    train_df = train_df.reset_index(drop=True, inplace=False)
    val_df = val_df.reset_index(drop=True, inplace=False)
    train_gen = data_gen(train_df, batch_size=args.batch)
    val_gen = data_gen(val_df, batch_size=args.batch)

    checkpoint_path = os.path.join(Config.MODEL_LOGS_DIR, Config.NIGHT_WEIGHTS)
    checkpoint_dir = os.path.dirname(checkpoint_path)
    cp_callback = tf.keras.callbacks.ModelCheckpoint(checkpoint_path,
                                                    save_weights_only=True,
                                                    save_best_only=True,
                                                    verbose=1)
    base_lr = 0.001

    def lr_linear_decay(epoch):
        return (base_lr * (1 - (epoch/args.epochs)))

    model = traffic_demand_model()
    model.compile(optimizer="rmsprop", loss=rmse, 
                metrics =["mean_squared_error"])
    history = model.fit_generator(generator=train_gen,
                                  validation_data=val_gen,
                                  steps_per_epoch=50,
                                  validation_steps=50,
                                  max_queue_size=10,
                                  epochs=args.epochs,
                                  callbacks=[tf.keras.callbacks.LearningRateScheduler(lr_linear_decay), cp_callback],
                                  verbose=1)