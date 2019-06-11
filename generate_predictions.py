import numpy as np
import pandas as pd
import tensorflow as tf
import keras
import geohash2
from Model.model import traffic_demand_model
import os
import argparse
from Config import Config
from Feature_Engineering.get_nearest_loc import get_neigh_grid, get_demand
from Utils.utils import dayhourmin_to_period, period_to_dayhourmin
from Utils.preprocess_data import clean_data, parrellize_clean_data


def predict_demand(df, list_geohashes):
    # return np.array([['x',1,1,1,1], ['y',2,2,2,2],
    #                 ['z',3,3,3,3], ['a',4,4,4,4], ['b',5,5,5,5]] etc)
    temp_df = df[df['geohash6'].isin(list_geohashes)]

    # temp_df = temp_df.reset_index(drop=True, inplace=False)
    # latest_per = temp_df['Period'].max()
    # temp_df = temp_df[temp_df['Period'] == latest_per]
    # lat = temp_df.get('latitude').values[0]
    # long = temp_df.get('longitude').values[0]
    # period = temp_df.get('Period').values[0]
    # feat = get_neigh_grid(temp_df, lat, long, period + 1)
    # #pred = model.predict(np.reshape(feat, (-1, 5, 5, 4)))[0]
    # pred = np.array([1,2,3,4,5])
    # row_pred = np.array([])
    # for i in range(len(pred)):
    #     temp_per = period + i + 1
    #     day, hour, minute = period_to_dayhourmin(temp_per)
    #     row = np.array([geohash, day, hour, minute, pred[i]])
    #     row_pred = np.vstack((row_pred, row)) if row_pred.size else row
    # return row_pred


def predict_all_future_demand(df):

    # 1) get list of all the geohashes in the dataset
    geohash_list = list(df.geohash6.unique()) 
    num_geohash = len(geohash_list)
    # 2) create new df 
    prediction_df = pd.DataFrame(columns=['Geohash6', 'day', 'Hour', 'Minute', 'Predicted Demand'])

    # 3) predict for geohash
    for batch in range(len(num_geohash)//128):
        list_geohashes = geohash_list[batch:batch+128]
        pred = predict_demand(df, list_geohashes)
        series_pred = [pd.Series(p, index=prediction_df.columns)
                       for p in pred]
        prediction_df = prediction_df.append(series_pred, ignore_index=True)
    return prediction_df

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-file', required=True, help='Full file Path')
    parser.add_argument('-output', default='Predictions.csv', help='Output File Name')
    args = parser.parse_args()

    print('Reading input file')
    try:
        df = pd.read_csv(args.file)
        print("Preprocessing dataset")
        df = parrellize_clean_data(df, clean_data)
        print("Processing Done")
        night_df = df[(df['day']<=25) & (df['Hour'] >=20)]
    except:
        raise Exception('Check your input test data csv file path again.')

    print('Loading Model...')
    model = traffic_demand_model()
    model.load_weights(os.path.join(Config.MODEL_LOGS_DIR, Config.NIGHT_WEIGHTS))
    print('Model Loaded Successfully')

    print('Dataset loaded successfully. Predicting now...')
    prediction_df = predict_all_future_demand(night_df)
    print('Prediction done')
    prediction_df.to_csv(args.output)
    print('Predictions csv saved to --> ' + str(args.output))