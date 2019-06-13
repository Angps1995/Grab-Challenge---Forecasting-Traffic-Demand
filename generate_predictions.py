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
    test_df = df[df['geohash6'].isin(list_geohashes)]
    test_df = test_df.reset_index(drop=True, inplace=False)
    max_idx = test_df.groupby(['geohash6'])['Period'].idxmax().values
    test_df = test_df.iloc[max_idx, :]
    list_lat = list(test_df['latitude'].values)
    list_long = list(test_df['longitude'].values)
    list_period = list(test_df['Period'].values)
    feats = get_neigh_grid(df, list_lat, list_long, list_period)
    feats = np.reshape(feats,(-1,5,5,4))
    feats = np.moveaxis(feats, -1, 1)
    pred = model.predict(np.reshape(feats, (len(list_geohashes), 4, 5, 5, 1)))
    pred = pred.flatten()
    row_pred = np.array([])
    for i in range(len(pred)):
        temp_per = list_period[i] + 1
        day, hour, minute = period_to_dayhourmin(temp_per)
        geohash = geohash2.encode(list_lat[i], list_long[i], 6)
        row = np.array([geohash, list_lat[i], list_long[i], day, hour, minute, temp_per, pred[i]])
        row_pred = np.vstack((row_pred, row)) if row_pred.size else row
    return row_pred


def predict_all_future_demand(df):

    # 1) get list of all the geohashes in the dataset
    geohash_list = list(df.geohash6.unique()) 
    df = df[['geohash6', 'latitude', 'longitude', 'day', 'Hour', 'Minute', 'Period','demand']]
    
    # 2) create pred df 
    prediction_df = pd.DataFrame(columns=['geohash6', 'latitude', 'longitude', 'day', 'Hour', 'Minute', 'Period', 'Predicted Demand'])
    predictions = np.array([])
    # 3) predict for geohash
    for i in range(5):
        predictions = predict_demand(df, geohash_list)
        series_pred = [pd.Series(p, index=prediction_df.columns)
                       for p in predictions]
        series_pred_1 = [pd.Series(p, index=df.columns)
                               for p in predictions]
        df = df.append(series_pred_1, ignore_index=True)
        prediction_df = prediction_df.append(series_pred, ignore_index=True)
        for col in df.columns.difference(['geohash6', 'demand', 'latitude', 'longitude']):
            df[col] = df[col].astype(int)
        for col in ['demand', 'latitude', 'longitude']:
            df[col] = df[col].astype(float)
        df['demand'] = df['demand'].astype(float)
    prediction_df['geohash6'] = prediction_df['geohash6'].astype(str)
    for col in prediction_df.columns.difference(['geohash6', 'Predicted Demand', 'latitude', 'longitude']):
        prediction_df[col]=prediction_df[col].astype(int)
    for col in ['Predicted Demand', 'latitude', 'longitude']:
        prediction_df[col] = prediction_df[col].astype(float)
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
        night_df = df[(df['day'] <= 24) & (df['Hour'] >=20)]
    except:
        raise Exception('Check your input test data csv file path again.')

    print('Loading Model...')
    model = traffic_demand_model()
    model.load_weights(os.path.join(Config.MODEL_LOGS_DIR, Config.WEIGHTS))
    print('Model Loaded Successfully')

    print('Dataset loaded successfully. Predicting now...')
    prediction_df = predict_all_future_demand(night_df)
    print('Prediction done')
    prediction_df.to_csv(args.output, index=False)
    print('Predictions csv saved to --> ' + str(args.output))