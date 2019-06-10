"""
Data Generator for training

Author: Ang Peng Seng
Date: June 2019
"""

import numpy as np
import pandas as pd
from Feature_Engineering.get_nearest_loc import get_neigh_grid, get_demand
from Utils.utils import dayhourmin_to_period, period_to_dayhourmin


def data_gen(df, batch_size):
    max_rows = len(df)
    while True:
        feats = np.array([])
        curr_demands = np.array([])
        rows = np.random.choice(a=max_rows, size=batch_size, replace=False)
        list_target_lat = [df.loc[row, 'latitude'] for row in rows]
        list_target_long = [df.loc[row, 'longitude'] for row in rows]
        list_periods = [df.loc[row, 'Period'] for row in rows]
        feats = get_neigh_grid(df, list_target_lat, list_target_long, list_periods)
        curr_demands = np.array([])
        for i in range(5):
            curr_demand = np.array(get_demand(df, list_target_lat, 
                            list_target_long, [per + i for per in list_periods]))
            curr_demands = np.vstack((curr_demands, curr_demand)) if curr_demands.size else curr_demand

        feats = np.reshape(feats,(-1,5,5,4))
        feats = np.moveaxis(feats, -1, 1)
        yield np.reshape(feats,(-1,4, 5,5,1)), np.reshape(curr_demands, (-1,5))


# def data_gen(df, batch_size):
#     max_rows = len(df)
#     while True:
#         feats = np.array([])
#         curr_demands = np.array([])
#         rows = np.random.choice(a=max_rows, size=batch_size, replace=False)
# #         for i in range(batch_size):
#         for i in range(len(rows)):
#             #row = i + multiplier
#             row = rows[i]
#             lat = df.loc[row, 'latitude']
#             long = df.loc[row, 'longitude']
#             day = df.loc[row, 'day']
#             hour = df.loc[row, 'Hour']
#             minute = df.loc[row, 'Minute']
#             feat = get_neigh_grid(df, lat, long, day, hour, minute)
#             feats = np.vstack((feats, feat)) if feats.size else feat
            
#             curr_demand = np.array([get_demand(df, lat, long, dayhourmin_to_period(day, hour, minute) + j) for j in range(5)])
#             curr_demands = np.vstack((curr_demands, curr_demand)) if curr_demands.size else curr_demand

#         feats = np.reshape(feats,(-1,5,5,4))
#         feats = np.moveaxis(feats, -1, 1)
#         yield np.reshape(feats,(-1,4, 5,5,1)), np.reshape(curr_demands, (-1,5))
