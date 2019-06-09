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
        for i in range(len(rows)):
            row = rows[i]
            lat = df.loc[row, 'latitude']
            long = df.loc[row, 'longitude']
            period = df.loc[row, 'Period']
            feat = get_neigh_grid(df, lat, long, period)
            feats = np.vstack((feats, feat)) if feats.size else feat
            
            #prev_16_dem = [get_demand(df, lat, long, dayhourmin_to_period(day, hour, minute) - i) for i in range(1, 17)]
            curr_demand = np.array([get_demand(df, lat, long, period + i) for i in range(5)])
            curr_demands = np.vstack((curr_demands, curr_demand)) if curr_demands.size else curr_demand

        # feats = np.reshape(feats,(-1,5,5,4))
        #feats = np.moveaxis(feats, -1, 1)
        #yield np.reshape(feats,(-1,4, 5,5,1)), np.reshape(curr_demands, (-1,5))
        yield np.reshape(feats,(-1, 5, 5, 4)), np.reshape(curr_demands, (-1, 5))