"""
Data Generator for training

Author: Ang Peng Seng
Date: June 2019
"""

import numpy as np
import pandas as pd
from Feature_Engineering.feature_extraction import get_neigh_grid, get_demand
from Utils.utils import dayhourmin_to_period, period_to_dayhourmin


def data_gen(df, batch_size):
    df_1 = df[(df['day'] > 14) &
              (df['day'] < 60)].reset_index(drop=True, inplace=False)
    max_rows = len(df_1)
    while True:
        feats = np.array([])
        rows = np.random.choice(a=max_rows, size=batch_size, replace=False)
        list_target_lat = [df_1.loc[row, 'latitude'] for row in rows]
        list_target_long = [df_1.loc[row, 'longitude'] for row in rows]
        list_periods = [df_1.loc[row, 'Period'] for row in rows]
        feats = get_neigh_grid(df, list_target_lat,
                               list_target_long, list_periods)
        actual_demand = np.array(get_demand(
                            df, list_target_lat,
                            list_target_long,
                            [per for per in list_periods]))

        yield np.reshape(feats, (-1, 5, 5, 4)), np.reshape(actual_demand, (-1, 1))

