"""
Script containing functions for feature extraction

Author: Ang Peng Seng
Date: June 2019
"""

import numpy as np
import geohash2
import pandas as pd
from multiprocessing import Pool, cpu_count
import sys
import os
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(ROOT_DIR)
from Config import Config
from Utils.utils import dayhourmin_to_period, period_to_dayhourmin


def get_demand(df, list_lat, list_long, list_period):
    list_geohash = [geohash2.encode(lat, long, 6)
                for lat, long in zip(list_lat, list_long)]
    query = df[(df['Period'].isin(list_period)) &
               (df['geohash6'].isin(list_geohash))]
    query = query.set_index(['geohash6', 'Period'])['demand'].to_dict()
    demands = []
    for geohash, per in zip(list_geohash, list_period):
        if (geohash, per) in query.keys():
            demands.append(query[(geohash, per)])
        else:
            demands.append(0)
    return demands


def get_neigh_grid(df, list_target_lat, list_target_long, list_period,
                   lat_diff=0.0054931640625, long_diff=0.010986328125):

    assert len(list_target_lat) == len(list_target_long) == len(list_period)
    batch_size = len(list_target_lat)
    grid = np.zeros((batch_size, 5, 5, 4))
    for r in range(grid.shape[1]):
        for c in range(grid.shape[2]):
            per_1 = get_demand(df,
                            [lat + ((c-2) * lat_diff) for lat in list_target_lat],
                            [long + ((2-r) * long_diff) for long in list_target_long],
                            [per - 1 for per in list_period])
            per_1day = get_demand(df,
                            [lat + ((c-2) * lat_diff) for lat in list_target_lat],
                            [long + ((2-r) * long_diff) for long in list_target_long],
                            [per - 96 for per in list_period])
            per_1week = get_demand(df, 
                            [lat + ((c-2) * lat_diff) for lat in list_target_lat],
                            [long + ((2-r) * long_diff) for long in list_target_long],
                            [per - 96*7 for per in list_period])
            per_2week = get_demand(df, 
                            [lat + ((c-2) * lat_diff) for lat in list_target_lat],
                            [long + ((2-r) * long_diff) for long in list_target_long],
                            [per - 96*14 for per in list_period])
            for b in range(batch_size):
                grid[b][r][c][0] = per_1[b]
                grid[b][r][c][1] = per_1day[b]
                grid[b][r][c][2] = per_1week[b]
                grid[b][r][c][3] = per_2week[b]
    return grid
