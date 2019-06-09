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


def get_demand(df, lat, long, period):
    day, hour, minute = period_to_dayhourmin(period)
    geohash = geohash2.encode(lat, long, 6)
    demand_queried = df[(df['Period'] == period) &
                        (df['geohash6'] == geohash)]['demand'].values
    if len(demand_queried) > 0:
        return demand_queried[0]
    else:
        return 0


def get_neigh_grid(df, target_lat, target_long, period,
                   lat_diff=0.0054931640625, long_diff=0.010986328125):
    '''
    Inputs:
    target_lat, target_long, day, hour, minute all in integers
    
    Outputs:
    Function will return a 5x5x4 grid where the target loc will 
    be in the middle of the grid with its closest neighbors surrounding.
    Each grid 'cell' will contain the aggregated demand of the previous period
    
    Latitude will be along the x-axis and Longitude along the y-axis
    '''
    # period = dayhourmin_to_period(day, hour, minute)
    grid = np.zeros((5,5,4))
    for r in range(grid.shape[0]):
        for c in range(grid.shape[1]):
            grid[r][c][0] = get_demand(df, target_lat + ((c-2) * lat_diff),
                                    target_long + ((2-r) * long_diff),
                                    period - 1)
            grid[r][c][1] = get_demand(df, target_lat + ((c-2) * lat_diff),
                                    target_long + ((2-r) * long_diff),
                                    period - 96)
            grid[r][c][2] = get_demand(df, target_lat + ((c-2) * lat_diff),
                                    target_long + ((2-r) * long_diff),
                                    period - 96*7)
            grid[r][c][3] = get_demand(df, target_lat + ((c-2) * lat_diff),
                                    target_long + ((2-r) * long_diff),
                                    period - 96*14)
    return grid
