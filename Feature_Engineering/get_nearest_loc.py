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


# def get_demand(df, lat, long, period):
#     day, hour, minute = period_to_dayhourmin(period)
#     geohash = geohash2.encode(lat, long, 6)
#     demand_queried = df[(df['Period'] == period) &
#                         (df['geohash6'] == geohash)]['demand'].values
#     if len(demand_queried) > 0:
#         return demand_queried[0]
#     else:
#         return 0


# def get_neigh_grid(df, target_lat, target_long, period,
#                    lat_diff=0.0054931640625, long_diff=0.010986328125):
#     '''
#     Inputs:
#     target_lat, target_long, day, hour, minute all in integers
    
#     Outputs:
#     Function will return a 5x5x4 grid where the target loc will 
#     be in the middle of the grid with its closest neighbors surrounding.
#     Each grid 'cell' will contain the aggregated demand of the previous period
    
#     Latitude will be along the x-axis and Longitude along the y-axis
#     '''
#     # period = dayhourmin_to_period(day, hour, minute)
#     grid = np.zeros((5,5,4))
#     for r in range(grid.shape[0]):
#         for c in range(grid.shape[1]):
#             grid[r][c][0] = get_demand(df, target_lat + ((c-2) * lat_diff),
#                                     target_long + ((2-r) * long_diff),
#                                     period - 1)
#             grid[r][c][1] = get_demand(df, target_lat + ((c-2) * lat_diff),
#                                     target_long + ((2-r) * long_diff),
#                                     period - 96)
#             grid[r][c][2] = get_demand(df, target_lat + ((c-2) * lat_diff),
#                                     target_long + ((2-r) * long_diff),
#                                     period - 96*7)
#             grid[r][c][3] = get_demand(df, target_lat + ((c-2) * lat_diff),
#                                     target_long + ((2-r) * long_diff),
#                                     period - 96*14)
#     return grid


# df = pd.read_csv(os.path.join(Config.DATA_DIR, Config.CLEANED_TRG_FILE))
# import time



# start1=time.time()
# test6 = imp_get_neigh_grid(df, [-5.353088, -5.413513, -5.325623, -5.353088, -5.413513],
#                             [90.807495, 90.664673, 90.906372, 90.752563, 90.719604], [901,832,443,234,1125])
# end1=time.time()
# print(end1-start1)
# print("START NEW")
# start=time.time()
# test1 = get_neigh_grid(df, -5.353088, 90.807495, 901)
# test2 = get_neigh_grid(df, -5.413513, 90.664673, 832)
# test3 = get_neigh_grid(df, -5.325623, 90.906372, 443)
# test4 = get_neigh_grid(df, -5.353088, 90.752563, 234)
# test5 = get_neigh_grid(df, -5.413513, 90.719604, 1125)
# end=time.time()
# print(end-start)

# t = np.array([])
# for i in [test1,test2,test3,test4, test5]:
#     t = np.vstack((t,i)) if t.size else i

# t=np.reshape(t,(-1,5,5,4))
# print(t)

# print ('better')
# print(test6)

# print(np.array_equal(t,test6))