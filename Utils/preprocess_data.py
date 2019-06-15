"""
Clean and add features for dataset

Author: Ang Peng Seng
Date: May 2019
"""

import pandas as pd
import numpy as np
import geohash2
from multiprocessing import Pool, cpu_count
import sys
import os
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(ROOT_DIR)
from Config import Config


def clean_data(df):
    df['latitude'] = df['geohash6'].apply(lambda x: float(geohash2.decode_exactly(x)[0]))
    df['longitude'] = df['geohash6'].apply(lambda x: float(geohash2.decode_exactly(x)[1]))
    df['latitude_error'] = df['geohash6'].apply(lambda x: float(geohash2.decode_exactly(x)[2]))
    df['longitude_error'] = df['geohash6'].apply(lambda x: float(geohash2.decode_exactly(x)[3]))
    df['Hour'] = df['timestamp'].apply(lambda x: int(x.split(':')[0]))
    df['Minute'] = df['timestamp'].apply(lambda x: int(x.split(':')[1]))
    df['Period'] = ((df['day'] - 1) * 24 * 4) + (df['Hour'] * 4) + df['Minute'] // 15 
    return df


def parrellize_clean_data(df, func):
    num_cores = max(1, cpu_count() - 2)  # at least 1 core used, -2 cpu to prevent machine from freezing
    num_partitions = num_cores
    df_split = np.array_split(df, num_partitions)
    pool = Pool(num_cores)
    df = pd.concat(pool.map(func, df_split))
    pool.close()
    pool.join()
    return df


if __name__ == '__main__':
    df = pd.read_csv(os.path.join(Config.DATA_DIR, Config.ORIGINAL_TRG_FILE))
    cleaned_df = parrellize_clean_data(df, clean_data)
    cleaned_df = cleaned_df.drop(['timestamp'], axis=1)
    cleaned_df.to_csv(os.path.join(Config.DATA_DIR, Config.CLEANED_TRG_FILE),
                      index=False)
    print("Preprocessing Done")
