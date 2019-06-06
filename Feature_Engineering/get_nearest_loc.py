import numpy as np
import geohash2


def dayhourmin_to_period(day, hour, minute):
    return ((day-1) * 24 * 4) + (hour * 4) + minute//15


def period_to_dayhourmin(period):
    day = period//96 + 1
    hour = (period - (day-1) * 96)//4
    minute = (period - ((day-1) * 96) - (hour*4)) * 15
    return (day, hour, minute)
   

def get_demand(df, lat, long, period):
    day, hour, minute = period_to_dayhourmin(period)
    geohash = geohash2.encode(lat, long, 6)
    demand_queried = df[(df['day'] == day) &
                        (df['Hour'] == hour) &
                        (df['Minute'] == minute) &
                        (df['geohash6'] == geohash)]['demand'].values
    if len(demand_queried) > 0:
        return demand_queried[0]
    else:
        return 0


def get_neigh_grid(df, target_lat, target_long, day, hour, minute,
                   lat_diff=0.0054931640625, long_diff=0.010986328125):
    '''
    Inputs:
    target_lat, target_long, day, hour, minute all in integers
    
    Outputs:
    Function will return a 5x5x1 grid where the target loc will
    be in the middle of the grid with its closest neighbors surrounding.
    Each grid 'cell' will contain the aggregated demand of the previous period
    
    Latitude will be along the x-axis and Longitude along the y-axis
    '''
    period = dayhourmin_to_period(day, hour, minute)
    grid = np.zeros((5, 5))
    for r in range(grid.shape[0]):
        for c in range(grid.shape[1]):
            grid[r][c] = [get_demand(df, target_lat + ((c-2) * lat_diff),
                                     target_long + ((2-r) * long_diff),
                                     period-1)]
            
    return grid
    