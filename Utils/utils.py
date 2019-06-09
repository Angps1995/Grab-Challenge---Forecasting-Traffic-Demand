"""
Utility Functions

Author: Ang Peng Seng
Date: May 2019
"""

import numpy as np 


def dayhourmin_to_period(day, hour, minute):
    return ((day-1) * 24 * 4) + (hour * 4) + minute//15


def period_to_dayhourmin(period):
    day = period//96 + 1
    hour = (period - (day-1) * 96)//4
    minute = (period - ((day-1) * 96) - (hour*4)) * 15
    return (day, hour, minute)

if __name__ == '__main__':
    check = 0
    for i in range(100):
        day = np.random.randint(1, 62)
        hour = np.random.randint(0, 24)
        minute = np.random.choice([0, 15, 30, 45])
        period = dayhourmin_to_period(day, hour, minute)
        testday, testhour, testminute = period_to_dayhourmin(period)
        check += (testday == day) & (testhour == hour) & (testminute == minute)
    
    if check == 100:
        print('Passed')
    else:
        print('Failed')