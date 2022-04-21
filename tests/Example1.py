# -*- coding: utf-8 -*-
"""
Created on Wed Apr 20 17:29:02 2022

@author: Bacalhau
"""

from itertools import combinations
import pandas as pd
import datetime as dt
import numpy as np
from datetime import datetime
from MVDataProcessing import PProcessing


if __name__ == "__main__":
    
    import time    
    
    print("Example:")    
    
    start_date = dt.datetime(2020,1,1)
    end_date = dt.datetime(2022,1,1)
    
    dummy = PProcessing.DummyData(start_date,end_date,remove_data=0.20,freq_min=7,delta_sec=39)
    
    time_init = time.perf_counter()        

    output = PProcessing.DataClean(dummy, start_date, end_date)
    
    print("Time spent: " + str(time.perf_counter()-time_init) )
    
    print(output)

    
    