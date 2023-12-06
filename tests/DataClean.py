"""
Created on Mon Apr 18 06:45:44 2022

@author: Bacalhau


Part2
0) Test Previous functions
2) Tailor the example maybe put data on a pickle file
3) Format to publish
4) Test on 2022 data

"""

import pandas as pd
import pandas
import datetime as dt
import numpy as np
import numpy
from datetime import datetime
import FinishedFunctions as f_remove
import matplotlib.pyplot as plt


#pd.set_option('display.max_rows', None)
#pd.set_option('display.max_columns', None)
#pd.set_option('display.width', None)
#pd.set_option('display.max_colwidth', None)


if __name__ == "__main__":

    '''

    import time
    import random
    import logging
    import logging.handlers

    logger = logging.getLogger(__name__)    
    logger.setLevel(logging.DEBUG)
    
    file_handler = logging.FileHandler('log.log')
    file_handler.setLevel(logging.DEBUG)
    
    cmd_handler = logging.StreamHandler()
    cmd_handler.setLevel(logging.INFO)
    
    file_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
    cmd_handler.setFormatter(logging.Formatter('%(message)s'))
    
    logger.handlers = []
    
    logger.addHandler(file_handler)
    logger.addHandler(cmd_handler)
    logger.info("Example:")
   
    
    data_inicio='2021-12-15'
    data_final='2023-01-15'
    
    start_date_dt = dt.datetime(int(data_inicio.split("-")[0]),int(data_inicio.split("-")[1]),int(data_inicio.split("-")[2]))
    end_date_dt = dt.datetime(int(data_final.split("-")[0]),int(data_final.split("-")[1]),int(data_final.split("-")[2]))
 
    
    #TESTE
    dummy = pd.read_csv('CALADJ2074_I.csv',names=['timestamp_aux','IA', 'IB', 'IV','IN'],skiprows=1,parse_dates=True)
    dummy.insert(loc=0, column='timestamp', value=pd.to_datetime(dummy.timestamp_aux.astype(str)))
    dummy = dummy.drop(columns=['timestamp_aux'])
    dummy.set_index('timestamp', inplace=True)
    
   
    time_stopper = []    
    time_stopper.append(['time_init',time.perf_counter()])
    output = f_remove.DataSynchronization(dummy,start_date_dt,end_date_dt,sample_freq= 5,sample_time_base='m')
    
    output.drop(columns=['IN'],inplace=True)

    fig, ax = plt.subplots()
    ax.plot(output.values)
    ax.set_title('Input')
    
    f_remove.CountMissingData(output,show=True)    
    time_stopper.append(['DataSynchronization',time.perf_counter()])    
    f_remove.output = f_remove.RemoveOutliersHardThreshold(output,hard_max=500,hard_min=0)        
    f_remove.CountMissingData(output,show=True)
    time_stopper.append(['RemoveOutliersHardThreshold',time.perf_counter()])    
    output = f_remove.RemoveOutliersMMADMM(output,len_mov_avg=2,std_def=5,plot=False)         
    f_remove.CountMissingData(output,show=True)
    time_stopper.append(['RemoveOutliersMMADMM',time.perf_counter()])
    output = f_remove.RemoveOutliersQuantile(output)    
    f_remove.CountMissingData(output,show=True)
    time_stopper.append(['RemoveOutliersQuantile',time.perf_counter()])
    output = f_remove.RemoveOutliersHistoGram(output,min_number_of_samples_limit=12*3)            
    f_remove.CountMissingData(output,show=True)
    time_stopper.append(['RemoveOutliersHistoGram',time.perf_counter()])
    
    #output.iloc[50000:60000,:] = np.nan
    #output.iloc[:10000, :] = np.nan

    fig, ax = plt.subplots()
    ax.plot(output.index.values,output.values)
    ax.set_title('No outliers')

    output = f_remove.PhaseProportionInput(output,threshold_accept = 0.60)
    f_remove.CountMissingData(output,show=True)
    time_stopper.append(['PhaseProportionInput',time.perf_counter()])

    fig, ax = plt.subplots()
    ax.plot(output.index.values,output.values)
    ax.set_title('PhaseProportionInput')


    #output = f_remove.NSSCInput(output,start_date_dt,end_date_dt)


    time_stopper.append(['NSSC', time.perf_counter()])
    
    fig, ax = plt.subplots()
    ax.plot(output.values)
    ax.set_title('Output')
    plt.show()


    #Simple Process
   
    output = f_remove.SimpleProcess(output,start_date_dt,end_date_dt,remove_from_process= ['IN'],sample_freq= 5,sample_time_base = 'm',pre_interpol = 12,pos_interpol = 12,prop_phases = True, integrate = False, interpol_integrate = 3)
  
    f_remove.CountMissingData(output,show=True)
    time_stopper.append(['PhaseProportionInput',time.perf_counter()])



    #TESTED - OK #output = RemoveOutliersMMADMM(dummy,df_avoid_periods = dummy_manobra)
    #TESTED - OK #output = CalcUnbalance(dummy)
    #TESTED - OK #output = RemoveOutliersQuantile(dummy,col_names = [],drop=False)
    #TESTED - OK #output = RemoveOutliersHistoGram(dummy,df_avoid_periods = dummy_manobra,min_number_of_samples_limit=12)
    #TESTED - OK #output = RemoveOutliersHardThreshold(dummy,hard_max=13.80,hard_min=0,df_avoid_periods = dummy_manobra)
    #TESTED - OK #output,index = SavePeriod(dummy,dummy_manobra)
    #TESTED - OK #dummy = DataClean(dummy,start_date_dt,end_date_dt,sample_freq= 5,sample_time_base='m')
    #TESTED - OK #output = ReturnOnlyValidDays(dummy,sample_freq = 5,threshold_accept = 1.0,sample_time_base = 'm')
    #TESTED - OK #output,_ = GetDayMaxMin(dummy,start_date_dt,end_date_dt,sample_freq = 5,threshold_accept = 1.0,exe_param='max')
    #TESTED - OK #output = MarkNanPeriod(dummy,dummy_manobra)
    #TESTED - OK #output = PhaseProportionInput(output,threshold_accept = 0.60,remove_from_process=['IN'])
    
    
    #output = GetWeekDayCurve(dummy,sample_freq = 5,threshold_accept = 1.0)    
    #output = SimpleProcess(dummy,start_date_dt,end_date_dt,sample_freq = 5,pre_interpol=1,pos_interpol=1,integrate=True,interpol_integrate=1)
    #output = SimpleProcess(dummy,start_date_dt,end_date_dt,sample_freq = 5)


    #------------------#
    #   CODE PROFILE   #
    #------------------#

    f_remove.TimeProfile(time_stopper,name='Main',show=True,estimate_for=1200)

    '''

    f_remove.ShowExample1()

    print("END")
    


