"""
Created on Mon Apr 18 06:45:44 2022

@author: Bacalhau

"""


import pandas
import numpy
from datetime import datetime
import matplotlib.pyplot
import time
import FinishedFunctions as f_remove



if __name__ == "__main__":

    start_date_dt = datetime(2021,1,1)  
    end_date_dt = datetime(2023,1,1)

    dummy = f_remove.CurrentDummyData()
    #dummy.plot(title="Current Input (with outliers [A])")
    
    dummy.drop(columns = ['IN'],inplace=True)

    time_stopper = [['time_init', time.perf_counter()]]
    
    
    output = f_remove.DataSynchronization(dummy, start_date_dt, end_date_dt, sample_freq=5, sample_time_base='m')     
    f_remove.CountMissingData(output, show=True)
    time_stopper.append(['DataSynchronization', time.perf_counter()])
    output.plot(title="DataSynchronization")
   
    output.iloc[0:12*24*90,2] = numpy.nan
    
    output.plot(title="Missing Phase")
    
    

    output = f_remove.RemoveOutliersHardThreshold(output, hard_max=500, hard_min=0)
    f_remove.CountMissingData(output, show=True)    
    time_stopper.append(['RemoveOutliersHardThreshold', time.perf_counter()])
    output.plot(title="+RemoveOutliersHardThreshold")

   
    
    output = f_remove.RemoveOutliersMMADMM(output, len_mov_avg=25, std_def=3, plot=False,min_var_def=3)#, remove_from_process=['IN'])
    f_remove.CountMissingData(output, show=True)
    time_stopper.append(['RemoveOutliersMMADMM', time.perf_counter()])
    output.plot(title="+RemoveOutliersMMADMM")
    

    
    output = f_remove.RemoveOutliersQuantile(output)
    f_remove.CountMissingData(output, show=True)
    time_stopper.append(['RemoveOutliersQuantile', time.perf_counter()])
    output.plot(title="+RemoveOutliersQuantile")
  


    output = f_remove.RemoveOutliersHistoGram(output, min_number_of_samples_limit=12 * 5)
    f_remove.CountMissingData(output, show=True)
    time_stopper.append(['RemoveOutliersHistoGram', time.perf_counter()])
    output.plot(title="+RemoveOutliersHistoGram")
    


    output = f_remove.PhaseProportionInput(output, threshold_accept=0.60,plot =False,apply_filter=True)        
    output.plot(title="+PhaseProportionInput")
    f_remove.CountMissingData(output, show=True)
    time_stopper.append(['PhaseProportionInput', time.perf_counter()])
 

    
    output.iloc[12*24*90:12*24*180,:] = numpy.nan
    
    f_remove.CountMissingData(output, show=True)
    time_stopper.append(['Lost all phases', time.perf_counter()])
    output.plot(title="Lost all phases")

    # Get day max/min values
    max_vet,_ = f_remove.GetDayMaxMin(output,start_date_dt,end_date_dt,sample_freq=5,threshold_accept=0.5,exe_param='max')         
    min_vet,_ = f_remove.GetDayMaxMin(output,start_date_dt,end_date_dt,sample_freq=5,threshold_accept=0.5,exe_param='min') 
    weekday_curve = f_remove.GetWeekDayCurve(output, sample_freq=5, threshold_accept=0.8,min_sample_per_day=3,min_sample_per_workday=9)


    '''
    X_pred = f_remove.GetNSSCPredictedSamples(max_vet, min_vet, weekday_curve,start_date_dt,end_date_dt, sample_freq=5,sample_time_base='m')
    time_stopper.append(['X_pred', time.perf_counter()])
    X_pred.plot(title="X_pred")

    output = f_remove.ReplaceData(output,
                X_pred,
                start_date_dt,
                end_date_dt,   
                num_samples_day = 12*24,
                day_threshold=0.5,
                patamar_threshold = 0.5,
                num_samples_patamar = 12*6,                
                sample_freq= 5,
                sample_time_base = 'm')

    f_remove.CountMissingData(output, show=True)
    time_stopper.append(['Output (NSSC)', time.perf_counter()])
    output.plot(title="Output (NSSC)")

    matplotlib.pyplot.show()
    
    f_remove.TimeProfile(time_stopper, name='Main', show=True, estimate_for=1000 * 5)

    #f_remove.ShowExampleSimpleProcess()
    
    '''