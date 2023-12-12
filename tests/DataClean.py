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

    start_date_dt = datetime(2023,1,1)  
    end_date_dt = datetime(2024,1,1)

    dummy = f_remove.CurrentDummyData(qty_weeks=1)
    dummy.drop(columns = ['IN'],inplace=True)

        
    time_stopper = [['time_init', time.perf_counter()]]
    
    
    output = f_remove.DataSynchronization(dummy, start_date_dt, end_date_dt, sample_freq=5, sample_time_base='m')     
    f_remove.CountMissingData(output, show=True)
    time_stopper.append(['DataSynchronization', time.perf_counter()])
    output.plot(title="DataSynchronization")
   
    output.iloc[12*24*3:12*24*10,2] = numpy.nan
    
    output.plot(title="Missing Phase")
    
    
    
    f_remove.PowerFactorDummyData(qty_weeks=1).plot()
    
    dummy_fp = pandas.read_csv("C:/Git/MVDataProcessing/CALADJ2074_FP.csv")    
    dummy_fp['timestamp'] = pandas.to_datetime(dummy_fp['timestamp'])
    dummy_fp.set_index('timestamp',inplace=True)
    dummy_fp = f_remove.DataSynchronization(dummy_fp, datetime(2022,6,13),datetime(2022,6,21), sample_freq=5, sample_time_base='m')
    dummy_fp = numpy.round(dummy_fp,2)
    dummy_fp.to_csv("dummy_fp.csv",index=False,header=False)
    
    

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
 

    
    output.iloc[12*24*15:12*24*26,:] = numpy.nan
    
    f_remove.CountMissingData(output, show=True)
    time_stopper.append(['Lost all phases', time.perf_counter()])
    output.plot(title="Lost all phases")

    # Get day max/min values
    max_vet,max_vet_idx = f_remove.GetDayMaxMin(output,start_date_dt,end_date_dt,sample_freq=5,threshold_accept=0.5,exe_param='max')         
    min_vet,min_vet_idx = f_remove.GetDayMaxMin(output,start_date_dt,end_date_dt,sample_freq=5,threshold_accept=0.5,exe_param='min') 
    weekday_curve = f_remove.GetWeekDayCurve(output, sample_freq=5, threshold_accept=0.8,min_sample_per_day=3,min_sample_per_workday=9)

    list_df_print = []
    for col in max_vet.columns:        
        df_print = pandas.DataFrame(index=max_vet_idx[col].values, dtype=object)
        df_print[col+'_max'] = max_vet[col].values        
        df_print = f_remove.DataSynchronization(df_print, start_date_dt, end_date_dt, sample_freq=5, sample_time_base='m')             
        list_df_print.append(df_print)
        
    
    for col in min_vet.columns:        
        df_print = pandas.DataFrame(index=min_vet_idx[col].values, dtype=object)
        df_print[col+'_min'] = min_vet[col].values        
        df_print = f_remove.DataSynchronization(df_print, start_date_dt, end_date_dt, sample_freq=5, sample_time_base='m')             
        list_df_print.append(df_print)
            
    df_print = pandas.DataFrame([])
    for col in list_df_print:  
        if(df_print.size==0):
            df_print = col                    
        else:
            df_print = pandas.concat((df_print,col),axis=1)
            
            
    
    ax = output.plot(title = 'Data')            
    df_print.plot.line(ax=ax,color='red',style='.')    
    
    matplotlib.pyplot.show()
    

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
    
