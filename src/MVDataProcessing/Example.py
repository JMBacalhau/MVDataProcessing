import matplotlib.pyplot
import pandas
import numpy
import datetime
import time
from datetime import datetime
import datetime as dt

from .Util import CurrentDummyData,DataSynchronization,CountMissingData,TimeProfile,GetWeekDayCurve,GetDayMaxMin
from .Clean import RemoveOutliersHardThreshold,RemoveOutliersQuantile,RemoveOutliersHistogram,RemoveOutliersMMADMM
from .Fill import ReplaceData,GetNSSCPredictedSamples,SimpleProcess,PhaseProportionInput

def ShowExampleSimpleProcess(plot: bool = True):
    
    """
    Demonstrates a simple data processing workflow using various functions to handle, analyze, and visualize data.
    
    The function executes a sequence of operations on dummy data, including data synchronization, outlier removal, 
    and data processing. It uses matplotlib to plot the results at each step. Additionally, it tracks the execution
    time for each step using the TimeProfile function and the number of missing data samples, outputting this 
    information to the console along with some explanation.
    
    
    Steps involved:
    - Close all existing matplotlib plots.
    - Generate dummy data and plot it.
    - Synchronize data with specified start and end dates.
    - Remove outliers using various methods (Hard Threshold, MMADMM, Quantile, Histogram).
    - Execute a simple data processing operation.
    - Plot the final output.
    - Display a time profile of the entire process.
    
        
    :param plot: Plot data for each step of the process. Disables the time profile.
    :type plot: bool,optional
    
    Returns:
        None: This function does not return any value.
    """
    
    #-------INPUT-------#
    if(plot):
        matplotlib.pyplot.close('all')
    
    
    start_date_dt = datetime(2023,1,1)
    end_date_dt = datetime(2025,1,1)
    
    print("In this example, the data loss and outliers are exaggerated for demonstration purposes.")
    print(f"The data will be processed between {start_date_dt} and {end_date_dt}")

    dummy = CurrentDummyData(qty_weeks=90)
    dummy.drop(columns = ['IN'],inplace=True)    
    
    if(plot):
        ax = dummy.plot(title = "1 - Three phase current")
        ax.set_ylabel("Current [A]")

    time_stopper = [['time_init', time.perf_counter()]]   
    
    #-------DATA SYNCHRONIZATION-------#
    
    output = DataSynchronization(dummy, start_date_dt, end_date_dt, sample_freq=5, sample_time_base='m')
    
    if(plot):
        ax = output.plot(title = "2 - Three phase current Sync")
        ax.set_ylabel("Current [A]")    

    print("Data Synchronization process done.")
    CountMissingData(output, show=True)
    time_stopper.append(['DataSynchronization', time.perf_counter()])
    
    #-------OUTLIER REMOVAL-------#
    output = RemoveOutliersHardThreshold(output, hard_max=500, hard_min=0)

    if(plot):
        ax = output.plot(title = "3 - Remove Outliers Hard Threshold")
        ax.set_ylabel("Current [A]")       
        
    print("Outliers removed using Hard Threshold.")
    CountMissingData(output, show=True)
    time_stopper.append(['RemoveOutliersHardThreshold', time.perf_counter()])
    
    
    output = RemoveOutliersMMADMM(output, len_mov_avg=3, std_def=4, plot=False)

    if(plot):
        ax = output.plot(title = "4 - Remove Outliers MMADMM")
        ax.set_ylabel("Current [A]")    

    print("Outliers removed using MMADMM.")
    CountMissingData(output, show=True)
    time_stopper.append(['RemoveOutliersMMADMM', time.perf_counter()])
    
    
    output = RemoveOutliersQuantile(output)

    if(plot):
        ax = output.plot(title = "5 - Remove Outliers Quantile")
        ax.set_ylabel("Current [A]")    

    print("Outliers removed using Quantile.")
    CountMissingData(output, show=True)
    time_stopper.append(['RemoveOutliersQuantile', time.perf_counter()])
    
    
    output = RemoveOutliersHistogram(output, min_number_of_samples_limit=12 * 5)
    
    if(plot):
        ax = output.plot(title = "6 - Remove Outliers Histogram")
        ax.set_ylabel("Current [A]")    

    print("Outliers removed using Histogram.")
    CountMissingData(output, show=True)
    time_stopper.append(['RemoveOutliersHistoGram', time.perf_counter()])

    #-------SIMPLE INPUTATION PROCESS-------#

    output = SimpleProcess(output, start_date_dt, end_date_dt,
                           sample_freq=5,
                           sample_time_base='m',
                           pre_interpol=1,
                           pos_interpol=6,
                           prop_phases=True,
                           integrate=True,
                           interpol_integrate=100)
    
    if(plot):
        ax = output.plot(title = "7 - Simple process data ")
        ax.set_ylabel("Current [A]") 
        
        
    print("Simple pre-made imputation process.\nORGANIZE->INTERPOLATE->PHASE_PROPORTION->INTERPOLATE->INTEGRATE->INTERPOLATE")
    CountMissingData(output, show=True)
    time_stopper.append(['SimpleProcessInput', time.perf_counter()])
    
    if(plot):
        matplotlib.pyplot.show()
    
    TimeProfile(time_stopper, name='Simple Process', show=~plot, estimate_for=1000 * 5)

    return



def ShowExampleNSSCProcess(plot: bool = True):
    """
    Demonstrates the normalized scaled standard weekday curve inputation method.

    This function goes through various steps of data processing including synchronization, outlier removal, data ng, and NSSC application. Each step is demonstrated with optional plotting for visual analysis. The data loss and outliers are exaggerated for demonstration purposes. The process is applied between a predefined start and end date, with multiple methods applied to handle missing data, outliers, and to predict and replace data in the final output.

    Parameters
    ----------
    plot : bool, optional
        If True, the function will plot the data at various stages of processing for visualization. 
        Defaults to True.

    Returns
    -------
    None
        The function does not return any value but optionally displays plots and prints information about the processing steps if `plot` is True.

    Notes
    -----
    - The function is meant for demonstration and educational purposes, showing various stages in data processing.
    - The process is specifically tailored for current data and may need adjustments for other types of data.
    - The example dates and parameters are hardcoded for demonstration and should be adapted for practical use.

    """
    
    if(plot):
        matplotlib.pyplot.close('all')
    
    #-----INPUT-----#
    start_date_dt = datetime(2023,1,1)  
    end_date_dt = datetime(2025,1,1)
    
    print("In this example, the data loss and outliers are exaggerated for demonstration purposes.")

    print(f"The data will be processed between {start_date_dt} and {end_date_dt}")

    dummy = CurrentDummyData(qty_weeks=90)
    dummy.drop(columns = ['IN'],inplace=True)
    
    #CREATE MISSING DATA ON ONE PHASE FOR A PERIOD AND THREE PHASE FOR ANOTHER
    dummy.iloc[12*24*7*40:12*24*7*50,2] = numpy.nan    
    dummy.iloc[12*24*7*60:12*24*7*64,:] = numpy.nan
    
    if(plot):
        ax = dummy.plot(title = "1 - Three phase current")
        ax.set_ylabel("Current [A]") 
    
    time_stopper = [['time_init', time.perf_counter()]]    
    
    #-----DATA SYNC-----#
    
    output = DataSynchronization(dummy, start_date_dt, end_date_dt, sample_freq=5, sample_time_base='m')     
        
    if(plot):
        ax = dummy.plot(title = "2 - Three phase Sync")
        ax.set_ylabel("Current [A]") 
    
    print("Data Synchronization process done.")
    CountMissingData(output, show=True)
    time_stopper.append(['DataSynchronization', time.perf_counter()])
    
        
    #-----OUTLIER REMOVAL-----#
    
    output = RemoveOutliersHardThreshold(output, hard_max=500, hard_min=0)
    
    if(plot):
        ax = output.plot(title = "3 - Remove Outliers Hard Threshold")
        ax.set_ylabel("Current [A]")   
    
    print("Outliers removed using Hard Threshold.")
    CountMissingData(output, show=True)    
    time_stopper.append(['RemoveOutliersHardThreshold', time.perf_counter()])
       
    
    output = RemoveOutliersMMADMM(output, len_mov_avg=25, std_def=3, plot=False,min_var_def=3)#, remove_from_process=['IN'])

    if(plot):
        ax = output.plot(title = "4 - Remove Outliers MMADMM")
        ax.set_ylabel("Current [A]")    

    print("Outliers removed using MMADMM.")
    CountMissingData(output, show=True)
    time_stopper.append(['RemoveOutliersMMADMM', time.perf_counter()])

    
    output = RemoveOutliersQuantile(output)
    
    if(plot):
        ax = output.plot(title = "5 - Remove Outliers Quantile")
        ax.set_ylabel("Current [A]")    
    
    print("Outliers removed using Quantile.")
    CountMissingData(output, show=True)
    time_stopper.append(['RemoveOutliersQuantile', time.perf_counter()])
    

    output = RemoveOutliersHistogram(output, min_number_of_samples_limit=12 * 5)
    
    if(plot):
        ax = output.plot(title = "6 - Remove Outliers Histogram")
        ax.set_ylabel("Current [A]")   
    
    print("Outliers removed using Histogram.")
    CountMissingData(output, show=True)
    time_stopper.append(['RemoveOutliersHistoGram', time.perf_counter()])
    

    #-----DATA ----#

    output = PhaseProportionInput(output, threshold_accept=0.60,plot =False,apply_filter=True)        
    
    if(plot):
        ax = output.plot(title = "7 - Phase proportion")
        ax.set_ylabel("Current [A]") 

    print("Processes input DataFrame to compute phase proportion based on various time frames and criteria.")    
    CountMissingData(output, show=True)
    time_stopper.append(['PhaseProportionInput', time.perf_counter()])
 
    
    #----NSSC----#
 
    print("In this example the NSSC is implemented in parts to be able to print and demonstrate each step. In production use the NSSCInput.")    
    
    print("Calculate the day max/min values")
    # Get day max/min values
    max_vet,max_vet_idx = GetDayMaxMin(output,start_date_dt,end_date_dt,sample_freq=5,threshold_accept=0.5,exe_param='max')         
    min_vet,min_vet_idx = GetDayMaxMin(output,start_date_dt,end_date_dt,sample_freq=5,threshold_accept=0.5,exe_param='min') 
    
    print("Get standard WeekDay Curve")
    weekday_curve = GetWeekDayCurve(output, sample_freq=5, threshold_accept=0.8,min_sample_per_day=3,min_sample_per_workday=9)

    #For plot purpose
    list_df_print = []
    for col in max_vet.columns:        
        df_print = pandas.DataFrame(index=max_vet_idx[col].values, dtype=object)
        df_print[col+'_max'] = max_vet[col].values        
        df_print = DataSynchronization(df_print, start_date_dt, end_date_dt, sample_freq=5, sample_time_base='m')             
        list_df_print.append(df_print)
        
    
    for col in min_vet.columns:        
        df_print = pandas.DataFrame(index=min_vet_idx[col].values, dtype=object)
        df_print[col+'_min'] = min_vet[col].values        
        df_print = DataSynchronization(df_print, start_date_dt, end_date_dt, sample_freq=5, sample_time_base='m')             
        list_df_print.append(df_print)
            
    df_print = pandas.DataFrame([])
    for col in list_df_print:  
        if(df_print.size==0):
            df_print = col                    
        else:
            df_print = pandas.concat((df_print,col),axis=1)
            
    
    if(plot):
        ax = output.plot(title = "8 - Min/Max value for each valid day")
        ax.set_ylabel("Current [A]") 
        df_print.plot.line(ax=ax,color='red',style='.')    
    
    
    print("Makes the prediction.")
    X_pred = GetNSSCPredictedSamples(max_vet, min_vet, weekday_curve,start_date_dt,end_date_dt, sample_freq=5,sample_time_base='m')
    
    if(plot):
        ax = X_pred.plot(title = "9 - The predicted time series based on the weekday curve and min/max values")
        ax.set_ylabel("Current [A]") 
    
    time_stopper.append(['X_pred', time.perf_counter()])
    

    output = ReplaceData(output,
                X_pred,
                start_date_dt,
                end_date_dt,   
                num_samples_day = 12*24,
                day_threshold=0.5,
                patamar_threshold = 0.5,
                num_samples_patamar = 12*6,                
                sample_freq= 5,
                sample_time_base = 'm')

    if(plot):
        ax = output.plot(title = "10 - Output (NSSC)")
        ax.set_ylabel("Current [A]") 
    

    print("NSSC Method applied.")    
    CountMissingData(output, show=True)
    time_stopper.append(['Output (NSSC)', time.perf_counter()])    

    if(plot):
        matplotlib.pyplot.show()
    else:
        TimeProfile(time_stopper, name='NSSC', show=~plot, estimate_for=1000 * 5)
    
    return