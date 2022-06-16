"""
Created on Mon Apr 18 06:45:44 2022

@author: Bacalhau

Part1


Part2
1) Input


Part3

1) Reports
2) Graphs
3) ToPDF


Part4

1) Publish

Part5

1) Test on 2021 dataset


"""

import pandas as pd
import pandas
import datetime as dt
import numpy as np
import numpy
from datetime import datetime
import FinishedFunctions as f_remove


#pd.set_option('display.max_rows', None)
#pd.set_option('display.max_columns', None)
#pd.set_option('display.width', None)
#pd.set_option('display.max_colwidth', None)


def ReturnOnlyValidDays(x_in: pd.DataFrame,
                        sample_freq: int = 5,
                        threshold_accept: float = 1.0,
                        sample_time_base: str = 'm',
                        remove_from_process = []) -> pd.DataFrame:
    """
    Returns all valid days. A valid day is one with no missing values for any 
    of the timeseries on each column.
    
    
    :param x_in: A pandas.core.frame.DataFrame where the index is of type "pandas.core.indexes.datetimes.DatetimeIndex" and each column contain an electrical
    quantity time series.    
    :type x_in: pandas.core.frame.DataFrame
    
    :param sample_freq: The sample frequency of the time series. Defaults to 5.  
    :type sample_freq: int,optional
    
    :param threshold_accept: The amount of samples that is required to consider a valid day. Defaults to 1 (100%).  
    :type threshold_accept: float,optional
    
    :param sample_time_base: The base time of the sample frequency. Specify if the sample frequency is in (h)ours, (m)inutes, or (s)econds. Defaults to (m)inutes.  
    :type sample_time_base: srt,optional
    
    :param remove_from_process: Columns to be kept off the process;  
    :type remove_from_process: list,optional
    
         
    :raises Exception: if x_in has no DatetimeIndex. 
    :raises Exception: if sample_time_base is is not in seconds, minutes or hours.
    
    
    :return: Y: The pandas.core.frame.DataFrame with samples filled based on the proportion between time series.
    :rtype: Y: pandas.core.frame.DataFrame

    """
    
    #-------------------#
    # BASIC INPUT CHECK #
    #-------------------#
    
    if not(isinstance(x_in.index, pd.DatetimeIndex)):  raise Exception("DataFrame has no DatetimeIndex.")    
    if sample_time_base not in ['s','m','h']:  raise Exception("The sample_time_base is not in seconds, minutes or hours.")
    
    #-------------------#
    
    X = x_in.copy(deep=True)
    
    if(len(remove_from_process)>0):         
        X = X.drop(remove_from_process,axis=1)
    
    
    qty_sample_dic = {'s':24*60*60,'m':24*60,'h':24}
    

    df_count = X.groupby([X.index.year,X.index.month,X.index.day]).count()/(qty_sample_dic[sample_time_base]/sample_freq)    
    time_vet_stamp = X.index[np.arange(0,len(X.index),int(24*60/sample_freq))]     
    df_count = df_count.reset_index(drop=True)    
    df_count.insert(0,'timestamp_day', time_vet_stamp)
    df_count.set_index('timestamp_day', inplace=True)    
    df_count = df_count>=threshold_accept    
    
    df_count = df_count.sum(axis=1) == df_count.shape[1]
    df_count.name = 'isValid'
    df_count = df_count.reset_index()
    X['timestamp_day'] = X.index.floor("D").values
    
    keep_X_index = X.index
    X = pd.merge(X, df_count,on = 'timestamp_day' ,how ='left')
    X.index = keep_X_index
    X = X.loc[X['isValid'] == True,:]
    
    X.drop(columns=['isValid','timestamp_day'],inplace=True)

        
    return X,df_count

def GetDayMaxMin(x_in,start_date_dt,end_date_dt,sample_freq = 5,threshold_accept = 1.0,exe_param='max'):
    """

    :param x_in: param start_date_dt:
    :param end_date_dt: param sample_freq:  (Default value = 5)
    :param threshold_accept: Default value = 1.0)
    :param exe_param: Default value = 'max')
    :param start_date_dt: 
    :param sample_freq:  (Default value = 5)

    """
    
    X = x_in.copy(deep=True)
    
    X,_ = ReturnOnlyValidDays(X,sample_freq,threshold_accept)
        
    if(exe_param=='max'):
        Y = X.groupby([X.index.year,X.index.month,X.index.day]).max()
    else:
        Y = X.groupby([X.index.year,X.index.month,X.index.day]).min()
        
    time_vet_stamp = X.index[np.arange(0,len(X.index),int(24*60/sample_freq))]     
    Y = Y.reset_index(drop=True)    
    Y.insert(0,'timestamp_day', time_vet_stamp)
    Y.set_index('timestamp_day', inplace=True)    
    
    Y = f_remove.DataSynchronization(Y, start_date_dt, end_date_dt,sample_freq = 1,sample_time_base='D')
    
    Y = Y.interpolate(method_type='linear')    
    
    return Y

def GetWeekDayCurve(x_in,sample_freq = 5,threshold_accept = 1.0,min_sample_per_day = 3,min_sample_per_workday = 9):
    """

    :param x_in: param sample_freq:  (Default value = 5)
    :param threshold_accept: Default value = 1.0)
    :param min_sample_per_day: Default value = 3)
    :param min_sample_per_workday: Default value = 9)
    :param sample_freq:  (Default value = 5)

    """
    
    #x_in = dummy.copy(deep=True)
    
    X = x_in.copy(deep=True)
    
    Y,df_count = ReturnOnlyValidDays(X,sample_freq,threshold_accept)
    
    #Get valid data statistics
    df_count = df_count.loc[df_count['isValid'] == True,:]
    df_stats = df_count.groupby(df_count['timestamp_day'].dt.weekday).count()
    days_unique = df_stats.shape[0]
    count_days_unique = df_stats['timestamp_day'].values
        
    
    #Has enough data do use ?
    if((days_unique==7) and (np.min(count_days_unique)>=min_sample_per_day)):
        print('Can calculate a curve for every weekday')
        
        Y = Y.groupby([Y.index.weekday,Y.index.hour,Y.index.minute]).mean()
        Y.index.names = ['WeekDay','Hour','Min']    
        Y = Y.reset_index()
        
        #Normalization max min each day
        grouper = Y.groupby([Y.WeekDay])    
        maxes = grouper.transform('max')
        mins = grouper.transform('min')
        
        Y.iloc[:,3:] = (Y.iloc[:,3:]-mins.iloc[:,2:])/(maxes.iloc[:,2:]-mins.iloc[:,2:])  
        
    else:
        work_days = df_stats.loc[df_stats.index<=4,'timestamp_day'].sum()
        sat_qty = df_stats.loc[df_stats.index==5,'timestamp_day'].sum()
        sun_qty = df_stats.loc[df_stats.index==6,'timestamp_day'].sum()
        
        if((work_days>=min_sample_per_workday) and sun_qty>=min_sample_per_day and sat_qty>=min_sample_per_day):
            print('Can calculate a curve for every weekday and use Sat. and Sun.')
            
            Y = Y.groupby([Y.index.weekday,Y.index.hour,Y.index.minute]).mean()
            Y.index.names = ['WeekDay','Hour','Min']    
            Y = Y.reset_index()
            
            #Normalization max min each day
            grouper = Y.groupby([Y.WeekDay])    
            maxes = grouper.transform('max')
            mins = grouper.transform('min')
            
            Y.iloc[:,3:] = (Y.iloc[:,3:]-mins.iloc[:,2:])/(maxes.iloc[:,2:]-mins.iloc[:,2:])  
            
            
            #FALTA PEGAR UM DIA DA SEMANA MAIS PROXIMO PARA COMPLETAR OS INEXISTENTES
            
        else:
            print('Use default curve.')
            
            #FALTA ESCREVER UMA DEFAULT E PERMITIR IMPORTAR
        
    return Y
   
    



if __name__ == "__main__":
    
    import time
    import random
    import logging
    import logging.handlers
    
    
    #-------------------------------------#
    #       INITIAL CONFIGURATION         #
    #-------------------------------------#
    
    
    
    #------------LOGGER------------#
    
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
  
    
    
    data_inicio='2021-01-01'
    data_final='2022-03-01'
    
    start_date_dt = dt.datetime(int(data_inicio.split("-")[0]),int(data_inicio.split("-")[1]),int(data_inicio.split("-")[2]))
    end_date_dt = dt.datetime(int(data_final.split("-")[0]),int(data_final.split("-")[1]),int(data_final.split("-")[2]))
 
        
    dummy = np.arange(start_date_dt, end_date_dt,np.timedelta64(5,'m'), dtype='datetime64')
    dummy = dummy + np.timedelta64(random.randint(0, 59),'s') # ADD a second to the end so during the sort this samples will be at last (HH:MM:01)   
        
    dummy = pd.DataFrame(dummy,columns=['timestamp'])
    
    dummy['IA'] = 100
    dummy['IB'] = 200
    dummy['IV'] = 300
    dummy['IN'] = 50
    
    #dummy['VA'] = 1
    #dummy['VB'] = 2
    #dummy['VV'] = 3
    #dummy['VN'] = 4
    
    
    dummy.set_index('timestamp', inplace=True)
    
    
    for col in ['VA','VB','VV']:
        dummy.loc[dummy.sample(frac=0.001).index, col] = np.nan
    
    
    time_init = time.perf_counter()    
    
    
    #TESTE
    dummy = pd.read_csv('CALADJ2074_I.csv',names=['timestamp_aux','IA', 'IB', 'IV','IN'],skiprows=1,parse_dates=True)
    dummy.insert(loc=0, column='timestamp', value=pd.to_datetime(dummy.timestamp_aux.astype(str)))
    dummy = dummy.drop(columns=['timestamp_aux'])
    dummy.set_index('timestamp', inplace=True)
    
    '''
    #TESTE MANOBRAS
    dummy_manobra = pd.read_csv('BancoManobras.csv',names=['EQ','ALIM1', 'ALIM2', 'data_inicio',"data_final"],skiprows=1,parse_dates=True)
    dummy_manobra = dummy_manobra.iloc[:,-2:]
    
    dummy_manobra = pd.DataFrame([[dt.datetime(2021,1,1),dt.datetime(2021,2,1)]])
    '''
   
    time_stopper = []    
    time_stopper.append(['time_init',time.perf_counter()])
    output = f_remove.DataSynchronization(dummy,start_date_dt,end_date_dt,sample_freq= 5,sample_time_base='m')
    #output.plot()
    
    
    f_remove.CountMissingData(output,show=True)    
    time_stopper.append(['DataSynchronization',time.perf_counter()])    
    f_remove.output = f_remove.RemoveOutliersHardThreshold(output,hard_max=500,hard_min=0)        
    f_remove.CountMissingData(output,show=True)
    time_stopper.append(['RemoveOutliersHardThreshold',time.perf_counter()])    
    output = f_remove.RemoveOutliersMMADMM(output,len_mov_avg=3,std_def=4,plot=False,remove_from_process=['IN'])         
    f_remove.CountMissingData(output,show=True)
    time_stopper.append(['RemoveOutliersMMADMM',time.perf_counter()])
    output = f_remove.RemoveOutliersQuantile(output)    
    f_remove.CountMissingData(output,show=True)
    time_stopper.append(['RemoveOutliersQuantile',time.perf_counter()])
    output = f_remove.RemoveOutliersHistoGram(output,min_number_of_samples_limit=12*5)        
    f_remove.CountMissingData(output,show=True)
    time_stopper.append(['RemoveOutliersHistoGram',time.perf_counter()])
    
    _ = GetDayMaxMin(output,start_date_dt,end_date_dt,sample_freq = 5,threshold_accept = 0.2,exe_param='max')
    time_stopper.append(['GetDayMaxMin',time.perf_counter()])
    _ = GetDayMaxMin(output,start_date_dt,end_date_dt,sample_freq = 5,threshold_accept = 0.2,exe_param='min')    
    time_stopper.append(['GetDayMaxMin',time.perf_counter()])
    
    #output.plot()
    
      
    output2 = f_remove.SimpleProcess(output,start_date_dt,end_date_dt,remove_from_process= ['IN'],sample_freq= 5,sample_time_base = 'm',pre_interpol = 12,pos_interpol = 12,prop_phases = True, integrate = False, interpol_integrate = 3)
   
  
    
    output = f_remove.PhaseProportonInput(output,threshold_accept = 0.60,remove_from_process=['IN'])
    f_remove.CountMissingData(output,show=True)
    time_stopper.append(['PhaseProportonInput',time.perf_counter()])
    
    
    
    output.plot()
    output2.plot()
    
    #TESTED - OK #output = RemoveOutliersMMADMM(dummy,df_avoid_periods = dummy_manobra)    
    #TESTED - OK #output = CalcUnbalance(dummy)
    #TESTED - OK #output = RemoveOutliersQuantile(dummy,col_names = [],drop=False)
    #TESTED - OK #output = RemoveOutliersHistoGram(dummy,df_avoid_periods = dummy_manobra,min_number_of_samples_limit=12)    
    #TESTED - OK #output = RemoveOutliersHardThreshold(dummy,hard_max=13.80,hard_min=0,df_avoid_periods = dummy_manobra)    
    #TESTED - OK #output,index = SavePeriod(dummy,dummy_manobra)    
    #TESTED - OK #dummy = DataClean(dummy,start_date_dt,end_date_dt,sample_freq= 5,sample_time_base='m')
    #TESTED - OK #output = ReturnOnlyValidDays(dummy,sample_freq = 5,threshold_accept = 1.0,sample_time_base = 'm')
    #TESTED - OK #output = GetDayMaxMin(dummy,start_date_dt,end_date_dt,sample_freq = 5,threshold_accept = 1.0,exe_param='max')
    #TESTED - OK #output = MarkNanPeriod(dummy,dummy_manobra)
    #TESTED - OK #output = PhaseProportonInput(output,threshold_accept = 0.60,remove_from_process=['IN'])
    
    
    
    #output = GetWeekDayCurve(dummy,sample_freq = 5,threshold_accept = 1.0)
    
    #output = SimpleProcess(dummy,start_date_dt,end_date_dt,sample_freq = 5,pre_interpol=1,pos_interpol=1,integrate=True,interpol_integrate=1)
    #output = SimpleProcess(dummy,start_date_dt,end_date_dt,sample_freq = 5)
    
    
    
    #------------------#
    #   CODE PROFILE   #
    #------------------#
    
    f_remove.TimeProfile(time_stopper,name='Main',show=True,estimate_for=1000*5)
