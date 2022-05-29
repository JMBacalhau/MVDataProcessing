"""
Created on Mon Apr 18 06:45:44 2022

@author: Bacalhau

Part1

7) Comment, clean and finish some functions


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
from itertools import permutations
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
   
    
   

#TODO FROM HERE





#TODO     
def SimpleProcess(X,start_date_dt,end_date_dt,sample_freq = 5,pre_interpol=False,pos_interpol=False,prop_phases=False,integrate=False,interpol_integrate=False):    
    """

    :param X: param start_date_dt:
    :param end_date_dt: param sample_freq:  (Default value = 5)
    :param pre_interpol: Default value = False)
    :param pos_interpol: Default value = False)
    :param prop_phases: Default value = False)
    :param integrate: Default value = False)
    :param interpol_integrate: Default value = False)
    :param start_date_dt: 
    :param sample_freq:  (Default value = 5)

    """
        
    #ORGANIZE->INTERPOLATE->PHASE_PROPORTION->INTERPOLATE->INTEGRATE->INTERPOLATE
    
    #Organize samples
    Y = f_remove.DataSynchronization(X,start_date_dt,end_date_dt,sample_freq,sample_time_base='m')
    
    #Interpolate before proportion between phases
    if(pre_interpol!=False):
        Y = Y.interpolate(method_type='linear',limit=pre_interpol)
    
    #Interpolate after proportion between phases
    if(pos_interpol!=False):
        Y = Y.interpolate(method_type='linear',limit=pos_interpol)        
             
    #Integralization 1h
    if(integrate!=False):        
        Y = f_remove.IntegrateHour(Y,sample_freq = 5)        
        
        #Interpolate after Integralization 1h
        if(interpol_integrate!=False):
            Y = Y.interpolate(method_type='linear',limit=interpol_integrate)                              
        
    
    return Y

def RemovePeriod(x_in: pd.DataFrame,df_remove: pd.DataFrame,remove_from_process: list = []) -> pd.DataFrame:
    """
    Marks as nan all specified timestamps

    """
    
    Y = x_in.copy(deep=True)    
     
    for index,row in df_remove.iterrows():
        Y.loc[np.logical_and(Y.index>=row[0],Y.index<=row[1]),Y.columns.difference(remove_from_process)] = np.nan        
        
    return Y

def SavePeriod(x_in: pd.DataFrame,df_save: pd.DataFrame) -> tuple:    
    """
    For a given set of periods (Start->End) returns the 

    Tá bugada!!!!!!!!!


    """
    
    Y = x_in.copy(deep=True)
    mark_index_not = x_in.index    
    
    for index,row in df_save.iterrows():
        Y = Y.loc[np.logical_and(Y.index>=row[0],Y.index<=row[1]),:]
        mark_index_not = mark_index_not[np.logical_and(mark_index_not>=row[0],mark_index_not<=row[1])]    
    
    return Y,mark_index_not

def RemoveOutliersHardThreshold(x_in: pd.DataFrame,
                                hard_max: bool = False,
                                hard_min: bool = False,
                                df_avoid_periods = pd.DataFrame([])) -> pd.DataFrame:
    """
    Removes outliers from the timeseries on each column using threshold.

    """
        
    Y = x_in.copy(deep=True)    
    
    Y[Y>=hard_max] = np.nan
    Y[Y<=hard_min] = np.nan
    
    if(df_avoid_periods.shape[0]!=0):
        df_values,index_return = SavePeriod(x_in,df_avoid_periods)        
        Y.loc[index_return,:] = df_values

    return Y

#TODO DOUBLE SIDE , LOWER, HIGHER, AVOID PHASE
def RemoveOutliersHistoGram(x_in: pd.DataFrame,
                            df_avoid_periods: pd.DataFrame = pd.DataFrame([]),
                            integrate_hour: bool = True,
                            sample_freq: int = 5,
                            min_number_of_samples_limit: int  =12) -> pd.DataFrame:
    """
    Removes outliers from the timeseries on each column using the histogram.
    The parameter 'min_number_of_samples_limit' specify the minimum amount of hours in integrate flag is True/samples
    that a value must have to be considered not an outlier.    


    """
    
    Y = x_in.copy(deep=True)
    
    #Remove outliers ouside the avoid period 
    if(integrate_hour):
        Y_int = f_remove.IntegrateHour(Y,sample_freq)    
        Y_int = Y_int.reset_index(drop=True)    
    
    for col in Y_int:
        Y_int[col] = Y_int[col].sort_values(ascending=False,ignore_index=True)
    
    if(Y_int.shape[0]<min_number_of_samples_limit):
        min_number_of_samples_limit = Y_int.shape[0]
    
    threshold_max =  Y_int.iloc[min_number_of_samples_limit+1,:]
    threshold_min =  Y_int.iloc[-min_number_of_samples_limit-1,:]
        
    for col in Y:
        Y.loc[np.logical_or(Y[col]>threshold_max[col],Y[col]<threshold_min[col]),col] = np.nan
            
    if(df_avoid_periods.shape[0]!=0):
        df_values,index_return = SavePeriod(x_in,df_avoid_periods)        
        Y.loc[index_return,:] = df_values
     
    return Y

def RemoveOutliersQuantile(x_in:  pd.DataFrame,
                           col_names: list = [],
                           drop: bool = False) -> pd.DataFrame:
    """
     Removes outliers from the timeseries on each column using the top and bottom
     quantile metric as an outlier marker.
     
    """
    
    Y = x_in.copy(deep=True)
    
    #If not specified runs on every coluns
    if(len(col_names)==0):
        col_names = x_in.columns
            
    for col_name in col_names:
        q1 = x_in[col_name].quantile(0.25)
        q3 = x_in[col_name].quantile(0.75)
        iqr = q3-q1 #Interquartile range
        fence_low  = q1-1.5*iqr
        fence_high = q3+1.5*iqr
        Y.loc[(Y[col_name] < fence_low) | (Y[col_name] > fence_high),col_name] = np.nan
        
    if(drop):
        Y.dropna(inplace=True)
    
    return Y

def RemoveOutliersMMADMM(x_in: pd.DataFrame,
                         df_avoid_periods: pd.DataFrame = pd.DataFrame([]),
                         len_mov_avg: int = 4*12,
                         std_def: int = 2,
                         min_var_def: float = 0.5,
                         allow_negatives: bool = False,
                         plot: bool =False) -> pd.DataFrame:
    """
    Removes outliers from the timeseries on each column using the (M)oving (M)edian (A)bslute 
    (D)eviation around the (M)oving (M)edian. 
     

    """
        
    Y = x_in.copy(deep=True)  
          
    # ------------------------ OUTLIERS ------------------------            

    X_mark_outlier = x_in.copy(deep=True)
    X_mark_outlier.loc[:,:] = False    
    
    #---------PROCESSAMENTO OUTLIERS POR MÉDIA MÓVEL   
    X_mad = x_in.copy(deep=True)
    X_moving_median = x_in.copy(deep=True)
    X_moving_up = x_in.copy(deep=True)
    X_moving_down = x_in.copy(deep=True)
      
    # DESVIO PADRÂO ABSOLUTO ENTORNO DA MEDIANA MOVEL
                       
    #------------ Computa Mediana Móvel ------------#                                      
    X_moving_median = X_moving_median.rolling(len_mov_avg).median().shift(-int(len_mov_avg/2))
           
    X_moving_median.iloc[-2*len_mov_avg:,:] = X_moving_median.iloc[-2*len_mov_avg:,:].fillna(method='ffill')
    X_moving_median.iloc[:2*len_mov_avg,:] = X_moving_median.iloc[:2*len_mov_avg,:].fillna(method='bfill')
    
    #------------ Computa MAD Móvel ------------#       
    X_mad = x_in-X_moving_median       
    X_mad = X_mad.rolling(len_mov_avg).median().shift(-int(len_mov_avg/2))       
    X_mad.iloc[-2*len_mov_avg:,:] = X_mad.iloc[-2*len_mov_avg:,:].fillna(method='ffill')
    X_mad.iloc[:2*len_mov_avg,:] = X_mad.iloc[:2*len_mov_avg,:].fillna(method='bfill')
           
    #------------ Coloca no mínimo 0.5kV de faixa de segurança para dados com baixa variância ------------#       
    X_mad[X_mad<=min_var_def]= min_var_def
    
    #------------ MAD Móvel Limites ------------#       
    X_moving_up = X_moving_median+std_def*X_mad
    X_moving_down = X_moving_median-std_def*X_mad
    
    #------------ Allow the lower limit to go negative. Only valid for kVar or bi-directional current/Power. ------------#
    if(~allow_negatives):
        X_moving_down[X_moving_down<=0] = 0
               
    #------------ Marcando outliers ------------#
    X_mark = (x_in>=X_moving_up) | (x_in<=X_moving_down)
    
    #------------ Não marca os intervalos onde não foi possível determinar ------------#   
    X_mark[ X_moving_up.isnull() | X_moving_down.isnull() ] = False              
    X_mark.iloc[:int(len_mov_avg/2),:] = False
    X_mark.iloc[-int(len_mov_avg/2),:] = False
    
    Y[X_mark] = np.nan
    
    #------------ Não marca os intervalos selecionados ------------#   
    if(df_avoid_periods.shape[0]!=0):
        df_values,index_return = SavePeriod(x_in,df_avoid_periods)        
        Y.loc[index_return,:] = df_values
    
    if(plot):
        ax = X_moving_median.plot()
        x_in.plot(ax=ax)
        X_mad.plot(ax=ax)
        X_moving_down.plot(ax=ax)
        X_moving_up.plot(ax=ax)
        
        
           
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
    f_remove.output = RemoveOutliersHardThreshold(output,hard_max=500,hard_min=0)        
    f_remove.CountMissingData(output,show=True)
    time_stopper.append(['RemoveOutliersHardThreshold',time.perf_counter()])
    output = RemoveOutliersMMADMM(output,len_mov_avg=3,std_def=4)   
    f_remove.CountMissingData(output,show=True)
    time_stopper.append(['RemoveOutliersMMADMM',time.perf_counter()])
    output = RemoveOutliersQuantile(output,drop=False)    
    f_remove.CountMissingData(output,show=True)
    time_stopper.append(['RemoveOutliersQuantile',time.perf_counter()])
    output = RemoveOutliersHistoGram(output,min_number_of_samples_limit=12*5)        
    f_remove.CountMissingData(output,show=True)
    time_stopper.append(['RemoveOutliersHistoGram',time.perf_counter()])
    
    _ = GetDayMaxMin(output,start_date_dt,end_date_dt,sample_freq = 5,threshold_accept = 0.2,exe_param='max')
    time_stopper.append(['GetDayMaxMin',time.perf_counter()])
    _ = GetDayMaxMin(output,start_date_dt,end_date_dt,sample_freq = 5,threshold_accept = 0.2,exe_param='min')    
    time_stopper.append(['GetDayMaxMin',time.perf_counter()])
    
    #output.plot()
    
    output = f_remove.PhaseProportonInput(output,threshold_accept = 0.60,remove_from_process=['IN'])
    f_remove.CountMissingData(output,show=True)
    time_stopper.append(['PhaseProportonInput',time.perf_counter()])
    
    
    
   
    
    
    #output.plot()
    
    #TESTED - OK #output = RemoveOutliersMMADMM(dummy,df_avoid_periods = dummy_manobra)    
    #TESTED - OK #output = CalcUnbalance(dummy)
    #TESTED - OK #output = RemoveOutliersQuantile(dummy,col_names = [],drop=False)
    #TESTED - OK #output = RemoveOutliersHistoGram(dummy,df_avoid_periods = dummy_manobra,min_number_of_samples_limit=12)    
    #TESTED - OK #output = RemoveOutliersHardThreshold(dummy,hard_max=13.80,hard_min=0,df_avoid_periods = dummy_manobra)    
    #TESTED - OK #output,index = SavePeriod(dummy,dummy_manobra)    
    #TESTED - OK #dummy = DataClean(dummy,start_date_dt,end_date_dt,sample_freq= 5,sample_time_base='m')
    #TESTED - OK #output = ReturnOnlyValidDays(dummy,sample_freq = 5,threshold_accept = 1.0,sample_time_base = 'm')
    #TESTED - OK #output = GetDayMaxMin(dummy,start_date_dt,end_date_dt,sample_freq = 5,threshold_accept = 1.0,exe_param='max')
    #TESTED - OK #output = RemovePeriod(dummy,dummy_manobra)
    #TESTED - OK #output = PhaseProportonInput(output,threshold_accept = 0.60,remove_from_process=['IN'])
    
    
    
    #output = GetWeekDayCurve(dummy,sample_freq = 5,threshold_accept = 1.0)
    
    #output = SimpleProcess(dummy,start_date_dt,end_date_dt,sample_freq = 5,pre_interpol=1,pos_interpol=1,integrate=True,interpol_integrate=1)
    #output = SimpleProcess(dummy,start_date_dt,end_date_dt,sample_freq = 5)
    
    
    
    #------------------#
    #   CODE PROFILE   #
    #------------------#
    
    f_remove.TimeProfile(time_stopper,name='Main',show=True,estimate_for=1000*5)


'''

BaseException
 +-- SystemExit
 +-- KeyboardInterrupt
 +-- GeneratorExit
 +-- Exception
      +-- StopIteration
      +-- StopAsyncIteration
      +-- ArithmeticError
      |    +-- FloatingPointError
      |    +-- OverflowError
      |    +-- ZeroDivisionError
      +-- AssertionError
      +-- AttributeError
      +-- BufferError
      +-- EOFError
      +-- ImportError
      |    +-- ModuleNotFoundError
      +-- LookupError
      |    +-- IndexError
      |    +-- KeyError
      +-- MemoryError
      +-- NameError
      |    +-- UnboundLocalError
      +-- OSError
      |    +-- BlockingIOError
      |    +-- ChildProcessError
      |    +-- ConnectionError
      |    |    +-- BrokenPipeError
      |    |    +-- ConnectionAbortedError
      |    |    +-- ConnectionRefusedError
      |    |    +-- ConnectionResetError
      |    +-- FileExistsError
      |    +-- FileNotFoundError
      |    +-- InterruptedError
      |    +-- IsADirectoryError
      |    +-- NotADirectoryError
      |    +-- PermissionError
      |    +-- ProcessLookupError
      |    +-- TimeoutError
      +-- ReferenceError
      +-- RuntimeError
      |    +-- NotImplementedError
      |    +-- RecursionError
      +-- SyntaxError
      |    +-- IndentationError
      |         +-- TabError
      +-- SystemError
      +-- TypeError
      +-- ValueError
      |    +-- UnicodeError
      |         +-- UnicodeDecodeError
      |         +-- UnicodeEncodeError
      |         +-- UnicodeTranslateError
      +-- Warning
           +-- DeprecationWarning
           +-- PendingDeprecationWarning
           +-- RuntimeWarning
           +-- SyntaxWarning
           +-- UserWarning
           +-- FutureWarning
           +-- ImportWarning
           +-- UnicodeWarning
           +-- BytesWarning
           +-- EncodingWarning
           +-- ResourceWarning
           
'''

            
    