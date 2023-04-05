"""
Created on Mon Apr 18 06:45:44 2022

@author: Bacalhau


Part2
0) Test Previous functions
1) Still need to finish GetWeekDayCurve. Se comment.
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


def ReturnOnlyValidDays(x_in: pd.DataFrame,
                        sample_freq: int = 5,
                        threshold_accept: float = 1.0,
                        sample_time_base: str = 'm',
                        remove_from_process=[]) -> tuple:
    """
    Returns all valid days. A valid day is one with no missing values for any 
    of the timeseries on each column.
    
    
    :param x_in: A pandas.core.frame.DataFrame where the index is of type "pandas.core.indexes.datetime.DatetimeIndex"
    and each column contain an electrical quantity time series.
    :type x_in: pandas.core.frame.DataFrame
    
    :param sample_freq: The sample frequency of the time series. Defaults to 5.  
    :type sample_freq: int,optional
    
    :param threshold_accept: The amount of samples that is required to consider a valid day. Defaults to 1 (100%).  
    :type threshold_accept: float,optional
    
    :param sample_time_base: The base time of the sample frequency. Specify if the sample frequency is in (h)ours,
    (m)inutes, or (s)econds. Defaults to (m)inutes.
    :type sample_time_base: srt,optional
    
    :param remove_from_process: Columns to be kept off the process;  
    :type remove_from_process: list,optional
    
         
    :raises Exception: if x_in has no DatetimeIndex. 
    :raises Exception: if sample_time_base is not in seconds, minutes or hours.
    
    
    :return: Y: A tupole with the pandas.core.frame.DataFrame with samples filled based on the proportion
    between time series and the number of valid days
    :rtype: Y: tuple

    """

    # BASIC INPUT CHECK
    
    if not(isinstance(x_in.index, pd.DatetimeIndex)):
        raise Exception("DataFrame has no DatetimeIndex.")
    if sample_time_base not in ['s', 'm', 'h']:
        raise Exception("The sample_time_base is not in seconds, minutes or hours.")

    X = x_in.copy(deep=True)
    
    if len(remove_from_process) > 0:
        X = X.drop(remove_from_process, axis=1)

    qty_sample_dic = {'s': 24 * 60 * 60, 'm': 24 * 60, 'h': 24}

    df_count = X.groupby([X.index.year, X.index.month, X.index.day]).count() / (
                qty_sample_dic[sample_time_base] / sample_freq)

    time_vet_stamp = X.index[np.arange(0, len(X.index), int((qty_sample_dic[sample_time_base] / sample_freq)))]
    df_count = df_count.reset_index(drop=True)
    df_count.insert(0, 'timestamp_day', time_vet_stamp)
    df_count.set_index('timestamp_day', inplace=True)
    df_count = df_count >= threshold_accept
    
    df_count = df_count.sum(axis=1) == df_count.shape[1]
    df_count.name = 'isValid'
    df_count = df_count.reset_index()
    X['timestamp_day'] = X.index.floor("D").values

    keep_X_index = X.index
    X = pd.merge(X, df_count, on='timestamp_day', how='left')
    X.index = keep_X_index
    X = X.loc[X['isValid'] == True, :]

    X.drop(columns=['isValid', 'timestamp_day'], inplace=True)

    return X, df_count


def GetDayMaxMin(x_in, start_date_dt, end_date_dt, sample_freq=5, threshold_accept=1.0, exe_param='max'):
    """
    Returns a tuple of pandas.core.frame.DataFrame containing the values of maximum or minimum of each day
    and the timestamp of each occurrence. For each weekday that is not a valid day the maximum or minimum
    is interpolated->ffill->bff. The interpolation is made regarding each weekday.

    :param x_in: A pandas.core.frame.DataFrame where the index is of type "pandas.core.indexes.datetime.DatetimeIndex"
    and each column contain an electrical quantity time series.
    :type x_in: pandas.core.frame.DataFrame

    :param start_date_dt:
    :param end_date_dt:

    :param sample_freq: The sample frequency of the time series. Defaults to 5.
    :type sample_freq: int,optional

    :param threshold_accept: The amount of samples that is required to consider a valid day. Defaults to 1 (100%).
    :type threshold_accept: float,optional

    :param exe_param: 'max' return the maximum and min return the minimum value of each valid day
    (Default value = 'max')
    :type exe_param: srt,optional

    :return: Y: The first parameter is a pandas.core.frame.DataFrame with maximum value for each day
    and the second parameter pandas.core.frame.DataFrame with the timestamps.
    :rtype: Y: tuple
    """

    # BASIC INPUT CHECK
    
    if not(isinstance(x_in.index, pd.DatetimeIndex)):
        raise Exception("DataFrame has no DatetimeIndex.")

    X = x_in.copy(deep=True)

    X, _ = ReturnOnlyValidDays(X, sample_freq, threshold_accept)

    if exe_param == 'max':
        Y = X.groupby([X.index.year, X.index.month, X.index.day]).max()
        vet_idx = X.groupby([X.index.year, X.index.month, X.index.day]).idxmax()
    else:
        Y = X.groupby([X.index.year, X.index.month, X.index.day]).min()
        vet_idx = X.groupby([X.index.year, X.index.month, X.index.day]).idxmin()

    # redo the timestamp index
    vet_idx.index.rename(['Year', 'Month', 'Day'], inplace=True)
    vet_idx = vet_idx.reset_index(drop=False)

    time_vet_stamp = pd.to_datetime(
        vet_idx['Year'].astype(str) + '-' + vet_idx['Month'].astype(str) + '-' + vet_idx['Day'].astype(str))

    vet_idx.drop(columns=['Year', 'Month', 'Day'], axis=1, inplace=True)
    vet_idx = vet_idx.reset_index(drop=True)
    vet_idx.insert(0, 'timestamp_day', time_vet_stamp)
    vet_idx.set_index('timestamp_day', inplace=True)

    # redo the timestamp index
    Y.index.rename(['Year', 'Month', 'Day'], inplace=True)
    Y = Y.reset_index(drop=False)

    time_vet_stamp = pd.to_datetime(Y['Year'].astype(str) + '-' + Y['Month'].astype(str) + '-' + Y['Day'].astype(str))

    Y.drop(columns=['Year', 'Month', 'Day'], axis=1, inplace=True)
    Y = Y.reset_index(drop=True)
    Y.insert(0, 'timestamp_day', time_vet_stamp)
    Y.set_index('timestamp_day', inplace=True)

    Y = f_remove.DataSynchronization(Y, start_date_dt, end_date_dt, sample_freq=1, sample_time_base='D')

    vet_idx = pd.merge(vet_idx, Y, left_index=True, right_index=True, how='right', suffixes=('', '_remove'))
    vet_idx.drop(columns=vet_idx.columns[vet_idx.columns.str.contains('_remove')], axis=1, inplace=True)

    # Missing days get midnight as the  hour of max and min
    for col in vet_idx.columns.values:
        vet_idx.loc[vet_idx[col].isna(), col] = vet_idx.index[vet_idx[col].isna()]

    # Interpolate by day of the week
    Y = Y.groupby(Y.index.weekday, group_keys=False).apply(lambda x: x.interpolate())
    Y = Y.groupby(Y.index.weekday, group_keys=False).apply(lambda x: x.ffill())
    Y = Y.groupby(Y.index.weekday, group_keys=False).apply(lambda x: x.bfill())

    return Y, vet_idx


def GetWeekDayCurve(x_in, sample_freq=5, threshold_accept=1.0, min_sample_per_day=3, min_sample_per_workday=9):
    """

    :param x_in: param sample_freq:  (Default value = 5)
    :param threshold_accept: Default value = 1.0)
    :param min_sample_per_day: Default value = 3)
    :param min_sample_per_workday: Default value = 9)
    :param sample_freq:  (Default value = 5)

    """

    # BASIC INPUT CHECK

    if not (isinstance(x_in.index, pd.DatetimeIndex)):
        raise Exception("DataFrame has no DatetimeIndex.")

    x_in = output.copy(deep=True)
    
    X = x_in.copy(deep=True)

    Y, df_count = ReturnOnlyValidDays(X, sample_freq, threshold_accept)

    # Get valid data statistics
    df_count = df_count.loc[df_count['isValid'], :]
    df_stats = df_count.groupby(df_count['timestamp_day'].dt.weekday).count()
    days_unique = df_stats.shape[0]
    count_days_unique = df_stats['timestamp_day'].values

    # Has enough data do use ?
    if (days_unique == 7) and (np.min(count_days_unique) >= min_sample_per_day):
        print('Can calculate a curve for every weekday')

        Y = Y.groupby([Y.index.weekday, Y.index.hour, Y.index.minute]).mean()
        Y.index.names = ['WeekDay', 'Hour', 'Min']
        Y = Y.reset_index()

        # Normalization max min each day
        grouper = Y.groupby([Y.WeekDay])
        maxes = grouper.transform('max')
        mins = grouper.transform('min')

        Y.iloc[:, 3:] = (Y.iloc[:, 3:] - mins.iloc[:, 2:]) / (maxes.iloc[:, 2:] - mins.iloc[:, 2:])
        
    else:
        work_days = df_stats.loc[df_stats.index <= 4, 'timestamp_day'].sum()
        sat_qty = df_stats.loc[df_stats.index == 5, 'timestamp_day'].sum()
        sun_qty = df_stats.loc[df_stats.index == 6, 'timestamp_day'].sum()

        if (work_days >= min_sample_per_workday) and sun_qty >= min_sample_per_day and sat_qty >= min_sample_per_day:
            print('Can calculate a curve for every weekday and use Sat. and Sun.')

            Y = Y.groupby([Y.index.weekday, Y.index.hour, Y.index.minute]).mean()
            Y.index.names = ['WeekDay', 'Hour', 'Min']
            Y = Y.reset_index()

            # Normalization max min each day
            grouper = Y.groupby([Y.WeekDay])
            maxes = grouper.transform('max')
            mins = grouper.transform('min')

            Y.iloc[:, 3:] = (Y.iloc[:, 3:] - mins.iloc[:, 2:]) / (maxes.iloc[:, 2:] - mins.iloc[:, 2:])
            
            
            #FALTA PEGAR UM DIA DA SEMANA MAIS PROXIMO PARA COMPLETAR OS INEXISTENTES
            
        else:
            print('Use default curve.')
            
            #FALTA ESCREVER UMA DEFAULT E PERMITIR IMPORTAR
        
    return Y
   
    
def GetNSSCPredictedSamples(max_vet: pd.DataFrame,
                            min_vet: pd.DataFrame,
                            weekday_curve: pd.DataFrame,
                            sample_freq: int = 5,
                            sample_time_base: str = 'm') -> pd.DataFrame:

    # BASIC INPUT CHECK

    if sample_time_base not in ['s', 'm', 'h']:
        raise Exception("The sample_time_base is not in seconds, minutes or hours.")

    max_vet = max_vet.iloc[np.repeat(np.arange(len(max_vet)), 12*24)]
    min_vet = min_vet.iloc[np.repeat(np.arange(len(min_vet)), 12*24)]

    time_array = np.arange(start_date_dt, end_date_dt, np.timedelta64(sample_freq, sample_time_base),
                           dtype='datetime64')

    vet_samples = pd.DataFrame(index=time_array, dtype=object)
    vet_samples.index.name = 'timestamp'

    num_days = int(vet_samples.shape[0] / (12 * 24))
    first_day = vet_samples.index[0].weekday()

    weekday_curve_vet_begin = weekday_curve.iloc[(first_day * 12 * 24):, :].reset_index(drop=True)
    num_mid_weeks = int(np.floor((num_days - (7 - first_day)) / 7))
    weekday_curve_vet_mid = pd.concat([weekday_curve] * num_mid_weeks)
    num_end_days = num_days - num_mid_weeks * 7 - (7 - first_day)
    weekday_curve_vet_end = weekday_curve.iloc[:num_end_days * (12 * 24), :].reset_index(drop=True)

    weekday_curve_vet = pd.concat([weekday_curve_vet_begin, weekday_curve_vet_mid, weekday_curve_vet_end])

    weekday_curve_vet = weekday_curve_vet.reset_index(drop=True)

    print(weekday_curve_vet)
    weekday_curve_vet.drop(columns=['WeekDay', 'Hour', 'Min'], inplace=True)
    weekday_curve_vet.index.name = 'timestamp'
    weekday_curve_vet.index = vet_samples.index

    max_vet.index = vet_samples.index
    min_vet.index = vet_samples.index

    Y = (max_vet - min_vet) * weekday_curve_vet + min_vet

    return Y


if __name__ == "__main__":
    
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
 
    '''    
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
    
    '''


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
    

    fig, ax = plt.subplots()
    ax.plot(output.values)
    ax.set_title('Input')
    
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
    output = f_remove.RemoveOutliersHistoGram(output,min_number_of_samples_limit=12*3)            
    f_remove.CountMissingData(output,show=True)
    time_stopper.append(['RemoveOutliersHistoGram',time.perf_counter()])
    
    output.iloc[50000:60000,:] = np.nan
    output.iloc[:10000, :] = np.nan

    fig, ax = plt.subplots()
    ax.plot(output.index.values,output.values)
    ax.set_title('No outliers')


    X, _ = ReturnOnlyValidDays(output, sample_freq=5, threshold_accept=0.2)



    output = f_remove.PhaseProportionInput(output,threshold_accept = 0.60,remove_from_process=['IN'])
    f_remove.CountMissingData(output,show=True)
    time_stopper.append(['PhaseProportionInput',time.perf_counter()])

    fig, ax = plt.subplots()
    ax.plot(output.index.values,output.values)
    ax.set_title('PhaseProportionInput')

    

    #NSSC Implementation    
    max_vet,_ = GetDayMaxMin(output,start_date_dt,end_date_dt,sample_freq = 5,threshold_accept = 0.2,exe_param='max')     
    time_stopper.append(['maxVet',time.perf_counter()])
    min_vet,_ = GetDayMaxMin(output,start_date_dt,end_date_dt,sample_freq = 5,threshold_accept = 0.2,exe_param='min')    
    time_stopper.append(['minVet',time.perf_counter()])
    weekday_curve = GetWeekDayCurve(output,sample_freq = 5,threshold_accept = 0.6)
    time_stopper.append(['WeekCurve',time.perf_counter()])
    X_pred = GetNSSCPredictedSamples(max_vet, min_vet, weekday_curve, sample_freq=5, sample_time_base='m')

    

    fig, ax = plt.subplots()
    ax.plot(X_pred.values)
    ax.set_title('X_pred')



    #Criarr as regras para substituir X_pred no vetor x_in
    
    
    num_samples_day = 12*24
    day_threshold = 0.5
    patamar_threshold = 0.5
    num_samples_patamar = 12*6
    sample_freq = 5
    sample_time_base = 'm'
    
    output_isnull_day = output.isnull().groupby([output.index.day,output.index.month,output.index.year]).sum()    
    output_isnull_day.columns = output_isnull_day.columns.values + "_mark"
    output_isnull_day = output_isnull_day/num_samples_day
    
    output_isnull_day.index.rename(['day','month','year'],inplace=True)    
    output_isnull_day.reset_index(inplace=True)    
    output_isnull_day.set_index(output_isnull_day['day'].astype(str) + '-' + output_isnull_day['month'].astype(str) + '-' + output_isnull_day['year'].astype(str),inplace=True)
    output_isnull_day.drop(columns = ['day', 'month', 'year'],inplace=True)
    
    
    output_isnull_day = output_isnull_day>=day_threshold        
    output_isnull_day = output_isnull_day.loc[~(output_isnull_day.sum(axis=1)==0),:]    
    
       
    
    
    output_isnull_patamar = output.copy(deep=True)
    output_isnull_patamar['dp'] = output_isnull_patamar.index.hour.map(f_remove.DayPeriodMapper)
    output_isnull_patamar = output.isnull().groupby([output_isnull_patamar.index.day,output_isnull_patamar.index.month,output_isnull_patamar.index.year,output_isnull_patamar.dp]).sum()        
    output_isnull_patamar.columns = output_isnull_patamar.columns.values + "_mark"
    output_isnull_patamar =output_isnull_patamar/num_samples_patamar
    
    output_isnull_patamar.index.rename(['day', 'month', 'year','dp'],inplace=True)   
    output_isnull_patamar.reset_index(inplace=True)    
    output_isnull_patamar.set_index(output_isnull_patamar['day'].astype(str) + '-' + output_isnull_patamar['month'].astype(str) + '-' + output_isnull_patamar['year'].astype(str) + '-' + output_isnull_patamar['dp'].astype(str),inplace=True)
    output_isnull_patamar.drop(columns = ['day', 'month', 'year','dp'],inplace=True)
    
    
    output_isnull_patamar = output_isnull_patamar>=patamar_threshold        
    output_isnull_patamar = output_isnull_patamar.loc[~(output_isnull_patamar.sum(axis=1)==0),:]    
    
    
    timearray = np.arange(start_date_dt, end_date_dt,np.timedelta64(sample_freq,sample_time_base), dtype='datetime64')    
    mark_substitute = pd.DataFrame(index=timearray,columns = output.columns.values, dtype=object)    
    mark_substitute.index.name = 'timestamp'
    mark_substitute.loc[:,:] = False
       
    
    
    
    
    index_day = { 'day': output.index.day.values.astype(str), 'month': output.index.month.values.astype(str), 'year': output.index.year.values.astype(str) }
    index_day = pd.DataFrame(index_day)    
    index_day = index_day['day'].astype(str) + '-' + index_day['month'].astype(str) + '-' + index_day['year'].astype(str)
    
    index_patamar = { 'day': output.index.day.values.astype(str), 'month': output.index.month.values.astype(str), 'year': output.index.year.values.astype(str) }
    index_patamar = pd.DataFrame(index_patamar)    
    index_patamar['dp'] = output.index.hour.map(f_remove.DayPeriodMapper)
    index_patamar = index_patamar['day'].astype(str) + '-' + index_patamar['month'].astype(str) + '-' + index_patamar['year'].astype(str) + '-' + index_patamar['dp'].astype(str)
    
            
    mark_substitute['index_patamar'] = index_patamar.values
    mark_substitute = pd.merge(mark_substitute, output_isnull_patamar,left_on='index_patamar',right_index=True,how='left').fillna(False)
    for col in output.columns.values:
        mark_substitute[col] = mark_substitute[col+'_mark']
        mark_substitute.drop(columns=[col+'_mark'],axis=1,inplace=True)
        
    mark_substitute.drop(columns=['index_patamar'],axis=1,inplace=True)
    
    mark_substitute['index_day'] = index_day.values
    mark_substitute = pd.merge(mark_substitute, output_isnull_day,left_on='index_day',right_index=True,how='left').fillna(False)    
    
    for col in output.columns.values:
        mark_substitute[col] = mark_substitute[col+'_mark']
        mark_substitute.drop(columns=[col+'_mark'],axis=1,inplace=True)
        
    mark_substitute.drop(columns=['index_day'],axis=1,inplace=True)

    
    
    output[mark_substitute] = X_pred[mark_substitute]

    time_stopper.append(['NSSC', time.perf_counter()])
    
    fig, ax = plt.subplots()
    ax.plot(output.values)
    ax.set_title('Output')
    plt.show()

    '''
    #Simple Process
   
    output = f_remove.SimpleProcess(output,start_date_dt,end_date_dt,remove_from_process= ['IN'],sample_freq= 5,sample_time_base = 'm',pre_interpol = 12,pos_interpol = 12,prop_phases = True, integrate = False, interpol_integrate = 3)
  
    f_remove.CountMissingData(output,show=True)
    time_stopper.append(['PhaseProportionInput',time.perf_counter()])
    '''


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
    
    f_remove.TimeProfile(time_stopper,name='Main',show=True,estimate_for=5*1000)
