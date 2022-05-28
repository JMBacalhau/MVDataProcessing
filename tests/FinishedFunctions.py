# -*- coding: utf-8 -*-
"""
Created on Sun May 22 20:48:11 2022

@author: Bacalhau
"""

import pandas
import numpy
import datetime
from datetime import datetime
from itertools import permutations


def TimeProfile(time_stopper: list,name: str = '',show: bool = False,estimate_for: int = 0):
    """
    Simple code profiler.
    
    
    :param time_stopper: A List that will hold all the stop times.
    :type time_stopper: list
    
    :param name: A name for this instance of time profile. Defaults to empty.
    :type name: str, optional
    
    :param show: If True shows the data on the console. Defaults to False.
    :type show: bool, optional
    
    :param estimate_for: A multiplier to be applied at the end. Takes the whole time analized and multiplies by "estimate_for".
    :type estimate_for: int
    
    :return: None
    :rtype: None
    
    How to use:
        
    Create a list ->  time_stopper = []   
    
    Put a -> time_stopper.append(['time_init',time.perf_counter()]) at the beginning.
    
    Put time_stopper.append(['Func_01',time.perf_counter()]) after the code block with the fist parameter beeing a name and
    the second beeing the time.
    
    Call this function at the end.
    
    Example:
    
    time_stopper.append(['time_init',time.perf_counter()])
    
    func1()
    time_stopper.append(['func1',time.perf_counter()])
    func2()
    time_stopper.append(['func2',time.perf_counter()])
    func3()
    time_stopper.append(['func3',time.perf_counter()])
    func4()
    time_stopper.append(['func4',time.perf_counter()])
     
    TimeProfile(time_stopper,'My Profiler',show=True,estimate_for=500)
    
    The estimate_for parameter makes the calculation as if you would run x times the code analyzed.
    
    """
    
    
    if(show):
        print("Profile: " + name)
        time_stopper = pandas.DataFrame(time_stopper,columns=['Type','time'])    
        #time_stopper['time'] = time_stopper['time']-time_stopper['time'].min()    
        time_stopper['Delta'] = time_stopper['time'] - time_stopper['time'].shift(periods=1, fill_value=0)    
        time_stopper = time_stopper.iloc[1:,:]
        time_stopper['%'] =  numpy.round(100*time_stopper['Delta']/time_stopper['Delta'].sum(),2)
        total_estimate = time_stopper['Delta'].sum()
        time_stopper = time_stopper.append(pandas.DataFrame([['Total',numpy.nan,time_stopper['Delta'].sum(),100]],columns=['Type','time','Delta','%']))    
        print(time_stopper)
        if(estimate_for!=0):
            print(f"Estimation for {estimate_for} runs: {numpy.round(total_estimate*estimate_for/(60*60),2)} hours.")

    return

#BUG Some sample_freq have trouble lol.
def DataSynchronization(x_in: pandas.core.frame.DataFrame,
              start_date_dt: datetime,
              end_date_dt: datetime,
              sample_freq: int = 5,
              sample_time_base: str = 'm') -> pandas.core.frame.DataFrame:        
    """
    Makes the Data Synchronization between the columns (time series) of the data provided.
    
    Theory background.:
        
    The time series synchronization is the first step in processing the dataset. The synchronization is vital 
    since the alignment between phases (φa, φb, φv) of the same quantity, between quantities (V, I, pf) of the
    same feeder, and between feeders, provides many advantages. The first one being the ability to combine all
    nine time series, the three-phase voltage, current, and power factor of each feeder to calculate the secondary
    quantities (Pactive/Preactive, Eactive/Ereactive).
    
    Furthermore, the synchronization between feeders provides the capability to analyze the iteration between them,
    for instance, in load transfers for scheduled maintenance and to estimate substation’s transformers quantities
    by the sum of all feeders.
         
    Most of the fuctions in this module assumes that the time series are "Clean" to a certain sample_freq. Therefore,
    this fuction must be executed first on the dataset.
    
    
    :param x_in: A pandas.core.frame.DataFrame where the index is of type "pandas.core.indexes.datetimes.DatetimeIndex" and each column contain an electrical
    quantity time series.    
    :type x_in: pandas.core.frame.DataFrame
        
    :param start_date_dt: The start date where the synchronization should start. 
    :type start_date_dt: datetime
    
    :param end_date_dt: The end date where the synchronization will consider samples.  
    :type end_date_dt: datetime
    
    :param sample_freq: The sample frequency of the time series. Defaults to 5.  
    :type sample_freq: int,optional
    
    :param sample_time_base: The base time of the sample frequency. Specify if the sample frequency is in (D)ay, (M)onth, (Y)ear, (h)ours, (m)inutes,
    or (s)econds. Defaults to (m)inutes.  
    :type sample_time_base: srt,optional
    
    
    :raises Exception: if x_in has no DatetimeIndex. 
    :raises Exception: if start_date_dt not in datetime format.
    :raises Exception: if end_date_dt not in datetime format.
    :raises Exception: if sample_time_base is not in (D)ay, (M)onth, (Y)ear, (h)ours, (m)inutes, or (s)econds.
    
    
    :return: Y: The synchronized pandas.core.frame.DataFrame
    :rtype: Y: pandas.core.frame.DataFrame

    """
    
    #-------------------#
    # BASIC INPUT CHECK #
    #-------------------#
    
    if not(isinstance(x_in.index, pandas.DatetimeIndex)):  raise Exception("DataFrame has no DatetimeIndex.")
    if not(isinstance(start_date_dt, datetime)):  raise Exception("Date not in datetime format.")
    if not(isinstance(end_date_dt, datetime)):  raise Exception("Date not in datetime format.")
    if sample_time_base not in ['s','m','h','D','M','Y']:  raise Exception("Same base time not valid. Ex. ['s','m','h','D','M','Y'] ")
    
    #-------------------#
        
    added_dic = {'s':'ms','m':'s','h':'m','D':'h','M':'D','Y':'M'}
    floor_dic = {'s':'S','m':'T','h':'H','D':'D','M':'M','Y':'Y'}    
        
    x_in.index = x_in.index.tz_localize(None) #Makes the datetimeIndex naive (no time zone)
    
    #----------------------------------------------------------------#
    #Creates a base vector that conntainXs all the samples between data_inicio and data_final filled timestamp and with nan
    
    qty_data = len(x_in.columns)
               
    timearray = numpy.arange(start_date_dt, end_date_dt,numpy.timedelta64(sample_freq,sample_time_base), dtype='datetime64')
    timearray = timearray + numpy.timedelta64(1,added_dic[sample_time_base]) # ADD a second/Minute/Hour/Day/Month to the end so during the sort 
                                                                             # this samples will be at last (HH:MM:01)
    
    vet_amostras = pandas.DataFrame(index=timearray,columns=range(qty_data), dtype=object)
    vet_amostras.index.name = 'timestamp'
    
        
    #----------------------------------------------------------------#
    #Creates the output dataframe which is the same but witohut the added second.
       
    Y = vet_amostras.copy(deep=True)    
    Y.index = Y.index.floor(floor_dic[sample_time_base])#Flush the seconds    
    
    #----------------------------------------------------------------#
    #Saves the name of the columns
    save_columns_name = x_in.columns.values    
    
    #----------------------------------------------------------------#
    #Start to process each column
    
    fase_list = numpy.arange(0,x_in.shape[1])   
    
    for fase in fase_list:                
        
        X = x_in.copy(deep=True)
        X.columns = Y.columns
        X = X.loc[~X.iloc[:,fase].isnull(),fase]#Gets only samples on the phase of interest                
        X = X[numpy.logical_and(X.index<end_date_dt,X.index>=start_date_dt)]#Get samples on between the start and end of the period of study
        
        if(X.shape[0]!=0):            
            
            #Process samples that are multiple of sample_freq
            df_X = X.copy(deep=True)
            df_vet_amostras = vet_amostras[fase] 
                        
            #remove seconds (00:00:00) to put this specific samples at the beginning during sort
            df_X = df_X.sort_index(ascending=True)#Ensures the sequence of timestamps
            df_X.index = df_X.index.round('1'+floor_dic[sample_time_base])#Remove seconds, rounding to the nearest minute
            df_X = df_X[df_X.index.minute % sample_freq == 0]#Samples that are multiple of sample_freq have preference 
            
            if(df_X.empty != True):
                              
                df_X = df_X[~df_X.index.duplicated(keep='first')]#Remove unecessary duplicates     
               
                #joins both vectors
                df_aux = pandas.concat([df_X, df_vet_amostras])
                df_aux = df_aux.sort_index(ascending=True)#Ensures the sequence of timestamps    
                
                #Elimina segundos (00:00:00), e elimina duplicatas deixando o X quando existente e vet_amostras quando não existe a amostra
                df_aux.index = df_aux.index.floor(floor_dic[sample_time_base])
                df_aux = df_aux[~df_aux.index.duplicated(keep='first')]#Remove unecessary duplicates     
                
                #Make sure that any round up that ended up out of the period of study is removed
                df_aux = df_aux[numpy.logical_and(df_aux.index<end_date_dt,df_aux.index>=start_date_dt)]
                
                Y.loc[:,fase] = df_aux
                
                
                
            #Process samples that are NOT multiple of sample_freq
            df_X = X.copy(deep=True)
            df_vet_amostras = vet_amostras[fase]                               
                            
            #remove seconds (00:00:00) to put this specific samples at the beginning during sort
            df_X = df_X.sort_index(ascending=True)#Ensures the sequence of timestamps
            df_X.index = df_X.index.round('1'+floor_dic[sample_time_base])#Remove seconds, rounding to the nearest minute
            df_X = df_X[df_X.index.minute % sample_freq != 0]#Samples that are NOT multiple of sample_freq have preference 
                            
            
            if(df_X.empty != True):
               
                                               
                df_X.index = df_X.index.round(str(sample_freq)+floor_dic[sample_time_base])#Aproximate sample to the closest multiple of sample_freq
                
                df_X = df_X[~df_X.index.duplicated(keep='first')]#Remove unecessary duplicates     
               
                #joins both vectors
                df_aux = pandas.concat([df_X, df_vet_amostras])
                df_aux = df_aux.sort_index(ascending=True)#Ensures the sequence of timestamps                
                
                #Remove seconds (00:00:00), and remove ducplicates leaving X when there is data and vet amostra when its empty
                df_aux.index = df_aux.index.floor(floor_dic[sample_time_base])
                df_aux = df_aux[~df_aux.index.duplicated(keep='first')]#Remove unecessary duplicates     
                
                
                #Make sure that any round up that ended up out of the period of study is removed
                df_aux = df_aux[numpy.logical_and(df_aux.index<end_date_dt,df_aux.index>=start_date_dt)]
                                                
                #Copy data to the output vecto olny if there not data there yet.
                Y.loc[Y.iloc[:,fase].isnull(),fase] = df_aux.loc[Y.iloc[:,fase].isnull()]
    
    
    #----------------------------------------------------------------#
    #Last operations before the return of Y
    
    Y = Y.astype(float)    
    Y.columns = save_columns_name# Gives back the original name of the columns in x_in
    
    return Y

def IntegrateHour(x_in: pandas.DataFrame,sample_freq: int = 5,sample_time_base: str = 'm') -> pandas.DataFrame:
    """
    Integrates the input pandas.core.frame.DataFrame to an hour samples.
    
    :param x_in: A pandas.core.frame.DataFrame where the index is of type "pandas.core.indexes.datetimes.DatetimeIndex" and each column contain an electrical
    quantity time series.    
    :type x_in: pandas.core.frame.DataFrame

    :param sample_freq: The sample frequency of the time series. Defaults to 5.  
    :type sample_freq: int,optional
    
    :param sample_time_base: The base time of the sample frequency. Specify if the sample frequency is in (m)inutes or (s)econds. Defaults to (m)inutes.  
    :type sample_time_base: srt,optional


    :raises Exception: if x_in has no DatetimeIndex. 
    
    
    :return: Y: The pandas.core.frame.DataFrame integrated by hour.
    :rtype: Y: pandas.core.frame.DataFrame

    """
    hour_divider = {'s':60*60,'m':60}
    
    #-------------------#
    # BASIC INPUT CHECK #
    #-------------------#
    
    if not(isinstance(x_in.index, pandas.DatetimeIndex)):  raise Exception("DataFrame has no DatetimeIndex.")
    
    Y = x_in.copy(deep=True)
    
    time_vet_stamp = Y.index[numpy.arange(0,len(Y.index),int(hour_divider[sample_time_base]/sample_freq))]    
    Y = Y.groupby([Y.index.year,Y.index.month,Y.index.day,Y.index.hour]).mean() 
    Y = Y.reset_index(drop=True)
    Y.insert(0,'timestamp', time_vet_stamp)
    Y.set_index('timestamp', inplace=True)
    
    return Y

def Correlation(x_in: pandas.DataFrame) -> float:
    """
    Calculates the correlation between each column of the DataFrame and outputs the average of all.    
    
    
    :param x_in: A pandas.core.frame.DataFrame where the index is of type "pandas.core.indexes.datetimes.DatetimeIndex" and each column contain an electrical
    quantity time series.    
    :type x_in: pandas.core.frame.DataFrame


    :return: corr_value: Value of the correlation
    :rtype: corr_value: float

    """
    
    corr_value = x_in.corr()[x_in.corr()!=1].mean().mean()
    
    return corr_value

def DayPeriodMapper(hour: int) -> int:
    """
    Maps a given hour to one of four periods of a day.
    
    For 0 to 5 (hour) -> 0 night
    For 6 to 11 (hour) -> 1 moorning
    For 12 to 17 (hour) -> 2 afternoon
    For 18 to 23 (hour) -> 3 evening
    
    :param hour: an hour of the day between 0 and 23.
    :type hour: int
    
    :return: mapped: Period of the day
    :rtype: mapped: int
    
    """
    
    
    return (
        0 if 0 <= hour < 6
        else
        1 if 6 <= hour < 12
        else
        2 if 12 <= hour < 18
        else
        3
    )

def DayPeriodMapperVet(hour: pandas.Series) -> pandas.Series:
    """
    Maps a given hour to one of four periods of a day.
    
    For 0 to 5 (hour) -> 0 night
    For 6 to 11 (hour) -> 1 moorning
    For 12 to 17 (hour) -> 2 afternoon
    For 18 to 23 (hour) -> 3 evening
    
    
    :param hour: A pandas.core.frame.Series with values between 0 and 23 to map each hour in the series to a period of the day. 
    this is a "vector" format for DayPeriodMapper function.
    :type hour: pandas.core.frame.Series
    
    :return: period_day: The hour pandas.core.frame.Series mapped to periods of the day
    :rtype: period_day: pandas.core.frame.Series
    
    """
    
    map_dict = {0:0,1:0,2:0,3:0,4:0,5:0,
                6:1,7:1,8:1,9:1,10:1,11:1,
                12:2,13:2,14:2,15:2,16:2,17:2,
                18:3,19:3,20:3,21:3,22:3,23:3}
    
    period_day = hour.map(map_dict)
    
    return period_day

def YearPeriodMapperVet(month: pandas.Series) -> pandas.Series:
    """
    Maps a given month to one of two periods of an year, being dry and humid .
    
    For october to march (month) -> 0 humid
    For april to september (month) -> 1 dry
    
    
    :param month: A pandas.core.frame.Series with values between 0 and 12 to map each month in the series to dry or humid.
    
    :return: season: The months pandas.core.frame.Series mapped to dry or humid.
    :rtype: season: pandas.core.frame.Series
    
    """
    
    map_dict = {10:0,11:0,12:0,1:0,2:0,3:0,
                4:1,5:1,6:1,7:1,7:1,9:1}
    
    season = month.map(map_dict)
    
    return season