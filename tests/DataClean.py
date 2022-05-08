"""
Created on Mon Apr 18 06:45:44 2022

@author: Bacalhau

Verificar se todos os d_avoid_ estao implementados
Raise exception and erros


Proporção entre fases
Input


Example Google style docstrings.

This module demonstrates documentation as specified by the `Google
Python Style Guide`_. Docstrings may extend over multiple lines.
Sections are created with a section header and a colon followed by a
block of indented text.

Example:
    Examples can be given using either the ``Example`` or ``Examples``
    sections. Sections support any reStructuredText formatting, including
    literal blocks::

        $ python example_google.py

Section breaks are created by resuming unindented text. Section breaks
are also implicitly created anytime a new section starts. 

Attributes:
    module_level_variable1 (int): Module level variables may be documented in
        either the ``Attributes`` section of the module docstring, or in an
        inline docstring immediately following the variable.

        Either form is acceptable, but the two should not be mixed. Choose
        one convention to document module level variables and be consistent
        with it.

Todo:
    * For module TODOs
    * You have to also use ``sphinx.ext.todo`` extension

.. _Google Python Style Guide:   
http://google.github.io/styleguide/pyguide.html

"""

import pandas as pd
import datetime as dt
import numpy as np
from datetime import datetime
from itertools import combinations

 
#pd.set_option('display.max_rows', None)
#pd.set_option('display.max_columns', None)
#pd.set_option('display.width', None)
#pd.set_option('display.max_colwidth', None)


def DataSynchronization(x_in: pd.DataFrame,
              start_date_dt: datetime,
              end_date_dt: datetime,
              sample_freq: int = 5,
              sample_time_base: str = 'm') -> pd.DataFrame:        
    """
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
    
    sample_time_base: [np.timedelta64]: (D)ay, (M)onth, (Y)ear, (h)ours, (m)inutes, or (s)econds.
    

    """
    
    added_dic = {'s':'ms','m':'s','h':'m','D':'h','M':'D','Y':'M'}
    floor_dic = {'s':'S','m':'T','h':'H','D':'D','M':'M','Y':'Y'}    
        
    x_in.index = x_in.index.tz_localize(None) #Makes the datetimeIndex naive (no time zone)
    
    #----------------------------------------------------------------#
    #Creates a base vector that conntainXs all the samples between data_inicio and data_final filled timestamp and with nan
    
    qty_data = len(x_in.columns)
               
    timearray = np.arange(start_date_dt, end_date_dt,np.timedelta64(sample_freq,sample_time_base), dtype='datetime64')
    timearray = timearray + np.timedelta64(1,added_dic[sample_time_base]) # ADD a second/Minute/Hour/Day/Month to the end so during the sort this samples will be at last (HH:MM:01)
    
    vet_amostras = pd.DataFrame(index=timearray,columns=range(qty_data), dtype=object)
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
    
    fase_list = np.arange(0,x_in.shape[1])   
    
    for fase in fase_list:                
        
        X = x_in.copy(deep=True)
        X.columns = Y.columns
        X = X.loc[~X.iloc[:,fase].isnull(),fase]#Gets only samples on the phase of interest                
        X = X[np.logical_and(X.index<end_date_dt,X.index>=start_date_dt)]#Get samples on between the start and end of the period of study
        
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
                df_aux = pd.concat([df_X, df_vet_amostras])
                df_aux = df_aux.sort_index(ascending=True)#Ensures the sequence of timestamps    
                
                #Elimina segundos (00:00:00), e elimina duplicatas deixando o X quando existente e vet_amostras quando não existe a amostra
                df_aux.index = df_aux.index.floor(floor_dic[sample_time_base])
                df_aux = df_aux[~df_aux.index.duplicated(keep='first')]#Remove unecessary duplicates     
                
                #Make sure that any round up that ended up out of the period of study is removed
                df_aux = df_aux[np.logical_and(df_aux.index<end_date_dt,df_aux.index>=start_date_dt)]
                
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
                df_aux = pd.concat([df_X, df_vet_amostras])
                df_aux = df_aux.sort_index(ascending=True)#Ensures the sequence of timestamps                
                
                #Remove seconds (00:00:00), and remove ducplicates leaving X when there is data and vet amostra when its empty
                df_aux.index = df_aux.index.floor(floor_dic[sample_time_base])
                df_aux = df_aux[~df_aux.index.duplicated(keep='first')]#Remove unecessary duplicates     
                
                
                #Make sure that any round up that ended up out of the period of study is removed
                df_aux = df_aux[np.logical_and(df_aux.index<end_date_dt,df_aux.index>=start_date_dt)]
                                                
                #Copy data to the output vecto olny if there not data there yet.
                Y.loc[Y.iloc[:,fase].isnull(),fase] = df_aux.loc[Y.iloc[:,fase].isnull()]
    
    
    #----------------------------------------------------------------#
    #Last operations before retun Y
    
    Y = Y.astype(float)    
    Y.columns = save_columns_name# Gives back the original name of the columns in X
    
    return Y

def IntegrateHour(x_in: pd.DataFrame,sample_freq: int = 5) -> pd.DataFrame:
    """

    :param Y: param sample_freq:  (Default value = 5)
    :param sample_freq:  (Default value = 5)

    """
    
    Y = x_in.copy(deep=True)
    
    time_vet_stamp = Y.index[np.arange(0,len(Y.index),int(60/sample_freq))]    
    Y = Y.groupby([Y.index.year,Y.index.month,Y.index.day,Y.index.hour]).mean() 
    Y = Y.reset_index(drop=True)
    Y.insert(0,'timestamp', time_vet_stamp)
    Y.set_index('timestamp', inplace=True)
    
    return Y

def Correlation(X: pd.DataFrame) -> float:
    """
    Calculates the correlation between each column of the DataFrame and outputs the average of all.
    

    """
    
    corr_value = X.corr()[X.corr()!=1].mean().mean()
    
    return corr_value
    
def DayPeriodMapper(hour: int) -> int:
    """
    Maps a given hour to one of four periods of a day.
    
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

def PhaseProportonInput(X,sample_freq = 5,threshold_accept = 0.75):
    """


    """
    
    Y = X.copy(deep=True)
    
    #Get number of samples that has at leat two phases to calculate unbalance
    number_of_possible_unbalace_calc = np.sum(np.sum(~X.isnull(),axis=1)>=2)
    
    if(number_of_possible_unbalace_calc>=X.shape[0]*0.3):#at least 30% of data
        
        #Make all the possible combinations between columns
        comb_vet = list(combinations(range(0,X.shape[1]),r=2))
        
        #make columns names
        comb_vet_str = []
        for comb in comb_vet:
            comb_vet_str.append(str(comb[0])+'-' +str(comb[1]))
        
        #Create output vector
        df_relation = pd.DataFrame(index=X.index,columns=comb_vet_str, dtype=object)        
        
        for i in range(0,len(comb_vet)):
            
            comb = comb_vet[i]
            comb_str = comb_vet_str[i]
            df_relation.loc[:,comb_str] = X.iloc[:,list(comb)].iloc[:,0]/X.iloc[:,list(comb)].iloc[:,1]
        
        df_relation.replace([np.inf, -np.inf], np.nan,inplace=True)
        
        #HOUR
        df_relation_aux = df_relation.groupby([df_relation.index.year,df_relation.index.month,df_relation.index.day,df_relation.index.hour]).mean()
        df_relation_mask = df_relation.groupby([df_relation.index.year,df_relation.index.month,df_relation.index.day,df_relation.index.hour]).count()/(60/sample_freq)        
        df_relation_mask = df_relation_mask>=threshold_accept        
        df_relation_aux = df_relation_aux[df_relation_mask]
            
        time_vet_stamp = df_relation.index[np.arange(0,len(df_relation.index),int(60/sample_freq))]            
        df_relation_aux = df_relation_aux.reset_index(drop=True)
        df_relation_aux.insert(0,'timestamp', time_vet_stamp)
        df_relation_aux.set_index('timestamp', inplace=True)
        
        
        #df_relation_test = df_relation.groupby([df_relation.index.year,df_relation.index.month,df_relation.index.day,df_relation.index.hour.map(DayPeriodMapper)]).mean()
    
    return Y

def ReturnOnlyValidDays(x_in: pd.DataFrame,
                        sample_freq: int = 5,
                        threshold_accept: float = 1.0,
                        sample_time_base: str = 'm') -> pd.DataFrame:
    """
    Returns all valid days. A valid day is one with no missing values for any 
    of the timeseries on each column.

    """
    
    X = x_in.copy(deep=True)
    
    qty_sample_dic = {'s':24*60*60,'m':24*60,'h':24}
    
    if sample_time_base in ['s','m','h']: 
        
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
    else:
        X = pd.DafaFrame([])
        df_count = pd.DafaFrame([])
        
        
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
    
    Y = DataSynchronization(Y, start_date_dt, end_date_dt,sample_freq = 1,sample_time_base='D')
    
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
    Y = DataSynchronization(X,start_date_dt,end_date_dt,sample_freq,sample_time_base='m')
    
    #Interpolate before proportion between phases
    if(pre_interpol!=False):
        Y = Y.interpolate(method_type='linear',limit=pre_interpol)
    
    #Interpolate after proportion between phases
    if(pos_interpol!=False):
        Y = Y.interpolate(method_type='linear',limit=pos_interpol)        
             
    #Integralization 1h
    if(integrate!=False):        
        Y = IntegrateHour(Y,sample_freq = 5)        
        
        #Interpolate after Integralization 1h
        if(interpol_integrate!=False):
            Y = Y.interpolate(method_type='linear',limit=interpol_integrate)                              
        
    
    return Y

def RemovePeriod(x_in: pd.DataFrame,df_remove: pd.DataFrame) -> pd.DataFrame:
    """
    Marks as nan all specified timestamps

    """
    
    Y = x_in.copy(deep=True)    
     
    for index,row in df_remove.iterrows():
        Y.loc[np.logical_and(Y.index>=row[0],Y.index<=row[1]),:] = np.nan        
        
    return Y

def SavePeriod(x_in: pd.DataFrame,df_save: pd.DataFrame) -> pd.DataFrame:    
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
        Y_int = IntegrateHour(Y,sample_freq)    
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

def CalcUnbalance(x_in: pd.DataFrame) -> pd.DataFrame:
    """
    Calculates the unbalance between phases for every timestamp.
    
    Equation:
        Y = (MAX-MEAN)/MEAN
    
    Ref.: Derating of induction motors operating with a combination of unbalanced voltages and over or undervoltages
    
    """
    
    Y = pd.DataFrame([],index=x_in.index)    
    
    Y['Unbalance'] = 100*(x_in.max(axis=1)-x_in.mean(axis=1))/x_in.mean(axis=1)
    
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
    print("Processado outliers...")

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
    print("Example:")    
    
    
    data_inicio='2021-01-01'
    data_final='2022-01-01'
    
    start_date_dt = dt.datetime(int(data_inicio.split("-")[0]),int(data_inicio.split("-")[1]),int(data_inicio.split("-")[2]))
    end_date_dt = dt.datetime(int(data_final.split("-")[0]),int(data_final.split("-")[1]),int(data_final.split("-")[2]))
 
        
    dummy = np.arange(start_date_dt, end_date_dt,np.timedelta64(5,'m'), dtype='datetime64')
    dummy = dummy + np.timedelta64(random.randint(0, 59),'s') # ADD a second to the end so during the sort this samples will be at last (HH:MM:01)   
        
    dummy = pd.DataFrame(dummy,columns=['timestamp'])
    
    dummy['VA'] = dummy['timestamp'].dt.hour
    dummy['VB'] = dummy['timestamp'].dt.minute*2
    dummy['VV'] = dummy['timestamp'].dt.minute*3
    dummy['VN'] = dummy['timestamp'].dt.minute*4
    
    #dummy['VA'] = 1
    #dummy['VB'] = 2
    #dummy['VV'] = 3
    #dummy['VN'] = 4
    
    
    dummy.set_index('timestamp', inplace=True)
    
    
    for col in ['VA','VB','VV']:
        dummy.loc[dummy.sample(frac=0.001).index, col] = np.nan
    
    
    time_init = time.perf_counter()    
    
    #TESTE
    dummy = pd.read_csv('SAFC2BSA.csv',names=['timestamp_aux','VA', 'VB', 'VV'],skiprows=1,parse_dates=True)
    dummy.insert(loc=0, column='timestamp', value=pd.to_datetime(dummy.timestamp_aux.astype(str)))
    dummy = dummy.drop(columns=['timestamp_aux'])
    dummy.set_index('timestamp', inplace=True)
    
    #TESTE MANOBRAS
    dummy_manobra = pd.read_csv('BancoManobras.csv',names=['EQ','ALIM1', 'ALIM2', 'data_inicio',"data_final"],skiprows=1,parse_dates=True)
    dummy_manobra = dummy_manobra.iloc[:,-2:]
    
    dummy_manobra = pd.DataFrame([[dt.datetime(2021,1,1),dt.datetime(2021,2,1)]])
    
    output = DataSynchronization(dummy,start_date_dt,end_date_dt,sample_freq= 5,sample_time_base='m')
       
    
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
    
    
    #output = GetWeekDayCurve(dummy,sample_freq = 5,threshold_accept = 1.0)
    
    #output = SimpleProcess(dummy,start_date_dt,end_date_dt,sample_freq = 5,pre_interpol=1,pos_interpol=1,integrate=True,interpol_integrate=1)
    #output = SimpleProcess(dummy,start_date_dt,end_date_dt,sample_freq = 5)
    
    print("Time spent: " + str(time.perf_counter()-time_init) )
    dummy.plot()
    output.plot()
    print(output)

    
    
    