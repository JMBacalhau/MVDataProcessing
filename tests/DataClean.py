# -*- coding: utf-8 -*-
"""
Created on Mon Apr 18 06:45:44 2022

@author: Bacalhau
"""
"""
DataClean
Pré Interpolação
proporção de fases
pós interpolação

"""


import pandas as pd
import datetime as dt
import numpy as np
from datetime import datetime
from itertools import combinations
 
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', None)

def DataClean(X_Original,start_date_dt,end_date_dt,sample_freq = 5):    
    
    
    #----------------------------------------------------------------#
    #Creates a base vector that conntainXs all the samples between data_inicio and data_final filled timestamp and with nan
    
    qty_data = len(X_Original.columns)
               
    timearray = np.arange(start_date_dt, end_date_dt,np.timedelta64(sample_freq,'m'), dtype='datetime64')
    timearray = timearray + np.timedelta64(1,'s') # ADD a second to the end so during the sort this samples will be at last (HH:MM:01)
    
    vet_amostras = pd.DataFrame(index=timearray,columns=range(qty_data), dtype=object)
    vet_amostras.index.name = 'timestamp'
    
        
    #----------------------------------------------------------------#
    #Creates the output dataframe which is the same but witohut the added second.
       
    Y = vet_amostras.copy(deep=True)    
    Y.index = Y.index.floor("T")#Flush the seconds    
    
    #----------------------------------------------------------------#
    #Saves the name of the columns
    save_columns_name = X_Original.columns.values    
    
    #----------------------------------------------------------------#
    #Start to process each column
    
    fase_list = np.arange(0,X_Original.shape[1])   
    
    for fase in fase_list:                
        
        X = X_Original.copy(deep=True)
        X.columns = Y.columns
        X = X.loc[~X.iloc[:,fase].isnull(),fase]#Gets only samples on the phase of interest                
        X = X[np.logical_and(X.index<end_date_dt,X.index>=start_date_dt)]#Get samples on between the start and end of the period of study
        
        if(X.shape[0]!=0):            
            
            #Process samples that are multiple of sample_freq
            df_X = X.copy(deep=True)
            df_vet_amostras = vet_amostras[fase] 
                        
            #remove seconds (00:00:00) to put this specific samples at the beginning during sort
            df_X = df_X.sort_index(ascending=True)#Ensures the sequence of timestamps
            df_X.index = df_X.index.round('1min')#Remove seconds, rounding to the nearest minute
            df_X = df_X[df_X.index.minute % sample_freq == 0]#Samples that are multiple of sample_freq have preference 
            
            if(df_X.empty != True):
                              
                df_X = df_X[~df_X.index.duplicated(keep='first')]#Remove unecessary duplicates     
               
                #joins both vectors
                df_aux = pd.concat([df_X, df_vet_amostras])
                df_aux = df_aux.sort_index(ascending=True)#Ensures the sequence of timestamps    
                
                #Elimina segundos (00:00:00), e elimina duplicatas deixando o X quando existente e vet_amostras quando não existe a amostra
                df_aux.index = df_aux.index.floor("T")
                df_aux = df_aux[~df_aux.index.duplicated(keep='first')]#Remove unecessary duplicates     
                
                #Make sure that any round up that ended up out of the period of study is removed
                df_aux = df_aux[np.logical_and(df_aux.index<end_date_dt,df_aux.index>=start_date_dt)]
                
                Y.loc[:,fase] = df_aux
                
                
                
            #Process samples that are NOT multiple of sample_freq
            df_X = X.copy(deep=True)
            df_vet_amostras = vet_amostras[fase]                               
                            
            #remove seconds (00:00:00) to put this specific samples at the beginning during sort
            df_X = df_X.sort_index(ascending=True)#Ensures the sequence of timestamps
            df_X.index = df_X.index.round('1min')#Remove seconds, rounding to the nearest minute
            df_X = df_X[df_X.index.minute % sample_freq != 0]#Samples that are NOT multiple of sample_freq have preference 
                            
            
            if(df_X.empty != True):
               
                                               
                df_X.index = df_X.index.round(str(sample_freq)+'min')#Aproximate sample to the closest multiple of sample_freq
                
                df_X = df_X[~df_X.index.duplicated(keep='first')]#Remove unecessary duplicates     
               
                #joins both vectors
                df_aux = pd.concat([df_X, df_vet_amostras])
                df_aux = df_aux.sort_index(ascending=True)#Ensures the sequence of timestamps                
                
                #Remove seconds (00:00:00), and remove ducplicates leaving X when there is data and vet amostra when its empty
                df_aux.index = df_aux.index.floor("T")
                df_aux = df_aux[~df_aux.index.duplicated(keep='first')]#Remove unecessary duplicates     
                
                
                #Make sure that any round up that ended up out of the period of study is removed
                df_aux = df_aux[np.logical_and(df_aux.index<end_date_dt,df_aux.index>=start_date_dt)]
                                                
                #Copy data to the output vecto olny if there not data there yet.
                Y.loc[Y.iloc[:,fase].isnull(),fase] = df_aux.loc[Y.iloc[:,fase].isnull()]
    
    
    #----------------------------------------------------------------#
    #Last operations before retun Y
    
    Y = Y.astype(float)    
    Y.columns = save_columns_name# Gives back the original name of the columns of X
    
    return Y


def IntegrateHour(Y,sample_freq = 5):
    
    time_vet_stamp = Y.index[np.arange(0,len(Y.index),int(60/sample_freq))]    
    Y = Y.groupby([Y.index.year,Y.index.month,Y.index.day,Y.index.hour]).mean() 
    Y = Y.reset_index(drop=True)
    Y.insert(0,'timestamp', time_vet_stamp)
    Y.set_index('timestamp', inplace=True)
    
    return Y


def Correlation(X):
    
    corr_value = X.corr()[X.corr()!=1].mean().mean()
    
    return corr_value
    
def DayPeriodMapper(hour):
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

def GetWeekDayCurve(X,sample_freq = 5,threshold_accept = 1.0):
    
    # X = output.copy(deep=True)
    
    df_count = X.groupby([X.index.year,X.index.month,X.index.day]).count()/(24*60/sample_freq)    
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
    
    
    grouper = X.groupby([X.index.weekday,X.index.hour])
    
    maxes = grouper.transform('max')
    mins = grouper.transform('min')
    
    X = (X-mins)/(maxes-mins)
    
    X.groupby([X.index.weekday,X.index.hour])

    
    return 0

    
    
def SimpleProcess(X,start_date_dt,end_date_dt,sample_freq = 5,pre_interpol=False,pos_interpol=False,prop_phases=False,integrate=False,interpol_integrate=False):    
        
    #ORGANIZE->INTERPOLATE->PHASE_PROPORTION->INTERPOLATE->INTEGRATE->INTERPOLATE
    
    #Organize samples
    Y = DataClean(X,start_date_dt,end_date_dt,sample_freq)
    
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

if __name__ == "__main__":
    
    import time
    import random
    print("Example:")    
    
    
    data_inicio='2020-01-01'
    data_final='2020-01-02'
    
    start_date_dt = dt.datetime(int(data_inicio.split("-")[0]),int(data_inicio.split("-")[1]),int(data_inicio.split("-")[2]))
    end_date_dt = dt.datetime(int(data_final.split("-")[0]),int(data_final.split("-")[1]),int(data_final.split("-")[2]))
 
        
    dummy = np.arange(start_date_dt, end_date_dt,np.timedelta64(5,'m'), dtype='datetime64')
    dummy = dummy + np.timedelta64(random.randint(0, 59),'s') # ADD a second to the end so during the sort this samples will be at last (HH:MM:01)   
        
    dummy = pd.DataFrame(dummy,columns=['timestamp'])
    
    dummy['VA'] = dummy['timestamp'].dt.minute
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
    #dummy = pd.read_csv('SAFC2BSA.csv',names=['timestamp_aux','VA', 'VB', 'VV'],skiprows=1,parse_dates=True)
    #dummy.insert(loc=0, column='timestamp', value=pd.to_datetime(dummy.timestamp_aux.astype(str)))
    #dummy = dummy.drop(columns=['timestamp_aux'])
    #dummy.set_index('timestamp', inplace=True)
    
    #output = SimpleProcess(dummy,start_date_dt,end_date_dt,sample_freq = 5,pre_interpol=1,pos_interpol=1,integrate=True,interpol_integrate=1)
    output = SimpleProcess(dummy,start_date_dt,end_date_dt,sample_freq = 5)
    
    print("Time spent: " + str(time.perf_counter()-time_init) )
    
    print(output)

    
    
    