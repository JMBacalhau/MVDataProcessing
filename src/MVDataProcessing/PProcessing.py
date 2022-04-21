# -*- coding: utf-8 -*-
"""
Created on Mon Apr 18 06:45:44 2022

@author: Bacalhau
"""

import pandas as pd
import datetime as dt
import numpy as np
from datetime import datetime

def DummyData(start_date=dt.datetime(2020,1,1), end_date=dt.datetime(2022,1,1),remove_data= 0.30,freq_min=5,delta_sec=39):
        
    dummy = np.arange(start_date, end_date,np.timedelta64(freq_min,'m'), dtype='datetime64')
    dummy = dummy + np.timedelta64(delta_sec,'s') # ADD a second to the end so during the sort this samples will be at last (HH:MM:01)   
        
    dummy = pd.DataFrame(dummy,columns=['timestamp'])
    
    dummy['VA'] = dummy['timestamp'].dt.day
    dummy['VB'] = dummy['timestamp'].dt.hour
    dummy['VV'] = dummy['timestamp'].dt.minute
    dummy['VN'] = dummy['timestamp'].dt.second
    
      
    for col in ['VA','VB','VV']:
        dummy.loc[dummy.sample(frac=remove_data).index, col] = np.nan

    return dummy

def DataClean(awnser,start_date_dt,end_date_dt,sample_freq = 5):    
    
    
    #----------------------------------------------------------------#
    #Creates a base vector that conntains all the samples between data_inicio and data_final filled timestamp and with nan
    
    qty_data = len(awnser.columns)-1
               
    timearray = np.arange(start_date_dt, end_date_dt,np.timedelta64(sample_freq,'m'), dtype='datetime64')
    timearray = timearray + np.timedelta64(1,'s') # ADD a second to the end so during the sort this samples will be at last (HH:MM:01)
    
    vet_amostras = pd.DataFrame(index=range(len(timearray)),columns=range(qty_data+1), dtype=object)
    vet_amostras[0] = timearray
    
    aux_name = list(np.arange(qty_data+1))    
    aux_name[0] = 'datahora'    
    vet_amostras.columns = aux_name    
        
   
    
    #----------------------------------------------------------------#
    #Creates the output dataframe which is the same but witohut the added second.
       
    Y = vet_amostras.copy(deep=True)    
    Y['datahora'] = Y['datahora'].dt.floor("T")#Flush the seconds    
        
    
    #----------------------------------------------------------------#
    #Start to process each column
    
    fase_list = np.arange(1,awnser.shape[1])   
    
    for fase in fase_list:                
        
        X = awnser.copy(deep=True)
        X.columns = aux_name
        X = X.loc[~X.iloc[:,fase].isnull(),['datahora',fase]]#Gets only samples on the phase of interest
        X = X.loc[np.logical_and(X.iloc[:,0]<end_date_dt,X.iloc[:,0]>=start_date_dt),:]#Get samples on between the start and end of the period of study
        
        if(X.shape[0]!=0):            
            
            #Process samples that are multiple of sample_freq
            df_X = X.copy(deep=True)
            df_vet_amostras = vet_amostras.loc[:,['datahora',fase]]            
                        
            #remove seconds (00:00:00) to put this specific samples at the beginning during sort
            df_X = df_X.sort_values(['datahora'],ascending=[1])#Ensures the sequence of timestamps
            df_X['datahora'] = df_X['datahora'].round('1min')#Remove seconds, rounding to the nearest minute
            df_X = df_X[df_X['datahora'].dt.minute % sample_freq == 0]#Samples that are multiple of sample_freq have preference 
            
            if(df_X.empty != True):
                            
                df_X = df_X.drop_duplicates(subset=['datahora'],keep= 'first')#Elimina duplicatas descenessárias              
               
                #Junta os dois Vetores e faz o sort
                df_aux = pd.concat([df_X, df_vet_amostras], ignore_index=True)
                df_aux = df_aux.sort_values(['datahora'],ascending=[1])
                
                #Elimina segundos (00:00:00), e elimina duplicatas deixando o X quando existente e vet_amostras quando não existe a amostra
                df_aux['datahora'] = df_aux['datahora'].dt.floor("T")
                df_aux.drop_duplicates(['datahora'],keep= 'first',inplace = True)
                df_aux.reset_index(inplace=True,drop=True)
                
                #Make sure that any round up that ended up out of the period of study is removed
                df_aux = df_aux.loc[np.logical_and(df_aux.iloc[:,0]<end_date_dt,df_aux.iloc[:,0]>=start_date_dt),:]
                
                Y.loc[:,fase] = df_aux.loc[:,fase]##Copy the results of the phase of interest to the Y (Output vector)                
                
                
                
            #Process samples that are NOT multiple of sample_freq
            df_X = X.copy(deep=True)
            df_vet_amostras = vet_amostras.loc[:,['datahora',fase]]                                 
                            
            #remove seconds (00:00:00) to put this specific samples at the beginning during sort
            df_X = df_X.sort_values(['datahora'],ascending=[1])#Ensures the sequence of timestamps
            df_X['datahora'] = df_X['datahora'].round('1min')#Remove seconds, rounding to the nearest minute            
            
            df_X = df_X[df_X['datahora'].dt.minute % sample_freq != 0]#Samples that are NOT multiple of sample_freq have preference 
                            
            
            if(df_X.empty != True):
               
                                               
                df_X.loc[:,'datahora'] = df_X.loc[:,'datahora'].round('5min')
                
                df_X = df_X.drop_duplicates(subset=['datahora'],keep= 'first')#Elimina duplicatas descenessárias              
               
                #Junta os dois Vetores e faz o sort
                df_aux = pd.concat([df_X, df_vet_amostras], ignore_index=True)
                df_aux = df_aux.sort_values(['datahora'],ascending=[1])
                
                #Elimina segundos (00:00:00), e elimina duplicatas deixando o X quando existente e vet_amostras quando não existe a amostra
                df_aux['datahora'] = df_aux['datahora'].dt.floor("T")
                df_aux.drop_duplicates(['datahora'],keep= 'first',inplace = True)
                df_aux.reset_index(inplace=True,drop=True)
                
                #Make sure that any round up that ended up out of the period of study is removed
                df_aux = df_aux.loc[np.logical_and(df_aux.iloc[:,0]<end_date_dt,df_aux.iloc[:,0]>=start_date_dt),:]
                
                #Subistitui no vetorr de saida apenas se na posição ainda não existir amostra caso contrário deixa a existente.
                Y.loc[Y.iloc[:,fase].isnull(),fase] = df_aux.loc[Y.iloc[:,fase].isnull(),fase]
    
    Y.loc[:,1:] = Y.loc[:,1:].astype(float)
      
    return Y