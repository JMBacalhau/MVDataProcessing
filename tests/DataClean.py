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
    end_date_dt = datetime(2021,1,15)

    dummy = f_remove.CurrentDummyData()
    dummy.plot(title="Current Input (with outliers [A])")

    time_stopper = [['time_init', time.perf_counter()]]
    
    
    output = f_remove.DataSynchronization(dummy, start_date_dt, end_date_dt, sample_freq=5, sample_time_base='m')     
    f_remove.CountMissingData(output, show=True)
    time_stopper.append(['DataSynchronization', time.perf_counter()])
    output.plot(title="DataSynchronization")
    
    output.iloc[0:12*24*2,2] = numpy.nan
    
    output = f_remove.RemoveOutliersHardThreshold(output, hard_max=500, hard_min=0)
    f_remove.CountMissingData(output, show=True)    
    time_stopper.append(['RemoveOutliersHardThreshold', time.perf_counter()])
    output.plot(title="+RemoveOutliersHardThreshold")


    output = f_remove.RemoveOutliersMMADMM(output, len_mov_avg=3, std_def=4, plot=False, remove_from_process=['IN'])
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


    #DEBUG  NAO DEVERIA COLOCAR VALORES SE NÃO EXISTEM AS 3 FASES 
    #VER SE É BUG DO NEUTRO

    
    # -------------------------#
    #          HOUR           #
    # -------------------------#

    from itertools import permutations

    # make output vector
    X = output.copy(deep=True)
    Y = X.copy(deep=True)
    threshold_accept = 0.6

    mask_valid = ~X.isnull()
    grouper_valid = mask_valid.groupby(
        [mask_valid.index.year, mask_valid.index.month, mask_valid.index.day, mask_valid.index.hour])
    count_valid = grouper_valid.transform('sum')

    mask_null = X.isnull()
    grouper_null = mask_null.groupby(
        [mask_null.index.year, mask_null.index.month, mask_null.index.day, mask_null.index.hour])
    count_null = grouper_null.transform('sum')

    mask_reject = count_valid / (count_null + count_valid) < threshold_accept

    grouper = X.groupby([X.index.year, X.index.month, X.index.day, X.index.hour])
    X_mean = grouper.transform('mean')

    X_mean[mask_reject] = numpy.nan

    # Make all the possible permutations between columns
    comb_vet = list(permutations(range(0, X_mean.shape[1]), r=2))

    

    # make columns names
    comb_vet_str = []
    for comb in comb_vet:
        comb_vet_str.append(str(comb[0]) + '-' + str(comb[1]))

    # Create relation vector
    df_relation = pandas.DataFrame(index=X_mean.index, columns=comb_vet_str, dtype=object)

    corr_vet = []
    for i in range(0, len(comb_vet)):
        comb = comb_vet[i]
        comb_str = comb_vet_str[i]
        df_relation.loc[:, comb_str] = X_mean.iloc[:, list(comb)].iloc[:, 0] / X_mean.iloc[:, list(comb)].iloc[:, 1]

        corr = X_mean.iloc[:, list(comb)].iloc[:, 0].corr(X_mean.iloc[:, list(comb)].iloc[:, 1])
        corr_vet.append([str(comb[0]) + '-' + str(comb[1]), corr])

    corr_vet = pandas.DataFrame(corr_vet, columns=['comb', 'corr'])
    corr_vet.set_index('comb', drop=True, inplace=True)
    corr_vet.sort_values(by=['corr'], ascending=False, inplace=True)

    df_relation.replace([numpy.inf, -numpy.inf], numpy.nan, inplace=True)

    

    for i in range(0, len(comb_vet)):
        comb = comb_vet[i]
        comb_str = comb_vet_str[i]
        df_relation.loc[:, comb_str] = df_relation.loc[:, comb_str] * X.iloc[:, list(comb)[1]]

    

    for i in range(0, len(comb_vet)):
        comb = comb_vet[i]
        comb_str = comb_vet_str[i]
        Y.loc[
            (Y.iloc[:, list(comb)[0]].isnull()) & (~df_relation.loc[:, comb_str].isnull()),
            Y.columns[list(comb)[0]]] = df_relation.loc[(Y.iloc[:, list(comb)[0]].isnull()) &
                                                        (~df_relation.loc[:, comb_str].isnull()), comb_str]


    '''
    # Interpolate before and after with proportion between phases
    output = output.interpolate(method_type='linear', limit=12)    
    output.plot(title="+PhaseProportionInput (Per_interpol)")
    output = f_remove.PhaseProportionInput(output, threshold_accept=0.60)        
    output.plot(title="+PhaseProportionInput")
    output = output.interpolate(method_type='linear', limit=12)
    output.plot(title="+PhaseProportionInput (Per_interpol)")
    f_remove.CountMissingData(output, show=True)
    time_stopper.append(['PhaseProportionInput', time.perf_counter()])
    '''



    '''
    output[0:200000] = numpy.nan
    f_remove.CountMissingData(output, show=True)
    time_stopper.append(['Lost a Phase', time.perf_counter()])
    output.plot(title="+Data Lost")

    # Get day max/min values
    max_vet,_ = f_remove.GetDayMaxMin(output,start_date_dt,end_date_dt,sample_freq=5,threshold_accept=0.5,exe_param='max')         
    min_vet,_ = f_remove.GetDayMaxMin(output,start_date_dt,end_date_dt,sample_freq=5,threshold_accept=0.5,exe_param='min') 
    weekday_curve = f_remove.GetWeekDayCurve(output, sample_freq=5, threshold_accept=0.8,min_sample_per_day=3,min_sample_per_workday=9)

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