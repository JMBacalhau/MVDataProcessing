import MVDataProcessing as mvp
import time
import pandas
import numpy
from datetime import datetime

#-------------#
#     UTIL    #
#-------------#

#TimeProfile
time_stopper = []
time_stopper.append(['init',time.perf_counter()])
time.sleep(.1)
time_stopper.append(['1',time.perf_counter()])
time.sleep(.1)
time_stopper.append(['2',time.perf_counter()])
time.sleep(.1)
time_stopper.append(['3',time.perf_counter()])
mvp.TimeProfile(time_stopper,'test',show=True,estimate_for=1000)

#CurrentDummyData
df = mvp.CurrentDummyData()
print(df)

#VoltageDummyData
df = mvp.VoltageDummyData()
print(df)

#PowerFactorDummyData
df = mvp.PowerFactorDummyData()
print(df)

#PowerDummyData
df = mvp.PowerDummyData()
print(df)

#EnergyDummyData
df = mvp.EnergyDummyData()
print(df)

#DataSynchronization
df = mvp.DataSynchronization(mvp.CurrentDummyData(),datetime(2023,1,1),datetime(2023,6,1),sample_freq=5,sample_time_base='m')
print(df)

df = mvp.DataSynchronization(mvp.CurrentDummyData(),datetime(2023,1,1),datetime(2023,6,1),sample_freq=10,sample_time_base='m')
print(df)

df = mvp.DataSynchronization(mvp.CurrentDummyData(),datetime(2023,1,1),datetime(2023,6,1),sample_freq=1,sample_time_base='h')
print(df)

df = mvp.DataSynchronization(mvp.CurrentDummyData(),datetime(2023,1,1),datetime(2023,6,1),sample_freq=1,sample_time_base='D')
print(df)

#IntegrateHour
df = mvp.IntegrateHour(mvp.CurrentDummyData(),sample_freq=5,sample_time_base='m')
print(df)

#Correlation
_ = mvp.Correlation(mvp.CurrentDummyData())
print(_)

#DayPeriodMapper
_ = mvp.DayPeriodMapper(5)
print(_)

_ = mvp.DayPeriodMapper(30)
print(_)

#DayPeriodMapperVet
_ = mvp.DayPeriodMapperVet(pandas.Series([0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23]))
print(_)

#YearPeriodMapperVet
_ = mvp.YearPeriodMapperVet(pandas.Series([0,1,2,3,4,5,6,7,8,9,10,11,12]))
print(_)


#CountMissingData
_ = mvp.CountMissingData(mvp.CurrentDummyData(),show=True)

df = mvp.CurrentDummyData()
df['IA'] = numpy.nan
_ = mvp.CountMissingData(df,remove_from_process=['IA'],show=True)
_ = mvp.CountMissingData(df,show=True)

#CalcUnbalance
_ = mvp.CalcUnbalance(mvp.CurrentDummyData())
print(_)

df = mvp.CurrentDummyData()
df['IA'] = numpy.nan
_ = mvp.CalcUnbalance(df,remove_from_process=['IA'])
print(_)


#SavePeriod
df_save = pandas.DataFrame([[datetime(2023,1,1),datetime(2023,1,7)],
                            [datetime(2023,2,1),datetime(2023,2,7)],
                            [datetime(2023,1,1),datetime(2023,5,7)]])

_,index_ =  mvp.SavePeriod(mvp.CurrentDummyData(), df_save)
print(_)
print(index_)


#MarkNanPeriod
df_remove_week = pandas.DataFrame([[datetime(2023,1,1),datetime(2023,1,7)],
                                 [datetime(2023,2,1),datetime(2023,2,7)],
                                 [datetime(2023,3,1),datetime(2023,3,7)]])
df = mvp.MarkNanPeriod(mvp.CurrentDummyData(),df_remove_week)
print(df)

df = mvp.MarkNanPeriod(mvp.CurrentDummyData(),df_remove_week,remove_from_process=['IA'])
print(df)


#ReturnOnlyValidDays
_ = mvp.ReturnOnlyValidDays(mvp.CurrentDummyData(),sample_freq=5,sample_time_base='m',threshold_accept=0.9)
print(_)

df = mvp.MarkNanPeriod(mvp.CurrentDummyData(),df_remove_week)
_ = mvp.ReturnOnlyValidDays(df,sample_freq=5,sample_time_base='m',threshold_accept=0.9)
print(_)


#GetDayMaxMin
_ , index_ = mvp.GetDayMaxMin(mvp.CurrentDummyData(),datetime(2023,1,1),datetime(2023,6,1),sample_freq=5,threshold_accept=0.9,exe_param='max')
print(_)
print(index_)
_ , index_ = mvp.GetDayMaxMin(mvp.CurrentDummyData(),datetime(2023,1,1),datetime(2023,6,1),sample_freq=5,threshold_accept=0.9,exe_param='min')
print(_)
print(index_)


#GetWeekDayCurve
_ = mvp.GetWeekDayCurve(mvp.CurrentDummyData(),sample_freq=5,threshold_accept=0.9,min_sample_per_day=3,min_sample_per_workday=9)
print(_)

df = mvp.CurrentDummyData()
df['IA'] = numpy.nan

_ = mvp.GetWeekDayCurve(df,sample_freq=5,threshold_accept=0.9,min_sample_per_day=3,min_sample_per_workday=9)
print(_)



#-------------#
#    CLEAN    #
#-------------#



df_avoid = pandas.DataFrame([[datetime(2023,1,1),datetime(2023,6,1)]])

#RemoveOutliersMMADMM
#_  = mvp.RemoveOutliersMMADMM(mvp.CurrentDummyData(),len_mov_avg=20,std_def=5,min_var_def=3,allow_negatives=False,plot=True)
_  = mvp.RemoveOutliersMMADMM(mvp.CurrentDummyData(),len_mov_avg=20,std_def=5,min_var_def=3,allow_negatives=False,plot=False)

#_  = mvp.RemoveOutliersMMADMM(mvp.CurrentDummyData(),len_mov_avg=20,std_def=5,min_var_def=3,allow_negatives=True,plot=True)
_  = mvp.RemoveOutliersMMADMM(mvp.CurrentDummyData(),len_mov_avg=20,std_def=5,min_var_def=3,allow_negatives=True,plot=False)

#_  = mvp.RemoveOutliersMMADMM(mvp.CurrentDummyData(),len_mov_avg=20,std_def=5,min_var_def=3,allow_negatives=False,plot=True,remove_from_process=['IA']))
_  = mvp.RemoveOutliersMMADMM(mvp.CurrentDummyData(),len_mov_avg=20,std_def=5,min_var_def=3,allow_negatives=False,plot=False,remove_from_process=['IA'])


#_  = mvp.RemoveOutliersMMADMM(mvp.CurrentDummyData(),len_mov_avg=20,std_def=5,min_var_def=3,allow_negatives=False,plot=True,remove_from_process=['IA'],df_avoid_periods=df_avoid)
_  = mvp.RemoveOutliersMMADMM(mvp.CurrentDummyData(),len_mov_avg=20,std_def=5,
                              min_var_def=3,allow_negatives=False,plot=False,remove_from_process=['IA'],df_avoid_periods=df_avoid)
#_.plot()

#_  = mvp.RemoveOutliersMMADMM(mvp.CurrentDummyData(),len_mov_avg=20,std_def=5,min_var_def=3,allow_negatives=False,plot=True,df_avoid_periods=df_avoid)
_  = mvp.RemoveOutliersMMADMM(mvp.CurrentDummyData(),len_mov_avg=20,std_def=5,
                              min_var_def=3,allow_negatives=False,plot=False,df_avoid_periods=df_avoid)
#_.plot()


#RemoveOutliersHardThreshold
_  = mvp.RemoveOutliersHardThreshold(mvp.CurrentDummyData(),df_avoid_periods=df_avoid,hard_max=250,hard_min=0)
#_.plot()

_  = mvp.RemoveOutliersHardThreshold(mvp.CurrentDummyData(),hard_max=250,hard_min=0)
#_.plot()


#RemoveOutliersQuantile
_ = mvp.RemoveOutliersQuantile(mvp.CurrentDummyData())
#_.plot()

_ = mvp.RemoveOutliersQuantile(mvp.CurrentDummyData(),remove_from_process=['IA'])
#_.plot()

_ = mvp.RemoveOutliersQuantile(mvp.CurrentDummyData(),remove_from_process=['IA'],df_avoid_periods=df_avoid)
#_.plot()


#RemoveOutliersHistogram
_ = mvp.RemoveOutliersHistogram(mvp.CurrentDummyData(),df_avoid_periods=df_avoid,remove_from_process=['IA'],min_number_of_samples_limit=20)
#_.plot()

_ = mvp.RemoveOutliersHistogram(mvp.CurrentDummyData(),df_avoid_periods=df_avoid,min_number_of_samples_limit=20)
#_.plot()



#-------------#
#     FILL    #
#-------------#



#PhaseProportionInput
df = mvp.CurrentDummyData()
df.iloc[0:50000,1] = numpy.nan
df.iloc[60000:75000,2] = numpy.nan
df.iloc[60000:75000,1] = numpy.nan
df.iloc[75000:90000,:] = numpy.nan
#df.plot()

_ = mvp.PhaseProportionInput(df,threshold_accept=0.6,plot=False,apply_filter=True)
#_.plot()


_ = mvp.SimpleProcess(df,datetime(2023,1,1),datetime(2024,1,1),pre_interpol=1,pos_interpol=1,prop_phases=True)
#_.plot()

_ = mvp.SimpleProcess(df,datetime(2023,1,1),datetime(2024,1,1),pre_interpol=1,pos_interpol=1,prop_phases=True,interpol_integrate=5)
#_.plot()

#GetNSSCPredictedSamples

#GetDayMaxMin
vet_max,_ = mvp.GetDayMaxMin(df,datetime(2023,1,1),datetime(2024,1,1),sample_freq=5,threshold_accept=0.9,exe_param='max')
vet_min,_ = mvp.GetDayMaxMin(df,datetime(2023,1,1),datetime(2024,1,1),sample_freq=5,threshold_accept=0.9,exe_param='min')

weekday_curve = mvp.GetWeekDayCurve(df,sample_freq=5,threshold_accept=0.9,min_sample_per_day=3,min_sample_per_workday=9)

X_pred = mvp.GetNSSCPredictedSamples(vet_max,vet_min,weekday_curve,datetime(2023,1,1),datetime(2024,1,1),)


#ReplaceData

df = mvp.ReplaceData(df,X_pred,datetime(2023,1,1),datetime(2024,1,1), sample_freq=5,sample_time_base='m')




#-------------#
#   EXAMPLE   #
#-------------#
'''

#Fill

ReplaceData
NSSCInput

#Example
ShowExampleSimpleProcess
ShowExampleNSSCProcess

'''
