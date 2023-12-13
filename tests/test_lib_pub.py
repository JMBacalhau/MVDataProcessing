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

df = mvp.CurrentDummyData()
df['IA'] = numpy.nan

_ = mvp.GetWeekDayCurve(df,sample_freq=5,threshold_accept=0.9,min_sample_per_day=3,min_sample_per_workday=9)

#-------------#
#    CLEAN    #
#-------------#


#-------------#
#     FILL    #
#-------------#


#-------------#
#   EXAMPLE   #
#-------------#
'''
#Util

SavePeriod


#Clean
RemoveOutliersMMADMM
RemoveOutliersHardThreshold
RemoveOutliersQuantile
RemoveOutliersHistogram

#Fill
PhaseProportionInput
SimpleProcess
GetNSSCPredictedSamples
ReplaceData
NSSCInput

#Example
ShowExampleSimpleProcess
ShowExampleNSSCProcess

'''
