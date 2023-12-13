import MVDataProcessing as mvp
import time
import pandas
import numpy

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

I = mvp.CurrentDummyData()
V = mvp.VoltageDummyData()
pf = mvp.PowerFactorDummyData()

I = I.iloc[:, :-1]

dummy = pandas.DataFrame([])

dummy['S'] = V['VA'] / numpy.sqrt(3) * I['IA'] + V['VB'] / numpy.sqrt(3) * I['IB'] \
                                               + V['VV'] / numpy.sqrt(3) * I['IV']
dummy['P'] = V['VA'] / numpy.sqrt(3) * I['IA'] * pf['FPA'] + V['VB'] / numpy.sqrt(3) * I['IB'] * pf['FPB'] \
                                                           + V['VV'] / numpy.sqrt(3) * I['IV'] * pf['FPV']                                                      

dummy['Q'] = dummy['S'].pow(2) - dummy['P'].pow(2)
dummy['Q'] = numpy.sqrt(dummy['Q'].abs())



mvp.CountMissingData(dummy.interpolate())

'''
#Util

DataSynchronization
IntegrateHour
Correlation
DayPeriodMapper
DayPeriodMapperVet
YearPeriodMapperVet
CountMissingData
CalcUnbalance
SavePeriod
MarkNanPeriod
ReturnOnlyValidDays
GetDayMaxMin
GetWeekDayCurve



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
