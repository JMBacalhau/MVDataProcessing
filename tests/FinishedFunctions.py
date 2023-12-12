# -*- coding: utf-8 -*-
"""
Created on Sun May 22 20:48:11 2022

@author: Bacalhau
"""

import pandas
import numpy
import datetime
import time
from datetime import datetime
import datetime as dt
from itertools import permutations
import random
import matplotlib.pyplot

def TimeProfile(time_stopper: list, name: str = '', show: bool = False, estimate_for: int = 0):
    """
    Simple code profiler.

    How to use:

    Create a list ->  time_stopper = []

    Put a -> time_stopper.append(['time_init',time.perf_counter()]) at the beginning.

    Put time_stopper.append(['Func_01',time.perf_counter()]) after the code block with the first parameter being
    a name and the second being the time.

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


    :param time_stopper: A List that will hold all the stop times.
    :type time_stopper: list

    :param name: A name for this instance of time profile. Defaults to empty.
    :type name: str, optional

    :param show: If True shows the data on the console. Defaults to False.
    :type show: bool, optional

    :param estimate_for: A multiplier to be applied at the end. Takes the whole
    time analyzed and multiplies by "estimate_for".
    :type estimate_for: int

    :return: None
    :rtype: None

    """

    if(show):
        print("Profile: " + name)
        time_stopper = pandas.DataFrame(time_stopper, columns=['Type', 'time'])
        # time_stopper['time'] = time_stopper['time']-time_stopper['time'].min()
        time_stopper['Delta'] = time_stopper['time'] - time_stopper['time'].shift(periods=1, fill_value=0)
        time_stopper = time_stopper.iloc[1:, :]
        time_stopper['%'] = numpy.round(100 * time_stopper['Delta'] / time_stopper['Delta'].sum(), 2)
        total_estimate = time_stopper['Delta'].sum()
        time_stopper = pandas.concat((time_stopper,
                                      pandas.DataFrame([['Total', numpy.nan, time_stopper['Delta'].sum(), 100]],
                                                       columns=['Type', 'time', 'Delta', '%'])))
        print(time_stopper)
        if estimate_for != 0:
            print(
                f"Estimation for {estimate_for} "
                f"runs: {numpy.round(total_estimate * estimate_for / (60 * 60), 2)} hours.")

    return


# BUG Some sample_freq have trouble lol.
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

    Most of the functions in this module assumes that the time series are "Clean" to a certain sample_freq. Therefore,
    this function must be executed first on the dataset.


    :param x_in: A pandas.core.frame.DataFrame where the index is of type "pandas.core.indexes.datetime.DatetimeIndex"
    and each column contain an electrical quantity time series.
    :type x_in: pandas.core.frame.DataFrame

    :param start_date_dt: The start date where the synchronization should start.
    :type start_date_dt: datetime

    :param end_date_dt: The end date where the synchronization will consider samples.
    :type end_date_dt: datetime

    :param sample_freq: The sample frequency of the time series. Defaults to 5.
    :type sample_freq: int,optional

    :param sample_time_base: The base time of the sample frequency. Specify if the sample frequency is in (D)ay,
    (M)onth, (Y)ear, (h)ours, (m)inutes, or (s)econds. Defaults to (m)inutes.
    :type sample_time_base: srt,optional


    :raises Exception: if x_in has no DatetimeIndex.
    :raises Exception: if start_date_dt not in datetime format.
    :raises Exception: if end_date_dt not in datetime format.
    :raises Exception: if sample_time_base is not in (D)ay, (M)onth, (Y)ear, (h)ours, (m)inutes, or (s)econds.


    :return: Y: The synchronized pandas.core.frame.DataFrame
    :rtype: Y: pandas.core.frame.DataFrame

    """

    #  BASIC INPUT CHECK

    if not (isinstance(x_in.index, pandas.DatetimeIndex)):
        raise Exception("x_in DataFrame has no DatetimeIndex.")
    if not (isinstance(start_date_dt, datetime)):
        raise Exception("start_date_dt Date not in datetime format.")
    if not (isinstance(end_date_dt, datetime)):
        raise Exception("end_date_dt Date not in datetime format.")
    if sample_time_base not in ['s', 'm', 'h', 'D', 'M', 'Y']:
        raise Exception("sample_time_base not valid. Ex. ['s','m','h','D','M','Y'] ")

    added_dic = {'s': 'ms', 'm': 's', 'h': 'm', 'D': 'h', 'M': 'D', 'Y': 'M'}
    floor_dic = {'s': 'S', 'm': 'T', 'h': 'H', 'D': 'D', 'M': 'M', 'Y': 'Y'}

    x_in.index = x_in.index.tz_localize(None)  # Makes the datetimeIndex naive (no time zone)

    '''
    Creates a base vector that contains all the samples 
    between start_date_dt and end_date_dt filled timestamp and with nan
    '''

    qty_data = len(x_in.columns)

    time_array = numpy.arange(start_date_dt, end_date_dt, numpy.timedelta64(sample_freq, sample_time_base),
                              dtype='datetime64')
    time_array = time_array + numpy.timedelta64(1, added_dic[
        sample_time_base])  # ADD a second/Minute/Hour/Day/Month to the end so during the sort
    # this samples will be at last (HH:MM:01)

    vet_samples = pandas.DataFrame(index=time_array, columns=range(qty_data), dtype=object)
    vet_samples.index.name = 'timestamp'

    # Creates the output dataframe which is the same but without the added second.

    df_y = vet_samples.copy(deep=True)
    df_y.index = df_y.index.floor(floor_dic[sample_time_base])  # Flush the seconds

    # Saves the name of the columns
    save_columns_name = x_in.columns.values

    # Start to process each column

    phase_list = numpy.arange(0, x_in.shape[1])

    for phase in phase_list:

        x = x_in.copy(deep=True)
        x.columns = df_y.columns
        x = x.loc[~x.iloc[:, phase].isnull(), phase]  # Gets only samples on the phase of interest
        x = x[numpy.logical_and(x.index < end_date_dt,
                                x.index >= start_date_dt)]

        if x.shape[0] != 0:

            # Process samples that are multiple of sample_freq
            df_x = x.copy(deep=True)
            df_vet_samples = vet_samples[phase]

            # remove seconds (00:00:00) to put this specific samples at the beginning during sort
            df_x = df_x.sort_index(ascending=True)  # Ensures the sequence of timestamps
            df_x.index = df_x.index.round(
                '1' + floor_dic[sample_time_base])  # Remove seconds, rounding to the nearest minute
            df_x = df_x[
                df_x.index.minute % sample_freq == 0]  # Samples that are multiple of sample_freq have preference

            if not df_x.empty:
                df_x = df_x[~df_x.index.duplicated(keep='first')]  # Remove unnecessary duplicates

                # joins both vectors
                df_aux = pandas.concat([df_x, df_vet_samples])
                df_aux = df_aux.sort_index(ascending=True)  # Ensures the sequence of timestamps

                '''
                Remove sec. (00:00:00), and remove duplicates leaving X when there is data 
                and vet amostra where its empty
                '''
                df_aux.index = df_aux.index.floor(floor_dic[sample_time_base])
                df_aux = df_aux[~df_aux.index.duplicated(keep='first')]  # Remove unnecessary duplicates

                # Make sure that any round up that ended up out of the period of study is removed
                df_aux = df_aux[numpy.logical_and(df_aux.index < end_date_dt, df_aux.index >= start_date_dt)]

                df_y.loc[:, phase] = df_aux

            # Process samples that are NOT multiple of sample_freq
            df_x = x.copy(deep=True)
            df_vet_samples = vet_samples[phase]

            # remove seconds (00:00:00) to put this specific samples at the beginning during sort
            df_x = df_x.sort_index(ascending=True)  # Ensures the sequence of timestamps
            df_x.index = df_x.index.round(
                '1' + floor_dic[sample_time_base])  # Remove seconds, rounding to the nearest minute
            df_x = df_x[
                df_x.index.minute % sample_freq != 0]  # Samples that are NOT multiple of sample_freq have preference

            if not df_x.empty:
                df_x.index = df_x.index.round(str(sample_freq) + floor_dic[
                    sample_time_base])  # Approximate sample to the closest multiple of sample_freq

                df_x = df_x[~df_x.index.duplicated(keep='first')]  # Remove unnecessary duplicates

                # joins both vectors
                df_aux = pandas.concat([df_x, df_vet_samples])
                df_aux = df_aux.sort_index(ascending=True)  # Ensures the sequence of timestamps

                '''
                Remove sec. (00:00:00), and remove duplicates leaving X when there is data 
                and vet amostra where its empty
                '''
                df_aux.index = df_aux.index.floor(floor_dic[sample_time_base])
                df_aux = df_aux[~df_aux.index.duplicated(keep='first')]  # Remove unnecessary duplicates

                # Make sure that any round up that ended up out of the period of study is removed
                df_aux = df_aux[numpy.logical_and(df_aux.index < end_date_dt, df_aux.index >= start_date_dt)]

                # Copy data to the output vector only if there is no data there yet.
                df_y.loc[df_y.iloc[:, phase].isnull(), phase] = df_aux.loc[df_y.iloc[:, phase].isnull()]

    # Last operations before the return of Y

    df_y = df_y.astype(float)
    df_y.columns = save_columns_name  # Gives back the original name of the columns in x_in

    return df_y


def IntegrateHour(x_in: pandas.DataFrame, sample_freq: int = 5,
                  sample_time_base: str = 'm') -> pandas.core.frame.DataFrame:
    """
    Integrates the input pandas.core.frame.DataFrame to an hour samples.

    :param x_in: A pandas.core.frame.DataFrame where the index is of type "pandas.core.indexes.datetimes.DatetimeIndex"
    and each column contain an electrical quantity time series.
    :type x_in: pandas.core.frame.DataFrame

    :param sample_freq: The sample frequency of the time series. Defaults to 5.
    :type sample_freq: int,optional

    :param sample_time_base: The base time of the sample frequency. Specify if the sample frequency is in (m)inutes
    or (s)econds. Defaults to (m)inutes.
    :type sample_time_base: srt,optional


    :raises Exception: if x_in has no DatetimeIndex.


    :return: df_y: The pandas.core.frame.DataFrame integrated by hour.
    :rtype: df_y: pandas.core.frame.DataFrame

    """
    hour_divider = {'s': 60 * 60, 'm': 60}

    # -------------------#
    # BASIC INPUT CHECK #
    # -------------------#

    if not (isinstance(x_in.index, pandas.DatetimeIndex)):
        raise Exception("x_in DataFrame has no DatetimeIndex.")

    df_y = x_in.copy(deep=True)

    time_vet_stamp = df_y.index[numpy.arange(0, len(df_y.index), int(hour_divider[sample_time_base] / sample_freq))]
    df_y = df_y.groupby([df_y.index.year, df_y.index.month, df_y.index.day, df_y.index.hour]).mean()
    df_y = df_y.reset_index(drop=True)
    df_y.insert(0, 'timestamp', time_vet_stamp)
    df_y.set_index('timestamp', inplace=True)

    return df_y


def Correlation(x_in: pandas.DataFrame) -> float:
    """
    Calculates the correlation between each column of the DataFrame and outputs the average of all.


    :param x_in: A pandas.core.frame.DataFrame where the index is of type "pandas.core.indexes.datetime.DatetimeIndex"
    and each column contain an electrical quantity time series.
    :type x_in: pandas.core.frame.DataFrame


    :return: corr_value: Value of the correlation
    :rtype: corr_value: float

    """

    corr_value = x_in.corr()[x_in.corr() != 1].mean().mean()

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


def DayPeriodMapperVet(hour: pandas.core.series.Series) -> pandas.core.series.Series:
    """
    Maps a given hour to one of four periods of a day.

    For 0 to 5 (hour) -> 0 night
    For 6 to 11 (hour) -> 1 moorning
    For 12 to 17 (hour) -> 2 afternoon
    For 18 to 23 (hour) -> 3 evening


    :param hour: A pandas.core.series.Series with values between 0 and 23 to map each hour in the series to a period
    of the day. this is a "vector" format for DayPeriodMapper function.
    :type hour: pandas.core.series.Series

    :return: period_day: The hour pandas.core.series.Series mapped to periods of the day
    :rtype: period_day: pandas.core.series.Series

    """

    map_dict = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0, 5: 0,
                6: 1, 7: 1, 8: 1, 9: 1, 10: 1, 11: 1,
                12: 2, 13: 2, 14: 2, 15: 2, 16: 2, 17: 2,
                18: 3, 19: 3, 20: 3, 21: 3, 22: 3, 23: 3}

    period_day = hour.map(map_dict)

    return period_day


def YearPeriodMapperVet(month: pandas.core.series.Series) -> pandas.core.series.Series:
    """
    Maps a given month to one of two periods of a year, being dry and humid .

    For october to march (month) -> 0 humid
    For april to september (month) -> 1 dry


    :param month: A pandas.core.series.Series with values between 0 and 12 to map each month
    in the series to dry or humid.

    :return: season: The months pandas.core.series.Series mapped to dry or humid.
    :rtype: season: pandas.core.series.Series

    """

    map_dict = {10: 0, 11: 0, 12: 0, 1: 0, 2: 0, 3: 0,
                4: 1, 5: 1, 6: 1, 7: 1, 8: 1, 9: 1}

    season = month.map(map_dict)

    return season


def PhaseProportionInput(x_in: pandas.core.frame.DataFrame,
                         threshold_accept: float = 0.75,
                         plot: bool = False,
                         apply_filter: bool = True, 
                         time_frame_apply:list = ['h','pd','D','M','S','Y','A'],
                         remove_from_process: list = []) -> pandas.core.frame.DataFrame:
   
    """
    Processes input DataFrame to compute phase proportion based on various time frames and criteria.

    Makes the imputation of missing data samples based on the ration between columns. (time series)

    Theory background.:

    Correlation between phases (φa,φb, φv) of the same quantity (V, I or pf) is used to infer a missing sample value
    based on adjacent    samples. Adjacent samples are those of the same timestamp i but from different phases that
    the one which is missing.    The main idea is to use a period where all three-phases (φa, φb, φv) exist and
    calculate the proportion between them. Having the relationship between phases, if one or two are missing
    in a given timestamp i it is possible to use the    remaining phase and the previous calculated ratio to
    fill the missing ones. The number of samples used to calculate the ratio around the missing sample is an
    important parameter. For instance if a sample is missing in the afternoon it is best to use samples from
    that same day and afternoon to calculate the ratio and fill the missing sample. Unfortunately, there might be not
    enough samples in that period to calculate the ratio.Therefore, in this step, different periods T of analysis
     around the missing sample reconsidered: hour, period of the day (dawn, morning, afternoon and night),
     day, month, season (humid/dry), and year.


    The correlation between the feeder energy demand and the period of the day or the season is very high.
    The increase in consumption in the morning and afternoon in industrial areas is expected as those are
    the periods where most factories are fully functioning. In residential areas, the consumption is expected
    to be higher in the evening; however, it is lower during the day’s early hours. Furthermore, in the summer,
    a portion of the network (vacation destination) can be in higher demand. Nonetheless, in another period of
    the year (winter), the same area could have a lower energy demand. Therefore, if there is not enough information
    on that particular day to compute the ratio between phases, a good alternative is to use data from the month.
    Finally, given the amount of missing data for a particular feeder, the only option could be the use of the
    whole year to calculate the ratio between phases. Regarding the minimum amount of data that a period
    should have to be valid it is assumed the default of 50% for all phases.

    :param x_in: Input DataFrame with a DatetimeIndex.
    :type x_in: pandas.core.frame.DataFrame
    :param threshold_accept: Threshold for accepting data based on null value proportion, defaults to 0.75.
    :type threshold_accept: float, optional
    :param plot: Flag to indicate if plots should be generated, defaults to False.
    :type plot: bool, optional
    :param apply_filter: Flag to indicate if outlier filter should be applied, defaults to True.
    :type apply_filter: bool, optional
    :param time_frame_apply: List of time frames to apply phase proportion analysis, defaults to ['h','pd','D','M','S','Y','A'].
    :type time_frame_apply: list, optional
    :param remove_from_process: List of columns to exclude from processing, defaults to an empty list.
    :type remove_from_process: list, optional
    :return: DataFrame with phase proportions computed and applied.
    :rtype: pandas.core.frame.DataFrame

    :raises Exception: If input DataFrame does not have a DatetimeIndex.
    :raises Exception: If input DataFrame has less than two columns.
    :raises Exception: If no time frames are provided in `time_frame_apply`.

    The function applies various transformations and calculations based on specified time frames, 
    handling missing data, computing correlations, and applying filters if required. It optionally 
    generates plots for the analysis. The final DataFrame includes computed phase proportions 
    and, if specified, the columns that were excluded from processing.
    
    """
    
    # -------------------#
    # BASIC INPUT CHECK #
    # -------------------#

    if not (isinstance(x_in.index, pandas.DatetimeIndex)):
        raise Exception("x_in DataFrame has no DatetimeIndex.")

    # -------------------#

    # x_in = output.copy(deep=True)

    time_stopper = [['Init', time.perf_counter()]]
    X = x_in.copy(deep=True)

    if len(remove_from_process) > 0:
        X = X.drop(remove_from_process, axis=1)

    if len(X.columns) < 2:
        raise Exception("Not enough columns. Need at least two.")
        
    if len(time_frame_apply)==0:
        raise Exception("Not enough time frames to apply. At least one.")

    # make output vector
    Y = X.copy(deep=True)

    time_stopper.append(['Copy', time.perf_counter()])
    
    if('h' in time_frame_apply):
    
        # -------------------------#
        #          HOUR            #
        # -------------------------#
    
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
    
        time_stopper.append(['Hour-Group', time.perf_counter()])
    
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
    
    
        str_plot = ""
        if(apply_filter):       
            df_relation = RemoveOutliersQuantile(df_relation) 
            str_plot = " (filtered)"  
    
        if(plot):
            df_relation.plot(title="Relation Hour" + str_plot)        
            
            
    
        time_stopper.append(['Hour-Corr', time.perf_counter()])
    
        for i in range(0, len(comb_vet)):
            comb = comb_vet[i]
            comb_str = comb_vet_str[i]
            df_relation.loc[:, comb_str] = df_relation.loc[:, comb_str] * X.iloc[:, list(comb)[1]]
    
        time_stopper.append(['Hour-Relation', time.perf_counter()])
    
        for i in range(0, len(comb_vet)):
            comb = comb_vet[i]
            comb_str = comb_vet_str[i]
            Y.loc[
                (Y.iloc[:, list(comb)[0]].isnull()) & (~df_relation.loc[:, comb_str].isnull()),
                Y.columns[list(comb)[0]]] = df_relation.loc[(Y.iloc[:, list(comb)[0]].isnull()) &
                                                            (~df_relation.loc[:, comb_str].isnull()), comb_str]
    
        #Make sure that if all phases are lost it does not use the phase proportion                     
        mark_lost_all_columns = X.isnull().sum(axis=1) >= len(X.columns)    
        Y[mark_lost_all_columns] = numpy.nan
               
    
        time_stopper.append(['Hour', time.perf_counter()])
    if('pd' in time_frame_apply):
        
        # -------------------------#
        #    PERIOD OF THE DAY     #
        # -------------------------#
    
        mask_valid = ~X.isnull()
        grouper_valid = mask_valid.groupby([mask_valid.index.year, mask_valid.index.month, mask_valid.index.day,
                                            DayPeriodMapperVet(mask_valid.index.hour)])
        count_valid = grouper_valid.transform('sum')
    
        mask_null = X.isnull()
        grouper_null = mask_null.groupby(
            [mask_null.index.year, mask_null.index.month, mask_null.index.day, DayPeriodMapperVet(mask_valid.index.hour)])
        count_null = grouper_null.transform('sum')
    
        mask_reject = count_valid / (count_null + count_valid) < threshold_accept
    
        grouper = X.groupby([X.index.year, X.index.month, X.index.day, DayPeriodMapperVet(mask_valid.index.hour)])
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
        
        str_plot = ""
        if(apply_filter):       
            df_relation = RemoveOutliersQuantile(df_relation) 
            str_plot = " (filtered)"   
        
        if(plot):
            df_relation.plot(title="Relation Period of the Day" + str_plot)
    
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
    
        #Make sure that if all phases are lost it does not use the phase proportion                     
        mark_lost_all_columns = X.isnull().sum(axis=1) >= len(X.columns)    
        Y[mark_lost_all_columns] = numpy.nan
    
        time_stopper.append(['Patamar', time.perf_counter()])
        
    if('D' in time_frame_apply):
        # -------------------------#
        #          DAY            #
        # -------------------------#
    
        mask_valid = ~X.isnull()
        grouper_valid = mask_valid.groupby([mask_valid.index.year, mask_valid.index.month, mask_valid.index.day])
        count_valid = grouper_valid.transform('sum')
    
        mask_null = X.isnull()
        grouper_null = mask_null.groupby([mask_null.index.year, mask_null.index.month, mask_null.index.day])
        count_null = grouper_null.transform('sum')
    
        mask_reject = count_valid / (count_null + count_valid) < threshold_accept
    
        grouper = X.groupby([X.index.year, X.index.month, X.index.day])
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
        
        str_plot = ""
        if(apply_filter):       
            df_relation = RemoveOutliersQuantile(df_relation) 
            str_plot = " (filtered)"
            
        if(plot):
            df_relation.plot(title="Relation Day" + str_plot)
    
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
                                                            
        #Make sure that if all phases are lost it does not use the phase proportion                     
        mark_lost_all_columns = X.isnull().sum(axis=1) >= len(X.columns)    
        Y[mark_lost_all_columns] = numpy.nan
    
        time_stopper.append(['Day', time.perf_counter()])
    
    if('M' in time_frame_apply):
        # -------------------------#
        #          MONTH          #
        # -------------------------#
    
        mask_valid = ~X.isnull()
        grouper_valid = mask_valid.groupby([mask_valid.index.year, mask_valid.index.month])
        count_valid = grouper_valid.transform('sum')
    
        mask_null = X.isnull()
        grouper_null = mask_null.groupby([mask_null.index.year, mask_null.index.month])
        count_null = grouper_null.transform('sum')
    
        mask_reject = count_valid / (count_null + count_valid) < threshold_accept
    
        grouper = X.groupby([X.index.year, X.index.month])
        X_mean = grouper.transform('mean')
    
        X_mean[mask_reject] = numpy.nan
    
        #  Make all the possible permutations between columns
        comb_vet = list(permutations(range(0, X_mean.shape[1]), r=2))
    
        #  make columns names
        comb_vet_str = []
        for comb in comb_vet:
            comb_vet_str.append(str(comb[0]) + '-' + str(comb[1]))
    
        #  Create relation vector
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
        
        str_plot = ""
        if(apply_filter):       
            df_relation = RemoveOutliersQuantile(df_relation) 
            str_plot = " (filtered)"
        
        if(plot):
            df_relation.plot(title="Relation Month" + str_plot)
    
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
    
                                                            
        #Make sure that if all phases are lost it does not use the phase proportion                     
        mark_lost_all_columns = X.isnull().sum(axis=1) >= len(X.columns)    
        Y[mark_lost_all_columns] = numpy.nan
                                                            
        time_stopper.append(['Month', time.perf_counter()])
        
    if('S' in time_frame_apply):
        # -------------------------#
        #       HUMID/DRY         #
        # -------------------------#
    
        mask_valid = ~X.isnull()
        grouper_valid = mask_valid.groupby([YearPeriodMapperVet(mask_valid.index.month)])
        count_valid = grouper_valid.transform('sum')
    
        mask_null = X.isnull()
        grouper_null = mask_null.groupby([YearPeriodMapperVet(mask_valid.index.month)])
        count_null = grouper_null.transform('sum')
    
        mask_reject = count_valid / (count_null + count_valid) < threshold_accept
    
        grouper = X.groupby([YearPeriodMapperVet(mask_valid.index.month)])
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
        
        str_plot = ""
        if(apply_filter):       
            df_relation = RemoveOutliersQuantile(df_relation) 
            str_plot = " (filtered)"
        
        if(plot):
            df_relation.plot(title="Relation Season" + str_plot)
    
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
    
                                                            
        #Make sure that if all phases are lost it does not use the phase proportion                     
        mark_lost_all_columns = X.isnull().sum(axis=1) >= len(X.columns)    
        Y[mark_lost_all_columns] = numpy.nan
                                                            
        time_stopper.append(['Season', time.perf_counter()])
    if('Y' in time_frame_apply):
        
        # -------------------------#
        #          YEAR           #
        # -------------------------#
    
        mask_valid = ~X.isnull()
        grouper_valid = mask_valid.groupby([mask_valid.index.year])
        count_valid = grouper_valid.transform('sum')
    
        mask_null = X.isnull()
        grouper_null = mask_null.groupby([mask_null.index.year])
        count_null = grouper_null.transform('sum')
    
        mask_reject = count_valid / (count_null + count_valid) < threshold_accept
    
        grouper = X.groupby([X.index.year])
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
        
        str_plot = ""
        if(apply_filter):       
            df_relation = RemoveOutliersQuantile(df_relation) 
            str_plot = " (filtered)"
            
        if(plot):
            df_relation.plot(title="Relation Year" + str_plot)
    
    
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
                                                            
                                                            
        #Make sure that if all phases are lost it does not use the phase proportion                     
        mark_lost_all_columns = X.isnull().sum(axis=1) >= len(X.columns)    
        Y[mark_lost_all_columns] = numpy.nan
    
        time_stopper.append(['Year', time.perf_counter()])

    if('A' in time_frame_apply):
        # -------------------------#
        #     ALL TIME SERIES     #
        # -------------------------#
    
        X_mean = X.copy(deep=True)
    
        for col in X_mean.columns.values:
            X_mean[col] = X_mean[col].mean()
    
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
        
        str_plot = ""
        if(apply_filter):       
            df_relation = RemoveOutliersQuantile(df_relation) 
            str_plot = " (filtered)"
        
        if(plot):
            df_relation.plot(title="Relation All Samples" + str_plot)
            
    
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
    
        #Make sure that if all phases are lost it does not use the phase proportion                     
        mark_lost_all_columns = X.isnull().sum(axis=1) >= len(X.columns)    
        Y[mark_lost_all_columns] = numpy.nan
                                                            
        time_stopper.append(['AllTimeSeries', time.perf_counter()])
    
        # return the keep out columns
        if len(remove_from_process) > 0:
            Y = pandas.concat([Y, x_in.loc[:, remove_from_process]], axis=1)
    
        time_stopper.append(['Final', time.perf_counter()])
    
        TimeProfile(time_stopper, name='Phase', show=False)
    
    if(plot):
        matplotlib.pyplot.show()

    return Y


def CountMissingData(x_in: pandas.core.frame.DataFrame, remove_from_process: list = [], show=False) -> float:
    """
    Calculates the number of vacacies on the dataset.


    :param x_in: A pandas.core.frame.DataFrame where the index is of type "pandas.core.indexes.datetime.DatetimeIndex"
    and each column contain an electrical quantity time series.
    :type x_in: pandas.core.frame.DataFrame

    :param remove_from_process: Columns to be kept off the process.
    :type remove_from_process: list,optional

    :param show: Specify if the function should print or not the value that is also returned.
    :type show: bool,optional


    :return: Y: Returns the amount of vacancies.
    :rtype: Y: float

    """
    Y = x_in.loc[:, x_in.columns.difference(remove_from_process)].isnull().sum().sum()
    if show:
        print(f"Total number of missing samples {Y}")

    return Y


def CalcUnbalance(x_in: pandas.core.frame.DataFrame, remove_from_process: list = []) -> pandas.core.frame.DataFrame:
    """
    Calculates the unbalance between phases for every timestamp.

    Equation:
        Y = (MAX-MEAN)/MEAN

    Ref.: Derating of induction motors operating with a combination of unbalanced voltages and over or under-voltages


    :param x_in: A pandas.core.frame.DataFrame where the index is of type "pandas.core.indexes.datetime.DatetimeIndex"
    and each column contain an electrical quantity time series.
    :type x_in: pandas.core.frame.DataFrame

    :param remove_from_process: Columns to be kept off the process.
    :type remove_from_process: list,optional

    :return: Y: A pandas.core.frame.DataFrame with the % of unbalance between columns (phases).
    :rtype: Y: pandas.core.frame.DataFrame

    """

    X = x_in.loc[:, x_in.columns.difference(remove_from_process)]

    Y = pandas.DataFrame([], index=x_in.index)

    Y['Unbalance'] = 100 * (X.max(axis=1) - X.mean(axis=1)) / X.mean(axis=1)

    return Y


def SavePeriod(x_in: pandas.core.frame.DataFrame,
               df_save: pandas.core.frame.DataFrame) -> tuple:
    """
    For a given set of periods (Start->End) returns the data. It also returns the indexes.

    :param x_in: A pandas.core.frame.DataFrame where the index is of type "pandas.core.indexes.datetime.DatetimeIndex" 
    and each column contain an electrical quantity time series.
    :type x_in: pandas.core.frame.DataFrame

    :param df_save: The first column with the start and the second column with the end date.
    :type df_save: pandas.core.frame.DataFrame

    :return: Y,mark_index_not: The input pandas.core.frame.DataFrame sliced by the df_save periods. it also returns
    the indexes
    :rtype: Y,mark_index_not: tuple

    """

    Y = x_in.copy(deep=True)
    mark_index_not = x_in.index

    for index, row in df_save.iterrows():
        Y = Y.loc[numpy.logical_and(Y.index >= row[0], Y.index <= row[1]), :]
        mark_index_not = mark_index_not[numpy.logical_and(mark_index_not >= row[0], mark_index_not <= row[1])]

    return Y, mark_index_not


def RemoveOutliersMMADMM(x_in: pandas.core.frame.DataFrame,
                         df_avoid_periods: pandas.core.frame.DataFrame = pandas.DataFrame([]),
                         len_mov_avg: int = 4 * 12,
                         std_def: float = 2,
                         min_var_def: float = 0.5,
                         allow_negatives: bool = False,
                         plot: bool = False,
                         remove_from_process: list = [],
                         ) -> pandas.core.frame.DataFrame:
    """
    Removes outliers from the timeseries on each column using the (M)oving (M)edian (A)bslute
    (D)eviation around the (M)oving (M)edian.

    A statistical method is used for removing the remaining outliers. In LEYS et al. (2019), the authors state that it
    is common practice the use of plus and minus the standard deviation (±σ) around the mean (µ), however,
    this measurement is particularly sensitive to outliers. Furthermore, the authors propose the use of the
    absolute deviation around the median.Therefore, in this work the limit was set by the median absolute
    deviation (MADj) around the moving median (Mj) where j denotes the number of samples of the moving window.
    Typically, an MV feeder has a seasonality where in the summer load is higher than in the winter or vice-versa.
    Hence, it is vital to use the moving median instead of the median of all the time series.


    :param x_in: A pandas.core.frame.DataFrame where the index is of type "pandas.core.indexes.datetime.DatetimeIndex"
    and each column contain an electrical quantity time series.
    :type x_in: pandas.core.frame.DataFrame

    :param df_avoid_periods: The first column with the start and the second column with the end date.
    :type df_avoid_periods: pandas.core.frame.DataFrame

    :param len_mov_avg: Size of the windows of the moving average.
    :type len_mov_avg: int,optional

    :param std_def: Absolute standard deviation to be computed around the moving average.
    :type std_def: float,optional

    :param min_var_def: For low variance data this parameter will set a minimum distance from the upper and lower
    boundaries.
    :type min_var_def: float,optional

    :param allow_negatives: Allow for the lower level to be below zero.
    :type allow_negatives: bool,optional

    :param plot: A plot of the boundaries and result to debug parameters.
    :type plot: bool,optional

    :param remove_from_process: Columns to be kept off the process.
    :type remove_from_process: list,optional

    :raises Exception: if x_in has no DatetimeIndex.

    :return: Y: A pandas.core.frame.DataFrame without the outliers
    :rtype: Y: pandas.core.frame.DataFrame

    """
    # -------------------#
    # BASIC INPUT CHECK #
    # -------------------#

    if not (isinstance(x_in.index, pandas.DatetimeIndex)):
        raise Exception("x_in DataFrame has no DatetimeIndex.")

    X = x_in.copy(deep=True)

    if len(remove_from_process) > 0:
        X = X.drop(remove_from_process, axis=1)

    Y = X.copy(deep=True)

    # ------------------------ OUTLIERS ------------------------

    X_mark_outlier = X.copy(deep=True)
    X_mark_outlier.loc[:, :] = False

    # ---------PROCESSAMENTO OUTLIERS POR MÉDIA MÓVEL
    X_moving_median = X.copy(deep=True)

    # DESVIO PADRÂO ABSOLUTO ENTORNO DA MEDIANA MOVEL

    # ------------ Computa Mediana Móvel ------------#
    X_moving_median = X_moving_median.rolling(len_mov_avg).median().shift(-int(len_mov_avg / 2))

    X_moving_median.iloc[-2 * len_mov_avg:, :] = X_moving_median.iloc[-2 * len_mov_avg:, :].fillna(method='ffill')
    X_moving_median.iloc[:2 * len_mov_avg, :] = X_moving_median.iloc[:2 * len_mov_avg, :].fillna(method='bfill')

    # ------------ Computa MAD Móvel ------------#
    X_mad = X - X_moving_median
    X_mad = X_mad.rolling(len_mov_avg).median().shift(-int(len_mov_avg / 2))
    X_mad.iloc[-2 * len_mov_avg:, :] = X_mad.iloc[-2 * len_mov_avg:, :].fillna(method='ffill')
    X_mad.iloc[:2 * len_mov_avg, :] = X_mad.iloc[:2 * len_mov_avg, :].fillna(method='bfill')

    # ------------ Coloca no mínimo de faixa de segurança para dados com baixa variância ------------#
    X_mad[X_mad <= min_var_def] = min_var_def

    # ------------ MAD Móvel Limites ------------#
    X_moving_up = X_moving_median + std_def * X_mad
    X_moving_down = X_moving_median - std_def * X_mad

    # Allow the lower limit to go negative. Only valid for kVar or bidirectional current/Power.
    if ~allow_negatives:
        X_moving_down[X_moving_down <= 0] = 0

    # ------------ Marcando outliers ------------#
    X_mark = (X >= X_moving_up) | (X <= X_moving_down)

    # ------------ Não marca os intervalos onde não foi possível determinar ------------#
    X_mark[X_moving_up.isnull() | X_moving_down.isnull()] = False
    X_mark.iloc[:int(len_mov_avg / 2), :] = False
    X_mark.iloc[-int(len_mov_avg / 2), :] = False

    Y[X_mark] = numpy.nan

    # ------------ Não marca os intervalos selecionados ------------#
    if df_avoid_periods.shape[0] != 0:
        df_values, index_return = SavePeriod(X, df_avoid_periods)
        Y.loc[index_return, :] = df_values

    # return the keep out columns
    if len(remove_from_process) > 0:
        Y = pandas.concat([Y, x_in.loc[:, remove_from_process]], axis=1)

    # For debug
    if plot:
        #ax = X_moving_median.plot()
        ax = x_in.plot(title = 'RemoveOutliersMMADMM')        
        #X_mad.plot(ax=ax)
        X_moving_down.plot.line(ax=ax,color='black',style='-')
        X_moving_up.plot(ax=ax,color='black',style='-')        
        X[X_mark].plot(ax=ax,color='red',style='.')
        #Y.plot()
        matplotlib.pyplot.show()

    return Y


def MarkNanPeriod(x_in: pandas.core.frame.DataFrame,
                  df_remove: pandas.core.frame.DataFrame,
                  remove_from_process: list = []) -> pandas.core.frame.DataFrame:
    """
    Marks as nan all specified timestamps

    :param x_in: A pandas.core.frame.DataFrame where the index is of type "pandas.core.indexes.datetime.DatetimeIndex"
    and each column contain an electrical quantity time series.
    :type x_in: pandas.core.frame.DataFrame

    :param df_remove: List of periods to mark as nan. The first column with the start and the second column with
    the end date all in datetime.
    :type df_remove: pandas.core.frame.DataFrame

    :param remove_from_process: Columns to be kept off the process;
    :type remove_from_process: list,optional

    :return: Y: The input pandas.core.frame.DataFrame with samples filled based on the proportion between time series.
    :rtype: Y: pandas.core.frame.DataFrame

    """

    Y = x_in.copy(deep=True)

    # Remove the keep out columns
    if len(remove_from_process) > 0:
        Y = Y.drop(remove_from_process, axis=1)

    for index, row in df_remove.iterrows():
        Y.loc[numpy.logical_and(Y.index >= row[0], Y.index <= row[1]), Y.columns.difference(
            remove_from_process)] = numpy.nan

    # return the keep out columns
    if len(remove_from_process) > 0:
        Y = pandas.concat([Y, x_in.loc[:, remove_from_process]], axis=1)

    return Y


def RemoveOutliersHardThreshold(x_in: pandas.core.frame.DataFrame,
                                hard_max: float,
                                hard_min: float,
                                remove_from_process: list = [],
                                df_avoid_periods=pandas.DataFrame([])) -> pandas.core.frame.DataFrame:
    """
    Removes outliers from the timeseries on each column using threshold.

    :param x_in: A pandas.core.frame.DataFrame where the index is of type "pandas.core.indexes.datetime.DatetimeIndex" 
    and each column contain an electrical quantity time series.
    :type x_in: pandas.core.frame.DataFrame

    :param hard_max: Max value for the threshold limit
    :type hard_max: float

    :param hard_min: Min value for the threshold limit
    :type hard_min: float

    :param remove_from_process: Columns to be kept off the process;
    :type remove_from_process: list,optional

    :param df_avoid_periods: The first column with the start and the second column with the end date.
    :type df_avoid_periods: pandas.core.frame.DataFrame


    :return: Y: A pandas.core.frame.DataFrame without the outliers
    :rtype: Y: pandas.core.frame.DataFrame

    """
    X = x_in.copy(deep=True)

    #  Remove keep out columns
    if len(remove_from_process) > 0:
        X = X.drop(remove_from_process, axis=1)

    Y = X.copy(deep=True)

    Y[Y >= hard_max] = numpy.nan
    Y[Y <= hard_min] = numpy.nan

    if df_avoid_periods.shape[0] != 0:
        df_values, index_return = SavePeriod(X, df_avoid_periods)
        Y.loc[index_return, :] = df_values

    # return the keep out columns
    if len(remove_from_process) > 0:
        Y = pandas.concat([Y, x_in.loc[:, remove_from_process]], axis=1)

    return Y


def RemoveOutliersQuantile(x_in: pandas.core.frame.DataFrame,
                           remove_from_process: list = [],
                           df_avoid_periods=pandas.DataFrame([])) -> pandas.core.frame.DataFrame:
    """
     Removes outliers from the timeseries on each column using the top and bottom
     quantile metric as an outlier marker.

     :param x_in: A pandas.core.frame.DataFrame where the index is of type "pandas.core.indexes.datetime.DatetimeIndex"
     and each column contain an electrical quantity time series.
     :type x_in: pandas.core.frame.DataFrame

     :param remove_from_process: Columns to be kept off the process;
     :type remove_from_process: list,optional

     :param df_avoid_periods: The first column with the start and the second column with the end date.
     :type df_avoid_periods: pandas.core.frame.DataFrame


     :return: Y: A pandas.core.frame.DataFrame without the outliers
     :rtype: Y: pandas.core.frame.DataFrame

    """

    X = x_in.copy(deep=True)

    # Remove the keep out columns
    if len(remove_from_process) > 0:
        X = X.drop(remove_from_process, axis=1)

    Y = X.copy(deep=True)

    for col_name in Y.columns:
        q1 = X[col_name].quantile(0.25)
        q3 = X[col_name].quantile(0.75)
        iqr = q3 - q1  # Inter quartile range        
        fence_low = q1 - 1.5 * iqr
        fence_high = q3 + 1.5 * iqr
        Y.loc[(Y[col_name] < fence_low) | (Y[col_name] > fence_high), col_name] = numpy.nan

    if df_avoid_periods.shape[0] != 0:
        df_values, index_return = SavePeriod(X, df_avoid_periods)
        Y.loc[index_return, :] = df_values

    # return the keep out columns
    if len(remove_from_process) > 0:
        Y = pandas.concat([Y, x_in.loc[:, remove_from_process]], axis=1)

    return Y


def RemoveOutliersHistogram(x_in: pandas.core.frame.DataFrame,
                            df_avoid_periods: pandas.DataFrame = pandas.DataFrame([]),
                            remove_from_process: list = [],
                            integrate_hour: bool = True,
                            sample_freq: int = 5,
                            min_number_of_samples_limit: int = 12) -> pandas.core.frame.DataFrame:
    """
    Removes outliers from the timeseries on each column using the histogram.
    The parameter 'min_number_of_samples_limit' specify the minimum amount of hours, if integrate flag is True, or
    samples that a value must have to be considered not an outlier.

    :param x_in: A pandas.core.frame.DataFrame where the index is of type "pandas.core.indexes.datetime.DatetimeIndex"
    and each column contain an electrical quantity time series.
    :type x_in: pandas.core.frame.DataFrame

    :param remove_from_process: Columns to be kept off the process;
    :type remove_from_process: list,optional

    :param df_avoid_periods: The first column with the start and the second column with the end date.
    :type df_avoid_periods: pandas.core.frame.DataFrame

    :param integrate_hour: Makes the analysis on the data integrated to an hour
    :type integrate_hour: bool,optional

    :param sample_freq: The sample frequency of the time series. Defaults to 5.
    :type sample_freq: int,optional

    :param min_number_of_samples_limit: The number of samples to be considered valid
    :type min_number_of_samples_limit: int,optional


    :return: Y: A pandas.core.frame.DataFrame without the outliers
    :rtype: Y: pandas.core.frame.DataFrame


    """

    X = x_in.copy(deep=True)

    # Remove the keep out columns
    if len(remove_from_process) > 0:
        X = X.drop(remove_from_process, axis=1)

    Y = X.copy(deep=True)

    # Remove outliers outside the avoid period
    if integrate_hour:
        Y_int = IntegrateHour(Y, sample_freq)
        Y_int = Y_int.reset_index(drop=True)
    else:
        Y_int = X.copy(deep=True)

    for col in Y_int:
        Y_int[col] = Y_int[col].sort_values(ascending=False, ignore_index=True)

    if Y_int.shape[0] < min_number_of_samples_limit:
        min_number_of_samples_limit = Y_int.shape[0]

    threshold_max = Y_int.iloc[min_number_of_samples_limit + 1, :]
    threshold_min = Y_int.iloc[-min_number_of_samples_limit - 1, :]

    for col in Y:
        Y.loc[numpy.logical_or(Y[col] > threshold_max[col], Y[col] < threshold_min[col]), col] = numpy.nan

    if df_avoid_periods.shape[0] != 0:
        df_values, index_return = SavePeriod(X, df_avoid_periods)
        Y.loc[index_return, :] = df_values

    # return the keep out columns
    if len(remove_from_process) > 0:
        Y = pandas.concat([Y, x_in.loc[:, remove_from_process]], axis=1)

    return Y


def SimpleProcess(x_in: pandas.core.frame.DataFrame,
                  start_date_dt: datetime,
                  end_date_dt: datetime,
                  remove_from_process: list = [],
                  sample_freq: int = 5,
                  sample_time_base: str = 'm',
                  pre_interpol: int = False,
                  pos_interpol: int = False,
                  prop_phases: bool = False,
                  integrate: bool = False,
                  interpol_integrate: int = False) -> pandas.core.frame.DataFrame:
    """

    Simple pre-made imputation process.

    ORGANIZE->INTERPOLATE->PHASE_PROPORTION->INTERPOLATE->INTEGRATE->INTERPOLATE


    :param x_in: A pandas.core.frame.DataFrame where the index is of type "pandas.core.indexes.datetime.DatetimeIndex"
    and each column contain an electrical quantity time series.
    :type x_in: pandas.core.frame.DataFrame

    :param start_date_dt: The start date where the synchronization should start.
    :type start_date_dt: datetime

    :param end_date_dt: The end date where the synchronization will consider samples.
    :type end_date_dt: datetime

    :param remove_from_process: Columns to be kept off the process Only on PhaseProportionInput step.
    :type remove_from_process: list,optional

    :param sample_freq: The sample frequency of the time series. Defaults to 5.
    :type sample_freq: int,optional

    :param sample_time_base: The base time of the sample frequency. Specify if the sample frequency is in (D)ay,
    (M)onth, (Y)ear, (h)ours, (m)inutes, or (s)econds. Defaults to (m)inutes.
    :type sample_time_base: srt,optional

    :param pre_interpol: Number of samples to limit the first interpolation after organizing the data.
    Defaults to False.
    :type pre_interpol: int,optional

    :param pos_interpol: Number of samples to limit the second interpolation after PhaseProportionInput the data.
    Defaults to False.
    :type pos_interpol: int,optional

    :param prop_phases: Apply the PhaseProportionInput method
    :type prop_phases: bool,optional

    :param integrate: Integrates to 1 hour time stamps. Defaults to False.
    :type integrate: bool,optional

    :param interpol_integrate: Number of samples to limit the third interpolation after IntegrateHour the data.
    Defaults to False.
    :type interpol_integrate: int,optional

    :return: Y: The x_in pandas.core.frame.DataFrame with no missing data. Treated with a simple step process.
    :rtype: Y: pandas.core.frame.DataFrame

    """

    # Organize samples
    Y = DataSynchronization(x_in, start_date_dt, end_date_dt, sample_freq, sample_time_base=sample_time_base)

    # Interpolate before proportion between phases
    if pre_interpol:
        Y = Y.interpolate(method_type='linear', limit=pre_interpol)

    # Uses proportion between phases
    if prop_phases:
        Y = PhaseProportionInput(Y, threshold_accept=0.60, remove_from_process=remove_from_process)

    # Interpolate after proportion between phases
    if pos_interpol:
        Y = Y.interpolate(method_type='linear', limit=pos_interpol)

    # Integralization 1h
    if integrate:
        Y = IntegrateHour(Y, sample_freq=5)

        # Interpolate after Integralization 1h
        if interpol_integrate:
            Y = Y.interpolate(method_type='linear', limit=interpol_integrate)

    return Y


def ReturnOnlyValidDays(x_in: pandas.core.frame.DataFrame,
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
    
    if not(isinstance(x_in.index, pandas.core.frame.DatetimeIndex)):
        raise Exception("DataFrame has no DatetimeIndex.")
    if sample_time_base not in ['s', 'm', 'h']:
        raise Exception("The sample_time_base is not in seconds, minutes or hours.")

    X = x_in.copy(deep=True)
    
    if len(remove_from_process) > 0:
        X = X.drop(remove_from_process, axis=1)

    qty_sample_dic = {'s': 24 * 60 * 60, 'm': 24 * 60, 'h': 24}

    df_count = X.groupby([X.index.year, X.index.month, X.index.day]).count() / (
                qty_sample_dic[sample_time_base] / sample_freq)

    time_vet_stamp = X.index[numpy.arange(0, len(X.index), int((qty_sample_dic[sample_time_base] / sample_freq)))]
    df_count = df_count.reset_index(drop=True)
    df_count.insert(0, 'timestamp_day', time_vet_stamp)
    df_count.set_index('timestamp_day', inplace=True)
    df_count = df_count >= threshold_accept
    
    df_count = df_count.sum(axis=1) == df_count.shape[1]
    df_count.name = 'isValid'
    df_count = df_count.reset_index()
    X['timestamp_day'] = X.index.floor("D").values

    keep_X_index = X.index
    X = pandas.merge(X, df_count, on='timestamp_day', how='left')
    X.index = keep_X_index
    X = X.loc[X['isValid'] == True, :]

    X.drop(columns=['isValid', 'timestamp_day'], inplace=True)
    df_count.set_index('timestamp_day', inplace=True)

    return X, df_count


def GetDayMaxMin(x_in: pandas.core.frame.DataFrame, start_date_dt: datetime, end_date_dt:datetime, sample_freq: int =5, threshold_accept:float=1.0, exe_param:str='max'):
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
    
    if not(isinstance(x_in.index, pandas.core.frame.DatetimeIndex)):
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

    time_vet_stamp = pandas.to_datetime(
        vet_idx['Year'].astype(str) + '-' + vet_idx['Month'].astype(str) + '-' + vet_idx['Day'].astype(str))

    vet_idx.drop(columns=['Year', 'Month', 'Day'], axis=1, inplace=True)
    vet_idx = vet_idx.reset_index(drop=True)
    vet_idx.insert(0, 'timestamp_day', time_vet_stamp)
    vet_idx.set_index('timestamp_day', inplace=True)

    # redo the timestamp index
    Y.index.rename(['Year', 'Month', 'Day'], inplace=True)
    Y = Y.reset_index(drop=False)

    time_vet_stamp = pandas.to_datetime(Y['Year'].astype(str) + '-' + Y['Month'].astype(str) + '-' + Y['Day'].astype(str))

    Y.drop(columns=['Year', 'Month', 'Day'], axis=1, inplace=True)
    Y = Y.reset_index(drop=True)
    Y.insert(0, 'timestamp_day', time_vet_stamp)
    Y.set_index('timestamp_day', inplace=True)

    Y = DataSynchronization(Y, start_date_dt, end_date_dt, sample_freq=1, sample_time_base='D')

    vet_idx = pandas.merge(vet_idx, Y, left_index=True, right_index=True, how='right', suffixes=('', '_remove'))
    vet_idx.drop(columns=vet_idx.columns[vet_idx.columns.str.contains('_remove')], axis=1, inplace=True)

    # Missing days get midnight as the  hour of max and min
    for col in vet_idx.columns.values:
        vet_idx.loc[vet_idx[col].isna(), col] = vet_idx.index[vet_idx[col].isna()]

    # Interpolate by day of the week
    Y = Y.groupby(Y.index.weekday, group_keys=False).apply(lambda x: x.interpolate())
    Y = Y.groupby(Y.index.weekday, group_keys=False).apply(lambda x: x.ffill())
    Y = Y.groupby(Y.index.weekday, group_keys=False).apply(lambda x: x.bfill())

    return Y, vet_idx


def GetWeekDayCurve(x_in: pandas.core.frame.DataFrame, sample_freq:int=5, threshold_accept:float=1.0, min_sample_per_day:int=3, min_sample_per_workday:int=9):
    """
    Analyzes and normalizes time series data in a DataFrame to compute average curves for each weekday, 
    considering various sampling and validity thresholds.

    :param x_in: Input DataFrame with a DatetimeIndex.
    :type: pandas.core.frame.DataFrame
    :param sample_freq: Sampling frequency in minutes, default is 5.
    :type: int
    :param threshold_accept: Threshold for accepting valid data, default is 1.0.
    :type: float
    :param min_sample_per_day: Minimum samples required per day to consider the data valid, default is 3.
    :type: int
    :param min_sample_per_workday: Minimum samples required per workday (Monday to Friday) to consider the data valid, default is 9.
    :type: int
    
    :raises Exception: If the DataFrame does not have a DatetimeIndex.

    :return: A DataFrame containing the normalized data for each weekday.
    :rtype: pandas.core.frame.DataFrame
    """

    # BASIC INPUT CHECK

    if not (isinstance(x_in.index, pandas.core.frame.DatetimeIndex)):
        raise Exception("DataFrame has no DatetimeIndex.")
   
    
    X = x_in.copy(deep=True)

    Y, df_count = ReturnOnlyValidDays(X, sample_freq, threshold_accept)

    # Get valid data statistics
    df_count = df_count.loc[df_count['isValid'], :]
    df_stats = df_count.groupby(df_count.index.weekday).count()

    # fill days that does not exist with count zero.
    for i_day in range(0,7):
        if i_day not in df_stats.index.values:
            print(f'Weekday {i_day} does not exist.')
            df_stats.loc[i_day] = 0

    # Has enough data do use ?
    if numpy.min(df_stats['isValid'].values) >= min_sample_per_day:
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
        work_days = df_stats.loc[df_stats.index <= 4, 'isValid'].sum()
        sat_qty = df_stats.loc[df_stats.index == 5, 'isValid'].sum()
        sun_qty = df_stats.loc[df_stats.index == 6, 'isValid'].sum()

        if (work_days >= min_sample_per_workday) and sun_qty >= min_sample_per_day and sat_qty >= min_sample_per_day:
            print('Can calculate a curve for every weekday and use Sat. and Sun.')

            Y['WeekDay'] = Y.index.weekday.values
            Y['Hour'] = Y.index.hour.values
            Y['Min'] = Y.index.minute.values
            Y = Y.reset_index(drop=True)
            Y.loc[Y['WeekDay'] <= 4, 'WeekDay'] = 0

            Y = Y.groupby([Y.WeekDay, Y.Hour, Y.Min]).mean()
            Y.index.names = ['WeekDay', 'Hour', 'Min']
            Y = Y.reset_index()

            # Normalization max min each day
            grouper = Y.groupby([Y.WeekDay])
            maxes = grouper.transform('max')
            mins = grouper.transform('min')

            Y.iloc[:, 3:] = (Y.iloc[:, 3:] - mins.iloc[:, 2:]) / (maxes.iloc[:, 2:] - mins.iloc[:, 2:])

            for i_day in [1, 2, 3, 4]:
                Y_day_aux = Y.loc[Y.WeekDay == 0, :].copy(deep=True)
                Y_day_aux.WeekDay = i_day
                Y = pandas.concat((Y, Y_day_aux))
            Y = Y.reset_index(drop=True)

        else:
            print('Not enough data using default curve.')
            Y = pandas.read_pickle("./default.wdc")

    return Y


def GetNSSCPredictedSamples(max_vet: pandas.core.frame.DataFrame,
                            min_vet: pandas.core.frame.DataFrame,
                            weekday_curve: pandas.core.frame.DataFrame,
                            start_date_dt: datetime, 
                            end_date_dt:datetime,
                            sample_freq: int = 5,                        
                            sample_time_base: str = 'm') -> pandas.core.frame.DataFrame:
    """
    Generate predicted samples for NSSC using maximum and minimum vectors, 
    and a curve based on weekdays.

    :param max_vet: The maximum vector DataFrame.
    :type max_vet: pandas.core.frame.DataFrame
    :param min_vet: The minimum vector DataFrame.
    :type min_vet: pandas.core.frame.DataFrame
    :param weekday_curve: DataFrame representing the curve based on weekdays.
    :type weekday_curve: pandas.core.frame.DataFrame
    :param sample_freq: The frequency of sampling. Defaults to 5.
    :type sample_freq: int
    :param sample_time_base: The base unit of time for sampling, can be 's', 'm', or 'h'. Defaults to 'm'.
    :type sample_time_base: str

    :raises Exception: If the sample_time_base is not 's', 'm', or 'h'.

    :return: A DataFrame with predicted values.
    :rtype: pandas.core.frame.DataFrame
    """
    
    
    # BASIC INPUT CHECK

    if sample_time_base not in ['s', 'm', 'h']:
        raise Exception("The sample_time_base is not in seconds, minutes or hours.")

    max_vet = max_vet.iloc[numpy.repeat(numpy.arange(len(max_vet)), 12*24)]
    min_vet = min_vet.iloc[numpy.repeat(numpy.arange(len(min_vet)), 12*24)]

    time_array = numpy.arange(start_date_dt, end_date_dt, numpy.timedelta64(sample_freq, sample_time_base),dtype='datetime64')

    vet_samples = pandas.core.frame.DataFrame(index=time_array, dtype=object)
    vet_samples.index.name = 'timestamp'

    num_days = int(vet_samples.shape[0] / (12 * 24))
    first_day = vet_samples.index[0].weekday()

    weekday_curve_vet_begin = weekday_curve.iloc[(first_day * 12 * 24):, :].reset_index(drop=True)
    num_mid_weeks = int(numpy.floor((num_days - (7 - first_day)) / 7))
    weekday_curve_vet_mid = pandas.concat([weekday_curve] * num_mid_weeks)
    num_end_days = num_days - num_mid_weeks * 7 - (7 - first_day)
    weekday_curve_vet_end = weekday_curve.iloc[:num_end_days * (12 * 24), :].reset_index(drop=True)

    weekday_curve_vet = pandas.concat([weekday_curve_vet_begin, weekday_curve_vet_mid, weekday_curve_vet_end])

    weekday_curve_vet = weekday_curve_vet.reset_index(drop=True)

    
    weekday_curve_vet.drop(columns=['WeekDay', 'Hour', 'Min'], inplace=True)
    weekday_curve_vet.index.name = 'timestamp'
    weekday_curve_vet.index = vet_samples.index

    max_vet.index = vet_samples.index
    min_vet.index = vet_samples.index

    Y = (max_vet - min_vet) * weekday_curve_vet + min_vet

    return Y


def ReplaceData(x_in:pandas.core.frame.DataFrame,
                x_replace:pandas.core.frame.DataFrame,
                start_date_dt: datetime,
                end_date_dt: datetime,                
                num_samples_day:int = 12*24,
                day_threshold:float = 0.5,
                patamar_threshold:float = 0.5,
                num_samples_patamar:int = 12*6,                
                sample_freq:int = 5,
                sample_time_base:str = 'm' ) -> pandas.core.frame.DataFrame:
    """
    Replaces data in a DataFrame based on specified conditions and thresholds.

    :param x_in: The input DataFrame containing the data to be analyzed and replaced.
    :type x_in: pandas.core.frame.core.frame.DataFrame
    :param x_replace: The DataFrame containing replacement data.
    :type x_replace: pandas.core.frame.core.frame.DataFrame
    :param start_date_dt: The start date for the data replacement process.
    :type start_date_dt: datetime
    :param end_date_dt: The end date for the data replacement process.
    :type end_date_dt: datetime
    :param num_samples_day: The number of samples per day, default is 288 (12 * 24).
    :type num_samples_day: int
    :param day_threshold: The threshold for day-based null value analysis, default is 0.5.
    :type day_threshold: float
    :param patamar_threshold: The threshold for patamar-based null value analysis, default is 0.5.
    :type patamar_threshold: float
    :param num_samples_patamar: The number of samples per patamar, default is 72 (12 * 6).
    :type num_samples_patamar: int
    :param sample_freq: The frequency of samples, default is 5.
    :type sample_freq: int
    :param sample_time_base: The time base unit for sampling, default is 'm' (minutes).
    :type sample_time_base: str
    :return: A DataFrame with data replaced based on the specified conditions.
    :rtype: pandas.core.frame.core.frame.DataFrame
    
    Note: `x_in` and `x_replace` must have the same structure and index type.
    """

    #Mark days and patamar with null values greater than threshold
    output_isnull_day = x_in.isnull().groupby([x_in.index.day,x_in.index.month,x_in.index.year]).sum()    
    output_isnull_day.columns = output_isnull_day.columns.values + "_mark"
    output_isnull_day = output_isnull_day/num_samples_day
    
    output_isnull_day.index.rename(['day','month','year'],inplace=True)    
    output_isnull_day.reset_index(inplace=True)    
    output_isnull_day.set_index(output_isnull_day['day'].astype(str) + '-' + output_isnull_day['month'].astype(str) + '-' + output_isnull_day['year'].astype(str),inplace=True)
    output_isnull_day.drop(columns = ['day', 'month', 'year'],inplace=True)
    
    
    output_isnull_day = output_isnull_day>=day_threshold        
    output_isnull_day = output_isnull_day.loc[~(output_isnull_day.sum(axis=1)==0),:]    
    
    #Mark Patamar with null values greater than threshold
    output_isnull_patamar = x_in.copy(deep=True)
    output_isnull_patamar['dp'] = output_isnull_patamar.index.hour.map(DayPeriodMapper)
    output_isnull_patamar = x_in.isnull().groupby([output_isnull_patamar.index.day,output_isnull_patamar.index.month,output_isnull_patamar.index.year,output_isnull_patamar.dp]).sum()        
    output_isnull_patamar.columns = output_isnull_patamar.columns.values + "_mark"
    output_isnull_patamar =output_isnull_patamar/num_samples_patamar
    
    output_isnull_patamar.index.rename(['day', 'month', 'year','dp'],inplace=True)   
    output_isnull_patamar.reset_index(inplace=True)    
    output_isnull_patamar.set_index(output_isnull_patamar['day'].astype(str) + '-' + output_isnull_patamar['month'].astype(str) + '-' + output_isnull_patamar['year'].astype(str) + '-' + output_isnull_patamar['dp'].astype(str),inplace=True)
    output_isnull_patamar.drop(columns = ['day', 'month', 'year','dp'],inplace=True)
    
    
    output_isnull_patamar = output_isnull_patamar>=patamar_threshold        
    output_isnull_patamar = output_isnull_patamar.loc[~(output_isnull_patamar.sum(axis=1)==0),:]    
    
    
    #Create a time array with the same size of x_in
    timearray = numpy.arange(start_date_dt, end_date_dt,numpy.timedelta64(sample_freq,sample_time_base), dtype='datetime64')    
    mark_substitute = pandas.core.frame.DataFrame(index=timearray,columns = x_in.columns.values, dtype=object)    
    mark_substitute.index.name = 'timestamp'
    mark_substitute.loc[:,:] = False
    
    #Create index for day and patamar
    index_day = { 'day': x_in.index.day.values.astype(str), 'month': x_in.index.month.values.astype(str), 'year': x_in.index.year.values.astype(str) }
    index_day = pandas.core.frame.DataFrame(index_day)    
    index_day = index_day['day'].astype(str) + '-' + index_day['month'].astype(str) + '-' + index_day['year'].astype(str)
    
    index_patamar = { 'day': x_in.index.day.values.astype(str), 'month': x_in.index.month.values.astype(str), 'year': x_in.index.year.values.astype(str) }
    index_patamar = pandas.core.frame.DataFrame(index_patamar)    
    index_patamar['dp'] = x_in.index.hour.map(DayPeriodMapper)
    index_patamar = index_patamar['day'].astype(str) + '-' + index_patamar['month'].astype(str) + '-' + index_patamar['year'].astype(str) + '-' + index_patamar['dp'].astype(str)
    
    
    mark_substitute['index_patamar'] = index_patamar.values
    mark_substitute = pandas.merge(mark_substitute, output_isnull_patamar,left_on='index_patamar',right_index=True,how='left').fillna(False)
    for col in x_in.columns.values:
        mark_substitute[col] = mark_substitute[col+'_mark']
        mark_substitute.drop(columns=[col+'_mark'],axis=1,inplace=True)
        
    mark_substitute.drop(columns=['index_patamar'],axis=1,inplace=True)
    
    mark_substitute['index_day'] = index_day.values
    mark_substitute = pandas.merge(mark_substitute, output_isnull_day,left_on='index_day',right_index=True,how='left').fillna(False)    
    
    for col in x_in.columns.values:
        mark_substitute[col] = mark_substitute[col+'_mark']
        mark_substitute.drop(columns=[col+'_mark'],axis=1,inplace=True)
        
    mark_substitute.drop(columns=['index_day'],axis=1,inplace=True)

    #Replace data
    x_out =  x_in.copy(deep=True)    
    x_out[mark_substitute] = x_replace[mark_substitute]


    return x_out


def NSSCInput(x_in: pandas.core.frame.DataFrame,
                 start_date_dt: datetime,
                 end_date_dt: datetime,
                 sample_freq: int = 5,
                 sample_time_base:str='m',
                 threshold_accept_min_max: float = 1.0,
                 threshold_accept_curve: float = 1.0,                 
                 num_samples_day:int = 12*24,
                 num_samples_patamar:int = 12*6,      
                 day_threshold:float = 0.5,
                 patamar_threshold:float = 0.5,                 
                 min_sample_per_day: int = 3,
                 min_sample_per_workday: int = 9) -> pandas.core.frame.DataFrame:
    """
    Implement the NSSC method.

    :param x_in: Input data frame.
    :type x_in: pandas.core.frame.DataFrame
    :param start_date_dt: Start date for the processing.
    :type start_date_dt: datetime
    :param end_date_dt: End date for the processing.
    :type end_date_dt: datetime
    :param sample_freq: Sampling frequency, default is 5.
    :type sample_freq: int
    :param threshold_accept_min_max: Threshold for accepting minimum and maximum values, default is 1.0.
    :type threshold_accept_min_max: float
    :param threshold_accept_curve: Threshold for accepting curve values, default is 1.0.
    :type threshold_accept_curve: float
    :param min_sample_per_day: Minimum number of samples per day, default is 3.
    :type min_sample_per_day: int
    :param num_samples_day: Number of samples per day, default is 288 (12*24).
    :type num_samples_day: int
    :param day_threshold: Day threshold value, default is 0.5.
    :type day_threshold: float
    :param patamar_threshold: Patamar threshold value, default is 0.5.
    :type patamar_threshold: float
    :param num_samples_patamar: Number of samples for patamar, default is 72 (12*6).
    :type num_samples_patamar: int
    :param sample_time_base: Base unit for sample time, default is 'm' for minutes.
    :type sample_time_base: str
    :param min_sample_per_workday: Minimum number of samples per workday, default is 9.
    :type min_sample_per_workday: int

    :return: Processed data frame.
    :rtype: pandas.core.frame.DataFrame
    """
    
    # Get day max/min values
    max_vet,_ = GetDayMaxMin(x_in,start_date_dt,end_date_dt,sample_freq,threshold_accept_min_max,exe_param='max')         
    min_vet,_ = GetDayMaxMin(x_in,start_date_dt,end_date_dt,sample_freq,threshold_accept_min_max,exe_param='min')  

    # Get weekday curve
    weekday_curve = GetWeekDayCurve(x_in, sample_freq, threshold_accept_curve, min_sample_per_day, min_sample_per_workday)
    
    # Get NSSC predicted samples
    X_pred = GetNSSCPredictedSamples(max_vet, min_vet, weekday_curve, sample_freq,sample_time_base)

    # Replace data
    x_out = ReplaceData(x_in,X_pred,start_date_dt,end_date_dt,num_samples_day,day_threshold,patamar_threshold,num_samples_patamar,sample_freq,sample_time_base)
    

    return x_out


def CurrentDummyData(qty_weeks:int = 12*4,start_date_dt:datetime = datetime(2023,1,1)):
    """
    Generates a DataFrame containing dummy time series data.
    
    This function creates a pandas DataFrame representing time series data over a specified number of weeks, starting from a given date. The data includes artificial variations to simulate different patterns, including seasonal variations and random noise. The DataFrame includes columns 'IA', 'IB', 'IV', and 'IN', each containing modified time series data. The index of the DataFrame is set to timestamps at 5-minute intervals, starting from the specified start date.
    
    Parameters
    ----------
    qty_weeks : int, optional
        The number of weeks to generate data for, by default 48 weeks (12*4).
    start_date_dt : datetime, optional
        The start date for the time series data, by default datetime(2023,1,1).
    
    Returns
    -------
    pandas.DataFrame
        A DataFrame containing the generated time series data with columns 'IA', 'IB', 'IV', and 'IN', and a timestamp index.
    
    Examples
    --------
    >>> dummy_data = CurrentDummyData(24, datetime(2023,1,1))
    >>> dummy_data.head()
    """
    
    dummy_week = pandas.DataFrame([[133.4,128.7,122.3,5.7],
                                [131.3,129.2,120.9,4.7],
                                [126.5,124.7,120.9,4.7],
                                [129.7,126.6,121.1,4.7],
                                [128.1,130.2,121.6,5.6],
                                [128.1,130.2,121.6,5.6],
                                [126.6,124.5,119.1,5.3],
                                [126.6,124.5,119.1,5.3],
                                [127.3,125.0,121.3,4.9],
                                [125.0,126.0,120.1,5.0],
                                [125.0,126.0,120.1,5.0],
                                [125.8,123.9,120.2,5.0],
                                [125.8,123.9,120.2,5.0],
                                [121.5,119.9,114.5,4.7],
                                [122.2,120.9,115.8,4.8],
                                [126.6,125.1,114.5,4.8],
                                [121.6,120.1,115.3,4.6],
                                [121.6,120.1,115.3,4.6],
                                [119.8,120.3,114.5,5.0],
                                [121.4,118.8,112.7,4.7],
                                [121.4,118.8,112.7,4.7],
                                [125.2,117.9,111.9,5.0],
                                [125.2,117.9,114.2,5.0],
                                [120.2,119.4,112.0,5.3],
                                [120.8,118.4,113.1,5.4],
                                [120.8,118.4,113.1,5.4],
                                [122.4,118.5,113.3,5.1],
                                [122.4,118.5,113.3,5.1],
                                [121.7,117.0,113.2,4.9],
                                [120.0,117.2,111.5,4.9],
                                [120.0,117.2,111.5,4.9],
                                [120.0,115.2,111.5,4.6],
                                [120.0,115.2,111.5,4.6],
                                [118.1,115.4,110.9,4.7],
                                [118.5,116.9,112.3,4.7],
                                [118.5,116.9,109.2,4.7],
                                [116.6,113.1,109.3,4.8],
                                [116.6,113.1,109.3,4.8],
                                [118.9,114.0,109.9,4.7],
                                [119.0,115.4,110.9,5.0],
                                [119.0,115.4,110.9,5.0],
                                [118.6,115.0,110.0,5.0],
                                [118.6,115.0,110.0,5.0],
                                [117.6,113.3,109.6,5.0],
                                [116.7,113.0,108.4,4.9],
                                [116.7,113.0,108.4,4.9],
                                [117.4,115.5,110.5,4.8],
                                [117.4,115.5,110.5,4.8],
                                [116.6,113.4,109.3,4.6],
                                [115.3,113.8,108.4,4.9],
                                [115.3,113.8,108.4,4.9],
                                [116.2,112.9,108.1,4.9],
                                [116.2,114.6,108.1,4.9],
                                [117.9,113.9,109.7,4.8],
                                [115.8,110.3,106.4,4.4],
                                [115.8,110.3,106.4,4.4],
                                [116.2,113.0,106.6,4.7],
                                [116.2,113.0,106.6,4.7],
                                [115.5,112.0,108.3,4.9],
                                [118.8,112.8,111.6,4.5],
                                [118.8,112.8,114.3,4.5],
                                [115.1,112.5,108.1,4.3],
                                [115.1,109.5,104.2,4.3],
                                [113.2,111.7,107.3,4.2],
                                [109.3,109.2,105.0,4.4],
                                [109.3,109.2,105.0,4.4],
                                [108.7,108.1,104.1,5.1],
                                [110.1,108.1,104.1,5.1],
                                [108.5,108.0,105.2,4.7],
                                [108.0,108.4,103.6,4.7],
                                [108.0,109.5,103.6,4.7],
                                [109.0,105.0,103.6,4.4],
                                [105.0,105.0,99.0,4.4],
                                [110.1,109.6,104.3,4.5],
                                [112.1,113.0,108.0,4.5],
                                [112.1,113.0,108.0,4.9],
                                [115.2,112.5,107.0,4.4],
                                [110.2,110.4,104.5,4.4],
                                [113.4,111.8,107.0,4.5],
                                [112.9,111.6,105.7,4.5],
                                [112.9,111.6,105.7,4.5],
                                [112.1,108.4,106.8,4.4],
                                [115.9,115.4,110.2,3.9],
                                [115.4,114.9,112.0,4.9],
                                [115.2,115.8,111.9,4.2],
                                [115.2,115.8,115.3,4.2],
                                [115.5,113.2,111.3,4.3],
                                [115.5,113.2,115.4,4.3],
                                [117.2,115.7,111.0,3.9],
                                [117.0,117.4,111.4,4.1],
                                [117.0,115.7,111.4,4.1],
                                [117.7,117.4,110.1,3.9],
                                [119.6,119.3,116.8,3.9],
                                [122.2,123.9,116.7,4.0],
                                [125.2,123.5,115.8,4.7],
                                [127.7,126.8,118.3,4.7],
                                [128.5,125.3,117.9,4.4],
                                [124.9,121.8,122.8,4.4],
                                [131.3,127.3,121.0,4.9],
                                [132.0,127.0,125.9,4.3],
                                [128.2,127.2,123.4,4.3],
                                [131.8,126.2,118.7,4.5],
                                [136.5,130.0,124.6,4.0],
                                [136.1,143.1,123.3,5.1],
                                [136.4,132.0,128.2,4.0],
                                [136.2,132.6,130.0,4.0],
                                [135.1,131.4,131.3,4.2],
                                [129.6,128.2,125.8,4.2],
                                [130.6,129.8,126.7,4.3],
                                [133.8,132.9,129.0,4.2],
                                [132.1,130.4,127.4,4.8],
                                [133.7,129.8,127.4,4.3],
                                [131.5,131.9,128.5,4.6],
                                [140.9,139.7,131.7,4.1],
                                [142.6,137.0,132.5,3.6],
                                [135.7,137.5,132.3,3.6],
                                [139.0,141.3,132.8,3.9],
                                [140.3,141.4,132.9,3.9],
                                [142.3,141.7,133.4,4.4],
                                [137.3,135.4,133.0,3.7],
                                [144.3,136.7,133.9,3.7],
                                [152.7,155.2,133.4,3.7],
                                [145.1,138.6,138.2,3.7],
                                [146.4,142.0,133.7,3.9],
                                [153.3,146.7,141.2,3.9],
                                [146.2,145.4,140.9,4.1],
                                [143.1,142.3,135.1,3.7],
                                [151.5,149.1,141.2,3.7],
                                [151.4,144.0,144.3,4.0],
                                [149.1,147.3,144.9,4.0],
                                [145.7,143.0,139.3,4.1],
                                [142.1,141.8,137.7,3.9],
                                [142.5,142.5,139.5,3.9],
                                [144.3,144.2,137.3,3.6],
                                [155.1,149.2,142.5,3.6],
                                [149.6,150.1,148.1,4.2],
                                [153.3,151.5,143.3,4.1],
                                [146.5,142.8,142.6,3.9],
                                [149.2,147.0,140.5,4.0],
                                [153.0,149.9,146.0,4.2],
                                [155.8,155.2,147.7,3.9],
                                [153.1,148.7,144.2,3.5],
                                [153.4,154.3,143.0,3.5],
                                [159.1,150.3,149.7,4.0],
                                [155.2,153.0,146.3,4.0],
                                [156.7,154.6,147.5,3.9],
                                [158.5,153.1,147.1,4.1],
                                [156.8,156.6,148.2,4.1],
                                [152.8,153.7,147.2,3.9],
                                [155.1,150.2,147.2,3.9],
                                [153.2,152.1,150.5,5.0],
                                [155.4,153.4,147.2,4.5],
                                [155.6,148.4,147.2,4.0],
                                [156.2,151.6,147.4,4.1],
                                [151.7,151.6,147.4,4.1],
                                [151.5,152.8,144.9,4.1],
                                [152.1,148.2,143.6,4.1],
                                [162.6,158.9,153.8,4.1],
                                [156.2,153.4,150.2,4.0],
                                [152.5,149.0,148.9,4.0],
                                [157.6,154.0,148.9,4.2],
                                [159.9,155.4,152.0,4.2],
                                [159.0,157.2,152.6,4.4],
                                [157.2,155.9,150.8,4.5],
                                [161.0,154.8,152.6,4.3],
                                [165.5,161.8,158.2,4.3],
                                [163.3,160.6,152.8,4.3],
                                [161.6,155.4,153.0,4.3],
                                [166.9,160.6,156.3,4.2],
                                [170.9,168.5,155.1,4.2],
                                [162.7,156.5,151.1,4.5],
                                [162.8,160.2,156.9,4.5],
                                [163.3,157.5,156.2,4.5],
                                [162.6,158.5,152.6,5.3],
                                [162.5,157.8,153.4,4.3],
                                [159.0,158.3,156.3,4.6],
                                [162.0,160.3,156.5,4.7],
                                [166.0,161.3,156.5,4.7],
                                [162.9,159.9,151.3,4.2],
                                [167.1,159.8,160.2,4.2],
                                [161.9,161.3,155.1,4.3],
                                [161.9,160.2,155.4,4.1],
                                [163.5,158.2,154.5,4.1],
                                [164.7,158.3,152.1,4.3],
                                [165.5,162.9,155.6,4.3],
                                [164.7,161.6,155.5,4.2],
                                [168.4,167.9,159.3,5.5],
                                [163.4,158.1,154.9,4.5],
                                [164.5,162.7,156.0,4.7],
                                [166.4,166.9,157.4,4.7],
                                [165.2,163.0,153.4,4.3],
                                [161.6,160.8,152.4,4.3],
                                [162.0,161.0,154.2,4.3],
                                [166.7,164.9,156.7,4.2],
                                [162.0,159.5,149.4,4.2],
                                [163.9,162.6,155.2,4.1],
                                [165.2,161.5,155.2,4.3],
                                [161.3,164.1,155.3,4.3],
                                [161.3,161.9,157.0,3.8],
                                [161.7,162.6,155.1,3.5],
                                [160.0,161.1,152.2,4.0],
                                [158.8,156.1,152.1,3.6],
                                [162.8,161.2,157.4,4.5],
                                [155.2,155.2,148.4,4.1],
                                [157.2,153.3,149.1,4.1],
                                [156.2,153.6,147.9,4.0],
                                [155.5,152.4,148.2,4.2],
                                [154.9,150.9,148.5,4.2],
                                [150.4,151.6,144.6,3.8],
                                [155.3,150.6,145.1,3.8],
                                [150.3,151.2,146.5,4.2],
                                [156.6,152.9,146.2,4.7],
                                [156.0,151.9,143.8,4.5],
                                [156.0,153.1,144.9,4.8],
                                [156.2,154.2,145.4,4.8],
                                [151.1,153.8,145.9,4.1],
                                [151.2,149.2,144.0,4.1],
                                [145.9,144.1,139.0,4.3],
                                [151.1,149.1,137.4,5.1],
                                [146.0,149.1,137.4,5.1],
                                [145.8,144.9,139.8,4.8],
                                [152.1,146.4,139.7,5.5],
                                [153.0,153.3,142.5,5.2],
                                [154.2,151.3,144.4,5.0],
                                [158.1,158.0,152.7,5.0],
                                [163.1,158.2,147.6,5.3],
                                [165.7,160.1,152.7,5.3],
                                [166.0,162.8,155.3,5.1],
                                [165.4,162.8,155.1,6.2],
                                [167.1,160.7,155.7,6.2],
                                [164.5,164.5,156.7,5.8],
                                [174.3,164.8,164.5,5.2],
                                [171.9,170.6,162.3,5.1],
                                [173.7,170.6,160.7,5.3],
                                [174.4,165.6,160.7,6.2],
                                [172.2,171.9,162.4,5.4],
                                [173.3,171.9,166.3,5.4],
                                [178.4,178.1,164.1,5.5],
                                [173.9,168.6,161.8,5.9],
                                [169.1,168.9,162.2,5.9],
                                [169.4,166.8,160.6,5.1],
                                [175.7,167.1,164.0,5.1],
                                [171.0,168.7,163.7,4.9],
                                [172.3,167.1,163.9,5.0],
                                [166.9,164.5,161.7,5.0],
                                [172.0,167.5,162.2,4.8],
                                [171.7,162.0,159.6,4.8],
                                [172.4,167.1,164.8,5.4],
                                [168.4,166.8,158.5,5.8],
                                [172.5,168.8,161.5,5.8],
                                [167.5,165.1,162.7,5.3],
                                [169.9,163.5,156.4,5.3],
                                [169.8,161.5,161.4,6.3],
                                [164.7,163.6,156.9,5.8],
                                [171.8,166.5,161.5,5.3],
                                [170.6,162.4,157.3,5.7],
                                [170.6,162.4,157.3,5.7],
                                [166.6,165.0,154.9,6.3],
                                [166.9,160.5,155.7,5.4],
                                [166.9,162.1,157.6,6.3],
                                [169.2,167.3,156.3,5.7],
                                [169.2,167.3,156.3,5.3],
                                [164.2,162.3,152.9,5.5],
                                [170.8,166.3,158.7,6.3],
                                [165.7,165.9,158.7,6.3],
                                [165.6,162.7,155.9,5.9],
                                [166.4,166.2,158.3,5.9],
                                [166.9,161.2,153.1,5.7],
                                [166.3,161.5,156.0,5.6],
                                [164.0,161.5,156.0,6.4],
                                [162.6,159.4,154.9,5.8],
                                [162.6,159.4,149.9,5.8],
                                [162.6,161.5,154.9,6.0],
                                [164.8,163.7,153.3,5.9],
                                [165.9,163.6,157.9,5.9],
                                [166.0,164.5,157.3,5.4],
                                [166.0,159.2,154.6,5.4],
                                [162.5,165.4,153.5,5.8],
                                [161.9,162.2,152.5,6.4],
                                [164.2,159.9,153.9,6.4],
                                [160.2,160.2,151.5,5.9],
                                [165.4,159.9,153.0,5.9],
                                [160.5,159.4,148.9,6.0],
                                [158.1,155.7,148.6,5.8],
                                [158.1,155.1,146.4,5.8],
                                [155.7,150.6,146.9,5.3],
                                [157.3,155.7,146.9,6.6],
                                [157.0,155.7,148.3,5.8],
                                [157.0,156.0,146.7,5.8],
                                [157.0,156.0,146.7,6.5],
                                [154.6,154.7,145.4,6.2],
                                [154.6,150.8,145.4,5.4],
                                [152.7,152.6,143.8,5.6],
                                [151.7,153.7,141.5,6.5],
                                [151.7,148.6,140.3,5.5],
                                [151.4,149.4,141.1,5.6],
                                [152.6,151.9,138.9,5.6],
                                [150.6,146.9,140.1,5.6],
                                [152.3,148.7,140.2,5.2],
                                [147.3,148.7,140.2,5.2],
                                [147.2,146.9,137.2,5.6],
                                [147.7,143.6,133.7,5.6],
                                [146.9,143.1,133.6,5.2],
                                [148.0,145.2,136.4,5.4],
                                [148.0,145.2,135.7,5.5],
                                [144.6,141.3,133.8,5.5],
                                [142.3,140.5,132.0,5.5],
                                [143.7,141.0,134.0,5.6],
                                [142.3,140.4,131.3,5.7],
                                [143.3,140.4,132.9,6.7],
                                [140.8,138.8,131.9,6.1],
                                [140.8,136.0,131.9,5.7],
                                [141.3,137.1,127.9,5.6],
                                [142.3,137.8,130.1,5.6],
                                [139.1,138.6,130.3,5.6],
                                [141.1,139.8,132.6,5.8],
                                [141.1,139.8,127.6,5.8],
                                [139.7,134.8,130.8,5.4],
                                [139.3,135.1,128.6,5.4],
                                [139.3,136.9,130.9,5.4],
                                [138.7,136.7,129.5,5.3],
                                [138.7,136.7,129.5,5.3],
                                [136.0,131.7,126.8,5.6],
                                [137.7,133.5,126.1,5.6],
                                [137.0,134.9,128.4,5.6],
                                [137.5,133.3,128.5,5.4],
                                [137.5,133.3,123.5,5.4],
                                [135.5,129.8,126.3,5.4],
                                [132.7,129.6,123.4,5.2],
                                [133.5,129.6,126.8,5.2],
                                [133.8,131.0,124.2,5.3],
                                [134.8,133.6,121.7,5.3],
                                [133.0,129.3,124.2,5.4],
                                [133.2,130.3,122.9,5.3],
                                [133.2,132.2,124.5,5.3],
                                [129.8,129.2,123.5,5.3],
                                [130.0,129.2,125.8,5.3],
                                [135.1,128.8,123.5,5.5],
                                [131.2,127.8,121.9,5.1],
                                [131.3,127.8,123.5,5.1],
                                [131.5,130.4,120.9,5.5],
                                [131.4,126.9,118.4,5.5],
                                [129.5,126.9,122.3,5.4],
                                [132.8,129.4,122.0,5.3],
                                [132.8,129.4,122.0,5.3],
                                [128.9,125.8,121.1,5.0],
                                [128.9,125.8,121.1,5.0],
                                [132.7,126.3,119.5,5.2],
                                [128.1,124.5,120.2,5.0],
                                [122.9,121.3,115.5,5.0],
                                [124.1,122.9,112.8,5.3],
                                [124.1,122.4,115.5,4.7],
                                [122.3,120.7,113.4,4.9],
                                [118.9,119.4,110.3,5.0],
                                [118.9,117.3,110.3,5.0],
                                [118.3,117.3,110.7,5.1],
                                [120.0,117.8,112.6,4.7],
                                [116.8,117.6,107.5,4.9],
                                [118.5,119.9,110.6,5.0],
                                [118.5,118.8,114.2,5.0],
                                [115.1,113.9,109.4,4.6],
                                [116.2,114.5,109.4,4.6],
                                [117.2,121.2,114.9,4.7],
                                [119.8,120.1,114.6,5.3],
                                [123.9,122.5,116.0,4.7],
                                [123.0,121.5,115.5,5.2],
                                [125.1,122.3,115.5,4.7],
                                [122.6,118.9,114.5,4.5],
                                [120.3,121.6,112.1,4.6],
                                [121.0,121.6,112.7,4.6],
                                [121.7,122.5,117.0,4.9],
                                [121.7,122.5,117.9,4.9],
                                [126.2,125.7,113.4,4.8],
                                [129.0,126.4,123.5,4.4],
                                [126.7,125.9,120.7,4.4],
                                [130.0,130.4,121.8,4.7],
                                [130.1,130.3,122.2,4.7],
                                [128.4,126.4,123.4,4.5],
                                [131.4,130.2,129.5,4.5],
                                [136.6,136.9,130.9,4.1],
                                [133.6,133.5,124.7,4.4],
                                [133.6,131.3,127.7,4.4],
                                [131.1,136.9,128.1,4.4],
                                [136.2,132.1,128.5,4.4],
                                [135.4,137.9,127.5,4.5],
                                [139.6,140.9,138.2,4.3],
                                [144.3,142.0,128.6,4.3],
                                [143.6,145.3,134.6,4.3],
                                [147.1,145.4,133.1,4.3],
                                [141.2,139.6,132.9,4.4],
                                [142.2,142.0,135.9,4.4],
                                [145.4,141.8,136.2,4.4],
                                [141.3,141.9,138.9,4.3],
                                [141.3,142.3,139.0,4.3],
                                [142.0,141.0,133.4,4.4],
                                [143.4,139.7,134.9,4.4],
                                [146.1,141.4,136.0,4.4],
                                [148.0,148.2,143.4,4.1],
                                [149.7,143.1,139.6,4.1],
                                [152.0,147.6,140.5,3.6],
                                [150.5,148.8,139.8,4.4],
                                [156.2,151.7,144.0,4.4],
                                [151.0,151.7,139.6,3.7],
                                [153.4,150.0,146.8,3.7],
                                [155.6,150.2,144.7,4.3],
                                [153.3,152.2,143.1,4.3],
                                [152.0,148.5,143.8,4.1],
                                [155.1,152.9,146.1,3.7],
                                [155.4,150.7,148.0,3.7],
                                [152.3,148.9,145.9,4.5],
                                [157.3,153.9,147.9,4.5],
                                [155.6,156.7,148.4,4.1],
                                [161.4,156.5,147.7,3.8],
                                [159.4,159.4,149.6,3.8],
                                [157.9,156.0,146.4,4.3],
                                [157.9,156.3,151.7,4.4],
                                [158.5,155.9,146.7,3.9],
                                [156.9,156.2,145.5,4.1],
                                [159.2,155.3,150.2,4.1],
                                [160.3,153.8,150.5,4.0],
                                [155.1,154.6,155.6,4.0],
                                [156.2,158.0,150.5,4.0],
                                [158.9,154.0,148.7,4.1],
                                [159.8,157.3,153.5,4.1],
                                [154.7,152.3,148.2,4.1],
                                [159.8,157.9,153.3,4.1],
                                [160.7,157.8,150.5,4.3],
                                [162.6,162.1,157.8,4.2],
                                [164.5,159.9,155.3,4.2],
                                [167.7,164.0,157.6,4.7],
                                [162.1,164.0,152.6,4.7],
                                [165.1,162.7,157.7,3.8],
                                [166.2,165.7,157.9,4.1],
                                [168.1,164.4,154.8,4.1],
                                [168.3,167.5,158.5,4.1],
                                [164.3,164.6,153.6,4.1],
                                [166.1,167.3,158.8,4.2],
                                [169.7,164.2,158.2,4.0],
                                [164.6,161.3,155.3,4.0],
                                [161.1,161.1,155.4,4.0],
                                [161.1,156.4,154.9,4.0],
                                [162.6,159.1,155.2,4.1],
                                [162.5,158.9,154.5,4.3],
                                [165.4,159.6,156.6,4.4],
                                [163.3,159.5,153.2,3.7],
                                [165.5,159.7,158.8,3.7],
                                [165.8,159.7,159.2,3.8],
                                [172.9,167.7,164.3,3.8],
                                [164.0,163.6,156.4,3.8],
                                [168.4,167.0,161.4,3.8],
                                [171.4,167.1,161.4,3.8],
                                [172.6,167.8,159.0,3.8],
                                [168.7,168.3,158.4,4.2],
                                [169.6,163.5,158.4,4.5],
                                [171.0,164.2,161.2,4.3],
                                [165.9,164.3,161.7,3.6],
                                [169.9,165.2,159.8,3.7],
                                [170.0,166.1,157.3,3.9],
                                [172.9,168.1,160.0,3.9],
                                [169.9,164.1,158.7,4.9],
                                [174.8,179.8,156.9,4.2],
                                [173.7,167.8,161.5,4.1],
                                [170.9,168.8,161.2,3.9],
                                [173.0,170.4,162.7,3.9],
                                [174.0,167.6,158.5,4.3],
                                [170.5,171.0,162.7,4.3],
                                [171.0,168.6,158.8,4.4],
                                [166.8,169.2,160.3,4.5],
                                [169.1,169.8,164.2,4.5],
                                [170.0,167.7,166.1,4.3],
                                [168.9,169.3,161.1,4.3],
                                [173.6,172.9,163.1,4.6],
                                [165.0,166.2,159.3,4.1],
                                [168.6,166.6,160.1,4.1],
                                [171.9,165.1,161.9,4.3],
                                [171.3,168.4,165.8,4.3],
                                [173.2,171.6,164.2,4.1],
                                [169.7,165.5,160.3,4.1],
                                [169.5,167.0,162.0,4.1],
                                [176.3,179.4,154.2,4.3],
                                [168.3,163.5,159.8,3.8],
                                [168.6,164.5,154.6,3.9],
                                [170.0,165.6,157.3,3.9],
                                [168.7,167.1,155.1,3.9],
                                [171.7,168.0,163.2,4.2],
                                [171.3,166.6,163.0,4.0],
                                [167.4,164.4,155.0,4.1],
                                [174.2,178.9,158.3,3.8],
                                [166.8,165.1,160.2,4.5],
                                [167.4,165.9,159.7,3.9],
                                [165.6,163.9,153.1,4.2],
                                [162.4,161.6,154.5,3.4],
                                [161.8,159.9,154.1,3.4],
                                [157.7,154.6,148.9,3.3],
                                [156.8,156.6,147.1,3.9],
                                [158.8,155.6,149.4,3.9],
                                [155.5,155.7,149.1,4.0],
                                [150.4,151.1,144.4,4.0],
                                [150.3,150.0,144.4,4.1],
                                [156.0,147.9,143.7,4.1],
                                [156.7,160.7,149.9,4.1],
                                [157.7,154.0,149.2,4.2],
                                [157.7,155.2,149.2,5.0],
                                [157.0,154.7,150.0,4.0],
                                [152.0,149.6,145.4,4.1],
                                [146.9,149.3,144.6,4.1],
                                [151.3,149.3,139.3,4.5],
                                [151.3,154.4,144.4,5.0],
                                [151.9,149.4,143.3,4.5],
                                [151.9,152.6,143.5,4.8],
                                [156.4,149.5,149.4,4.8],
                                [154.5,154.7,147.1,5.1],
                                [161.6,160.7,154.6,5.1],
                                [167.3,165.7,153.8,6.1],
                                [165.8,164.7,158.1,5.8],
                                [165.8,164.7,158.1,5.8],
                                [167.9,170.6,160.9,5.5],
                                [172.3,170.6,160.9,6.1],
                                [170.9,175.7,164.9,5.7],
                                [177.3,174.3,166.0,5.5],
                                [177.3,174.3,166.0,5.5],
                                [177.2,176.1,165.1,5.9],
                                [177.4,176.1,165.1,5.9],
                                [177.7,174.9,165.0,5.6],
                                [178.3,179.7,171.2,5.8],
                                [178.3,181.2,171.2,5.8],
                                [177.8,176.2,165.1,5.6],
                                [177.8,176.2,170.5,5.6],
                                [179.6,177.9,170.7,5.7],
                                [182.6,176.0,168.6,5.6],
                                [177.7,176.0,170.5,5.6],
                                [177.7,175.9,170.4,5.7],
                                [182.7,175.9,170.4,5.7],
                                [177.6,173.3,165.5,5.8],
                                [177.0,172.8,167.6,5.9],
                                [177.0,172.8,167.2,5.9],
                                [177.1,174.8,167.6,5.7],
                                [177.1,174.8,167.6,5.7],
                                [178.2,173.4,166.5,5.9],
                                [173.6,171.0,162.8,5.9],
                                [172.5,171.0,162.8,5.9],
                                [172.7,168.8,161.4,5.3],
                                [172.7,171.0,161.4,5.3],
                                [171.9,165.9,160.6,5.2],
                                [170.2,169.9,160.8,5.8],
                                [170.2,169.9,160.8,6.2],
                                [173.2,168.5,157.0,5.5],
                                [167.4,168.5,157.0,5.5],
                                [170.1,171.0,158.8,5.7],
                                [169.6,165.9,162.6,6.2],
                                [169.6,171.0,162.6,7.4],
                                [170.8,170.3,161.6,6.6],
                                [170.8,165.8,161.6,6.6],
                                [173.0,169.6,162.6,6.3],
                                [170.6,167.2,160.3,6.2],
                                [172.8,167.2,160.3,6.2],
                                [171.8,171.1,163.0,5.6],
                                [167.8,165.9,157.1,5.6],
                                [167.8,170.4,161.0,6.6],
                                [169.8,165.7,158.2,5.9],
                                [172.9,171.0,162.3,5.9],
                                [170.6,167.9,160.3,6.0],
                                [170.6,166.0,160.3,6.0],
                                [170.4,166.1,157.3,6.0],
                                [169.5,166.6,157.4,6.3],
                                [169.5,166.6,157.4,6.3],
                                [169.0,171.0,157.1,6.6],
                                [168.3,168.8,157.3,6.5],
                                [168.3,166.0,157.3,6.5],
                                [166.2,163.9,156.0,6.3],
                                [162.7,160.8,156.0,6.3],
                                [167.9,165.9,154.5,6.3],
                                [163.5,161.6,153.1,6.2],
                                [162.8,161.6,157.0,6.2],
                                [164.4,160.8,152.0,6.2],
                                [164.4,160.8,152.0,6.2],
                                [161.3,158.4,152.4,6.2],
                                [159.1,156.1,149.6,6.3],
                                [159.1,155.8,149.6,6.3],
                                [158.3,161.2,149.0,5.8],
                                [158.3,156.1,149.0,5.8],
                                [158.1,156.4,148.9,6.0],
                                [157.0,157.5,147.2,6.3],
                                [157.0,157.5,147.2,6.3],
                                [158.1,155.4,147.6,5.9],
                                [152.6,155.4,147.6,5.9],
                                [154.6,151.0,147.3,5.9],
                                [152.7,152.2,144.8,5.8],
                                [152.7,152.2,142.2,5.8],
                                [151.6,145.9,141.6,5.9],
                                [147.2,146.0,141.6,6.4],
                                [150.2,148.5,140.0,5.9],
                                [150.8,148.9,142.9,5.8],
                                [150.8,146.7,142.1,5.8],
                                [151.2,148.0,141.5,6.0],
                                [147.3,148.0,137.0,6.4],
                                [147.0,146.3,139.0,5.4],
                                [146.3,144.3,137.2,5.5],
                                [146.3,144.3,137.2,6.5],
                                [147.9,144.6,137.6,5.6],
                                [147.9,144.6,137.6,5.6],
                                [144.5,142.9,136.6,6.7],
                                [145.5,143.3,136.5,5.9],
                                [145.5,143.3,136.5,5.7],
                                [142.5,142.8,134.2,6.0],
                                [141.2,139.2,134.2,5.7],
                                [142.0,139.9,137.2,5.8],
                                [144.9,141.4,135.4,5.7],
                                [141.7,139.0,135.4,5.7],
                                [143.2,140.3,132.6,6.1],
                                [143.2,140.3,132.6,6.1],
                                [140.7,138.0,130.0,5.7],
                                [143.6,139.4,131.6,5.6],
                                [143.6,139.4,131.6,5.6],
                                [140.7,136.4,129.9,5.5],
                                [140.7,136.4,129.9,5.7],
                                [142.0,136.1,129.7,6.1],
                                [141.3,138.5,129.9,5.8],
                                [141.3,138.5,129.9,5.8],
                                [138.1,135.6,130.1,5.7],
                                [136.6,135.6,130.1,5.7],
                                [139.3,136.3,131.0,5.7],
                                [138.4,135.6,128.8,5.8],
                                [141.6,135.6,128.8,5.8],
                                [138.5,134.0,132.2,5.8],
                                [138.5,134.0,132.1,5.8],
                                [137.4,134.4,129.3,5.9],
                                [136.6,133.5,127.9,5.9],
                                [136.6,133.5,127.9,5.9],
                                [136.0,133.3,126.8,5.5],
                                [136.0,133.3,126.8,5.5],
                                [136.5,133.8,127.5,5.6],
                                [135.0,131.5,125.7,5.9],
                                [135.0,131.5,125.7,5.9],
                                [136.9,132.9,125.5,5.8],
                                [136.9,132.9,125.5,5.8],
                                [131.1,128.9,122.0,5.6],
                                [129.5,126.3,119.3,5.4],
                                [129.5,126.3,119.3,5.4],
                                [124.7,123.8,116.5,5.7],
                                [124.7,128.9,116.5,5.7],
                                [125.6,124.8,118.0,5.2],
                                [120.9,121.7,114.5,5.0],
                                [126.0,123.6,117.1,5.0],
                                [121.4,118.3,114.4,5.1],
                                [121.4,123.5,117.5,5.1],
                                [122.9,118.4,112.5,5.3],
                                [121.2,123.7,116.6,5.0],
                                [121.2,118.7,112.2,5.8],
                                [121.2,124.2,117.3,4.8],
                                [126.6,123.8,117.3,4.8],
                                [127.2,126.2,122.3,4.9],
                                [126.8,127.2,120.7,4.9],
                                [126.8,123.8,117.2,4.9],
                                [132.2,128.6,122.5,4.9],
                                [127.1,128.6,122.5,4.9],
                                [128.6,128.6,122.6,4.9],
                                [128.9,126.8,120.5,4.8],
                                [132.1,128.9,120.5,4.8],
                                [132.8,125.8,122.4,4.6],
                                [132.2,125.8,127.6,4.6],
                                [132.5,128.5,122.3,5.8],
                                [134.5,131.6,124.8,4.8],
                                [137.5,133.7,127.3,4.7],
                                [135.6,134.1,127.2,5.7],
                                [135.6,139.1,132.5,4.7],
                                [132.5,134.1,127.5,4.9],
                                [138.5,137.1,129.8,4.6],
                                [138.5,137.1,129.8,4.6],
                                [141.7,141.5,132.5,5.2],
                                [140.2,138.7,133.2,5.2],
                                [148.0,141.9,133.5,4.7],
                                [143.3,143.1,135.8,4.5],
                                [145.6,142.8,138.0,4.5],
                                [148.6,145.9,137.3,4.7],
                                [145.5,148.1,141.1,4.7],
                                [147.5,145.9,136.4,4.7],
                                [148.7,145.7,140.2,4.5],
                                [149.8,149.8,144.9,4.5],
                                [151.0,148.6,144.3,4.1],
                                [155.7,148.2,144.3,4.1],
                                [157.0,159.2,141.4,4.3],
                                [149.0,147.9,141.5,4.0],
                                [152.4,150.3,148.8,4.4],
                                [150.0,148.9,146.3,4.2],
                                [155.6,149.6,143.9,4.2],
                                [151.7,152.6,145.7,3.9],
                                [154.5,145.1,143.3,3.9],
                                [157.6,154.9,147.1,3.8],
                                [154.6,155.7,147.4,3.9],
                                [154.5,152.6,147.4,3.9],
                                [159.8,155.4,147.5,4.1],
                                [157.1,154.2,147.7,4.1],
                                [157.2,157.0,147.8,4.3],
                                [162.2,158.3,149.1,4.2],
                                [161.8,158.6,149.3,4.2],
                                [161.0,158.4,151.3,4.2],
                                [163.1,160.6,150.2,4.2],
                                [163.1,161.2,156.3,4.1],
                                [163.1,160.7,153.2,4.5],
                                [163.4,155.7,149.8,4.4],
                                [163.4,160.0,153.7,4.1],
                                [164.3,162.8,153.8,4.1],
                                [159.2,158.4,149.4,4.2],
                                [164.4,158.2,149.8,4.2],
                                [164.6,163.4,154.7,4.4],
                                [159.6,158.6,154.9,4.1],
                                [165.7,164.9,157.5,4.1],
                                [161.0,156.9,152.5,4.1],
                                [166.8,158.5,157.9,4.1],
                                [167.5,164.1,153.3,4.0],
                                [170.8,169.2,159.8,4.1],
                                [165.1,165.6,160.1,4.1],
                                [167.5,165.2,156.4,4.3],
                                [167.9,163.3,156.8,4.4],
                                [169.2,169.0,159.4,4.3],
                                [173.0,174.1,161.5,4.6],
                                [170.9,170.3,157.0,4.6],
                                [171.6,169.7,162.0,4.6],
                                [171.5,169.8,157.3,4.6],
                                [169.2,168.6,158.2,4.7],
                                [175.8,175.9,166.4,4.6],
                                [173.5,172.5,166.4,4.6],
                                [172.6,173.5,165.3,4.3],
                                [169.1,169.9,159.7,4.3],
                                [171.5,170.0,165.1,4.5],
                                [170.4,169.3,164.7,4.5],
                                [171.8,171.4,163.4,4.4],
                                [172.2,171.4,168.5,4.4],
                                [168.9,170.2,162.9,4.4],
                                [174.5,170.6,162.2,4.5],
                                [171.7,168.3,164.5,4.6],
                                [165.7,166.8,159.8,4.6],
                                [171.4,172.0,164.8,4.4],
                                [171.4,171.9,164.8,4.4],
                                [174.1,170.9,170.0,4.2],
                                [171.8,170.4,164.7,4.6],
                                [174.8,168.8,167.7,4.6],
                                [175.1,174.6,166.0,4.5],
                                [180.1,175.3,166.7,4.5],
                                [174.6,173.9,167.9,4.6],
                                [181.1,189.9,166.4,4.4],
                                [171.2,173.6,165.1,4.4],
                                [174.4,170.3,166.9,4.6],
                                [178.1,173.9,169.3,4.5],
                                [179.3,174.2,164.2,4.3],
                                [175.0,173.7,165.1,4.5],
                                [173.1,174.7,166.3,4.5],
                                [168.0,169.6,166.5,4.6],
                                [180.1,177.9,171.5,4.6],
                                [175.2,175.9,166.6,4.3],
                                [177.9,178.8,166.7,4.3],
                                [178.0,177.9,167.0,4.3],
                                [176.8,175.1,167.8,4.8],
                                [178.2,175.3,167.1,4.8],
                                [176.8,176.1,171.5,4.3],
                                [175.7,175.6,166.2,4.3],
                                [176.6,175.9,167.7,4.3],
                                [179.5,177.1,169.6,4.2],
                                [177.0,177.1,168.1,4.2],
                                [177.4,177.0,168.5,4.5],
                                [174.7,174.4,168.2,4.7],
                                [173.8,172.9,169.7,4.7],
                                [174.3,175.4,168.3,4.8],
                                [174.3,173.4,169.7,4.8],
                                [174.2,174.0,169.8,4.9],
                                [173.5,177.3,169.2,4.9],
                                [178.5,179.4,172.7,5.0],
                                [176.3,171.7,168.5,4.7],
                                [171.5,171.7,169.0,4.7],
                                [170.6,171.9,164.6,4.5],
                                [170.6,171.9,164.7,4.6],
                                [172.1,172.1,165.2,4.1],
                                [171.1,171.1,163.2,4.6],
                                [174.4,171.8,165.3,4.6],
                                [176.8,172.9,165.0,4.2],
                                [172.0,169.4,165.1,4.2],
                                [172.4,164.7,165.1,4.3],
                                [173.3,170.5,165.5,4.8],
                                [171.2,169.2,159.6,4.8],
                                [168.6,167.1,159.9,4.7],
                                [165.3,165.7,159.8,4.7],
                                [162.7,161.0,154.7,4.5],
                                [166.1,163.8,153.8,4.5],
                                [159.8,158.3,150.9,4.5],
                                [159.7,158.2,150.9,4.8],
                                [159.7,158.2,152.2,4.8],
                                [159.7,158.3,153.6,4.1],
                                [158.1,155.8,149.5,4.5],
                                [164.9,163.5,153.9,4.5],
                                [159.9,160.7,148.8,5.1],
                                [159.2,163.7,154.1,5.1],
                                [159.8,158.4,152.6,4.8],
                                [158.9,154.0,147.3,4.5],
                                [158.8,158.2,147.3,4.5],
                                [155.1,154.9,144.0,4.5],
                                [155.1,154.9,144.1,4.5],
                                [158.9,163.5,146.3,4.7],
                                [160.4,158.3,152.0,4.9],
                                [164.2,163.6,154.0,4.9],
                                [169.3,168.9,159.5,4.7],
                                [174.3,168.9,164.8,5.7],
                                [179.4,174.7,170.3,5.2],
                                [175.3,174.3,165.0,5.8],
                                [175.3,174.3,165.0,5.8],
                                [179.7,173.6,170.4,5.2],
                                [179.7,178.7,170.2,5.2],
                                [179.1,178.7,170.4,5.6],
                                [184.3,184.0,171.1,5.6],
                                [179.2,178.9,171.1,5.6],
                                [179.9,178.3,170.5,5.6],
                                [185.0,178.3,170.7,5.6],
                                [179.9,178.9,170.6,6.1],
                                [178.7,177.3,172.5,5.5],
                                [185.0,177.3,170.6,5.5],
                                [184.3,184.0,175.7,5.7],
                                [184.3,178.7,175.7,5.7],
                                [184.2,179.2,174.6,5.9],
                                [181.9,177.3,171.4,5.9],
                                [181.9,177.3,171.6,5.9],
                                [183.8,177.3,171.8,6.0],
                                [180.1,173.6,171.8,6.0],
                                [178.2,173.5,169.6,5.8],
                                [180.3,178.5,169.9,6.0],
                                [185.1,178.5,175.9,6.0],
                                [182.1,176.9,170.7,6.3],
                                [185.2,176.9,170.6,6.3],
                                [182.0,173.1,171.5,5.7],
                                [177.1,172.4,167.1,5.7],
                                [174.9,172.4,167.1,5.7],
                                [173.9,170.9,162.5,5.8],
                                [180.0,170.9,162.5,5.8],
                                [174.9,172.7,166.0,6.3],
                                [173.5,169.7,163.2,6.8],
                                [175.0,172.8,165.7,6.8],
                                [174.8,167.8,160.9,6.0],
                                [174.8,172.8,166.3,6.0],
                                [176.4,167.7,165.9,6.2],
                                [174.2,168.8,161.2,6.0],
                                [175.0,168.8,161.2,6.0],
                                [173.9,171.4,162.0,6.2],
                                [173.9,172.8,162.0,6.2],
                                [175.1,167.6,162.1,6.4],
                                [173.0,168.8,161.0,6.5],
                                [173.0,168.8,161.0,6.8],
                                [174.4,172.8,161.0,6.6],
                                [174.4,167.8,161.0,6.6],
                                [173.9,171.5,161.5,6.4],
                                [174.6,168.9,161.5,6.6],
                                [174.6,168.9,161.5,6.6],
                                [172.9,169.9,161.7,6.8],
                                [170.0,169.9,161.7,6.8],
                                [175.5,172.9,160.9,6.6],
                                [173.3,169.2,158.9,6.6],
                                [170.4,168.0,158.9,6.6],
                                [168.6,166.2,156.2,6.5],
                                [168.6,166.2,161.4,7.8],
                                [169.5,168.2,156.3,6.9],
                                [168.1,166.6,156.6,6.7],
                                [168.1,166.6,156.6,6.7],
                                [165.1,167.1,157.0,6.7],
                                [165.1,167.1,157.0,6.7],
                                [164.7,164.2,155.2,6.8],
                                [164.7,163.0,155.2,6.8],
                                [165.2,163.0,155.8,6.8],
                                [163.7,162.8,153.2,6.7],
                                [163.7,162.8,151.2,6.7],
                                [160.2,157.7,151.1,6.8],
                                [160.2,157.7,151.1,6.8],
                                [161.9,157.6,150.9,6.6],
                                [159.1,157.2,152.8,6.6],
                                [159.1,157.2,152.8,6.6],
                                [157.1,156.4,147.8,6.3],
                                [157.1,156.4,147.8,6.3],
                                [154.9,152.7,145.9,6.2],
                                [154.7,152.7,146.0,6.1],
                                [154.7,152.7,146.0,5.8],
                                [154.1,152.9,145.0,6.1],
                                [154.1,152.9,145.0,5.9],
                                [153.5,147.8,146.2,5.9],
                                [149.6,147.7,141.6,5.9],
                                [149.5,147.7,141.6,5.9],
                                [149.3,148.2,141.5,6.0],
                                [149.3,148.2,141.5,6.0],
                                [149.9,148.2,140.4,6.0],
                                [150.1,145.1,139.6,5.9],
                                [150.1,145.1,139.6,5.9],
                                [149.2,145.2,140.3,5.9],
                                [149.2,145.2,140.3,5.9],
                                [148.0,147.6,135.3,6.1],
                                [147.6,147.13,134.98,6.1],
                                [147.21,146.66,134.66,6.11],
                                [146.81,146.19,134.35,6.11],
                                [146.41,145.71,134.03,6.11],
                                [146.02,145.24,133.71,6.12],
                                [145.62,144.77,133.39,6.12],
                                [145.23,144.3,133.08,6.13],
                                [144.83,143.83,132.76,6.13],
                                [144.43,143.36,132.44,6.13],
                                [144.04,142.89,132.12,6.14],
                                [143.64,142.41,131.8,6.14],
                                [143.24,141.94,131.49,6.14],
                                [142.85,141.47,131.17,6.15],
                                [142.45,141.0,130.85,6.15],
                                [142.05,140.53,130.53,6.15],
                                [141.66,140.06,130.21,6.16],
                                [141.26,139.59,129.9,6.16],
                                [140.86,139.11,129.58,6.16],
                                [140.47,138.64,129.26,6.17],
                                [140.07,138.17,128.94,6.17],
                                [139.68,137.7,128.63,6.18],
                                [139.28,137.23,128.31,6.18],
                                [138.88,136.76,127.99,6.18],
                                [138.49,136.29,127.67,6.19],
                                [138.09,135.81,127.35,6.19],
                                [137.69,135.34,127.04,6.19],
                                [137.3,134.87,126.72,6.2],
                                [136.9,134.4,126.4,6.2],
                                [135.15,131.9,127.25,6.05],
                                [133.4,129.4,128.1,5.9],
                                [136.3,131.6,122.8,5.4],
                                [134.2,131.6,122.8,5.4],
                                [137.5,132.2,126.4,5.2],
                                [137.5,132.2,126.4,5.2],
                                [134.1,129.6,122.9,5.4],
                                [130.8,127.8,120.4,4.9],
                                [130.8,127.8,117.9,4.9],
                                [128.4,128.1,118.5,5.1],
                                [128.4,124.6,118.5,5.5],
                                [123.6,124.7,112.6,5.0],
                                [124.7,121.9,116.6,5.1],
                                [124.7,121.9,116.6,5.1],
                                [125.5,119.9,113.9,4.8],
                                [121.5,119.9,113.9,4.8],
                                [121.9,125.6,113.4,4.9],
                                [123.1,121.8,115.4,4.9],
                                [123.1,121.8,110.2,5.6],
                                [123.1,122.8,113.7,5.6],
                                [126.1,125.7,119.0,5.6],
                                [124.3,126.5,114.2,4.9],
                                [125.0,123.3,114.9,5.1],
                                [128.0,128.7,120.0,4.6],
                                [125.4,126.2,118.7,4.8],
                                [123.7,123.1,116.7,4.8],
                                [128.4,127.5,118.5,5.6],
                                [127.9,127.0,120.4,5.5],
                                [128.2,128.1,120.2,5.5],
                                [127.1,127.5,120.2,4.9],
                                [125.7,124.2,122.2,4.9],
                                [126.8,129.3,120.9,4.6],
                                [129.9,129.9,123.5,5.2],
                                [129.9,129.8,123.5,4.6],
                                [136.4,135.1,127.9,5.0],
                                [137.8,140.4,127.7,5.7],
                                [143.2,140.5,133.0,4.6],
                                [137.8,139.5,133.3,4.5],
                                [137.3,135.4,128.5,4.5],
                                [135.6,135.6,131.5,4.8],
                                [136.6,137.2,131.6,4.8],
                                [137.0,137.3,132.1,4.5],
                                [139.8,138.4,130.7,4.5],
                                [138.6,134.9,128.6,4.5],
                                [138.7,138.6,130.3,4.4],
                                [144.7,141.3,132.6,4.4],
                                [149.3,145.0,137.8,4.5],
                                [144.1,142.4,134.3,4.5],
                                [146.4,146.3,136.2,4.5],
                                [157.5,164.9,137.6,4.4],
                                [148.5,146.6,139.6,4.4],
                                [145.3,143.7,134.5,3.9],
                                [147.9,147.2,137.2,3.6],
                                [150.4,147.5,139.1,4.5],
                                [152.0,151.4,142.6,4.2],
                                [152.8,150.0,141.2,4.2],
                                [154.9,148.6,138.4,4.1],
                                [154.6,150.3,147.6,4.4],
                                [152.4,149.5,141.6,4.4],
                                [155.6,151.8,145.3,4.2],
                                [152.3,148.8,141.4,4.2],
                                [154.0,154.2,145.6,4.0],
                                [154.7,151.9,146.9,4.0],
                                [154.5,150.4,148.0,4.3],
                                [156.8,154.8,144.4,4.3],
                                [157.4,155.8,146.3,4.1],
                                [163.7,155.5,152.2,4.2],
                                [156.9,156.7,146.4,3.5],
                                [157.7,155.6,144.9,3.7],
                                [158.6,157.4,149.9,3.7],
                                [158.6,157.6,148.4,3.8],
                                [154.2,153.9,151.4,3.8],
                                [159.4,155.7,149.9,3.6],
                                [159.1,155.8,145.1,4.1],
                                [158.0,153.6,145.8,4.1],
                                [158.1,156.3,146.0,4.8],
                                [159.8,156.2,149.1,4.5],
                                [163.1,155.5,149.9,4.8],
                                [167.2,158.6,154.9,4.2],
                                [166.1,160.8,158.7,4.0],
                                [163.1,160.2,155.1,4.0],
                                [165.7,161.6,152.7,4.1],
                                [164.0,164.9,156.0,4.1],
                                [163.7,163.4,153.8,4.4],
                                [165.9,160.1,153.9,4.4],
                                [165.1,164.0,155.7,4.3],
                                [165.1,163.7,155.8,4.6],
                                [168.3,165.9,160.1,4.6],
                                [173.0,169.2,154.7,4.4],
                                [168.1,166.7,159.1,4.9],
                                [166.5,164.7,157.8,4.6],
                                [178.9,181.6,161.2,4.5],
                                [170.7,169.3,157.6,4.5],
                                [166.4,166.2,158.8,4.6],
                                [167.4,169.9,158.8,4.6],
                                [162.3,165.1,153.7,5.1],
                                [167.2,167.8,159.3,4.5],
                                [167.2,163.8,160.5,4.5],
                                [165.1,166.3,160.4,4.3],
                                [167.2,166.8,160.4,4.3],
                                [167.2,166.9,155.4,4.5],
                                [165.3,166.1,156.0,4.5],
                                [169.3,167.5,158.3,4.5],
                                [166.2,172.1,162.2,4.9],
                                [171.1,170.5,163.9,4.9],
                                [172.1,172.6,164.4,4.7],
                                [167.0,167.0,159.2,4.4],
                                [172.3,173.0,160.9,4.4],
                                [170.6,171.5,161.5,4.9],
                                [171.5,172.7,166.1,4.9],
                                [176.0,175.6,164.4,4.5],
                                [176.6,177.8,160.5,4.6],
                                [176.3,173.8,165.0,4.6],
                                [175.0,174.4,167.8,4.8],
                                [175.5,174.7,167.0,4.8],
                                [175.8,174.7,166.6,4.8],
                                [173.3,174.1,166.5,4.6],
                                [174.8,172.4,161.4,4.6],
                                [178.2,173.0,166.3,4.4],
                                [173.9,172.9,166.8,4.4],
                                [177.0,171.4,163.0,4.4],
                                [182.1,186.3,164.2,4.6],
                                [177.5,174.9,167.8,4.6],
                                [174.4,176.9,165.2,4.6],
                                [177.7,176.5,166.3,4.6],
                                [177.7,177.0,168.8,4.5],
                                [174.9,175.4,166.7,4.9],
                                [176.0,178.2,171.9,4.9],
                                [174.0,173.1,166.5,4.6],
                                [177.1,173.0,166.9,4.6],
                                [177.2,174.1,170.0,4.5],
                                [175.8,174.0,164.8,4.3],
                                [177.8,174.2,167.2,4.9],
                                [173.9,173.2,163.9,4.2],
                                [175.3,173.5,163.0,5.2],
                                [175.4,174.5,168.1,4.2],
                                [173.9,170.9,168.4,4.2],
                                [176.7,173.3,168.4,4.3],
                                [175.8,174.3,168.6,4.1],
                                [173.5,174.0,167.7,4.1],
                                [173.5,171.2,168.0,4.3],
                                [175.4,173.3,163.7,4.3],
                                [176.8,170.9,162.6,4.1],
                                [171.9,171.3,163.0,4.2],
                                [171.9,168.0,162.6,4.6],
                                [172.2,168.6,164.1,4.1],
                                [168.7,167.7,163.9,4.1],
                                [168.8,169.5,163.7,4.1],
                                [172.0,171.5,165.0,4.3],
                                [165.9,166.4,159.6,4.3],
                                [168.1,165.1,158.8,4.3],
                                [161.7,161.4,153.8,4.3],
                                [159.5,162.7,153.8,4.5],
                                [160.2,156.1,149.6,4.3],
                                [160.9,159.7,154.5,4.3],
                                [155.7,155.1,149.3,4.6],
                                [152.4,151.3,149.3,4.6],
                                [162.7,156.4,154.6,4.4],
                                [157.0,157.4,147.6,4.6],
                                [157.0,161.5,147.6,4.6],
                                [158.7,156.5,151.6,4.4],
                                [158.7,156.5,151.6,4.4],
                                [152.6,151.5,144.6,4.6],
                                [149.4,150.2,143.4,4.4],
                                [152.5,150.2,144.5,4.4],
                                [147.4,146.5,139.4,4.9],
                                [147.4,146.5,139.4,4.9],
                                [149.1,151.5,139.4,4.8],
                                [152.2,153.7,144.1,5.4],
                                [157.6,156.4,150.1,5.4],
                                [157.8,159.8,150.1,5.4],
                                [162.9,166.5,155.4,5.4],
                                [162.9,166.6,155.4,5.4],
                                [165.5,165.2,156.4,5.4],
                                [165.5,165.2,160.4,5.4],
                                [169.9,166.6,160.4,5.7],
                                [169.9,167.1,165.5,5.7],
                                [172.7,170.0,165.7,5.2],
                                [173.9,173.3,165.5,5.9],
                                [178.5,177.1,165.5,5.9],
                                [175.9,175.3,170.7,5.4],
                                [175.9,175.3,165.5,5.4],
                                [178.1,177.3,167.9,5.4],
                                [182.5,181.2,170.9,5.5],
                                [184.0,181.2,170.9,5.5],
                                [179.0,181.1,172.3,5.9],
                                [179.0,177.3,172.3,5.9],
                                [178.9,176.0,165.5,5.6],
                                [178.0,177.0,165.6,6.1],
                                [178.0,177.0,166.6,6.1],
                                [178.2,175.3,167.0,5.6],
                                [178.2,177.6,166.7,5.6],
                                [178.0,174.3,168.3,5.7],
                                [180.4,175.8,167.5,6.1],
                                [180.0,175.8,167.7,6.7],
                                [177.4,175.0,165.6,6.4],
                                [174.9,172.6,162.6,6.4],
                                [174.2,172.2,161.8,6.2],
                                [176.5,173.6,162.9,6.0],
                                [176.5,173.6,162.9,6.0],
                                [174.5,170.7,164.1,6.1],
                                [174.5,170.7,164.1,6.1],
                                [172.3,173.0,163.0,6.1],
                                [176.1,172.1,164.0,6.2],
                                [176.1,172.1,164.0,6.2],
                                [176.6,172.0,164.1,6.0],
                                [176.6,172.0,164.1,5.8],
                                [169.9,168.4,161.5,5.8],
                                [170.4,169.4,161.3,7.2],
                                [175.0,168.0,161.3,6.1],
                                [172.3,170.5,160.8,6.1],
                                [175.2,173.2,160.8,6.1],
                                [174.0,169.2,160.3,6.5],
                                [172.7,171.0,160.8,6.2],
                                [170.1,171.0,160.8,6.2],
                                [170.0,173.3,161.3,6.4],
                                [170.0,168.3,157.6,6.4],
                                [169.6,166.8,162.7,7.2],
                                [171.6,169.7,159.6,6.5],
                                [171.6,169.7,159.6,6.5],
                                [172.2,170.0,161.0,6.7],
                                [172.2,170.0,161.0,6.7],
                                [171.7,171.0,161.8,6.6],
                                [169.7,168.0,157.5,6.7],
                                [169.7,168.0,157.5,6.7],
                                [168.8,168.5,158.8,7.2],
                                [168.8,168.5,158.8,7.2],
                                [165.0,168.2,157.7,6.5],
                                [166.7,165.7,154.2,6.5],
                                [166.7,163.2,152.7,6.5],
                                [164.6,165.2,152.7,6.2],
                                [164.6,165.2,152.8,6.2],
                                [162.6,163.3,153.1,6.2],
                                [165.9,164.6,153.7,6.7],
                                [165.9,164.6,153.7,6.2],
                                [162.7,161.5,153.7,6.3],
                                [159.5,161.5,153.7,6.3],
                                [161.4,162.2,152.7,6.5],
                                [161.4,162.2,152.7,6.5],
                                [160.7,158.2,152.6,6.3],
                                [160.9,159.7,150.9,6.2],
                                [160.9,159.7,150.9,6.2],
                                [160.4,159.6,147.8,6.3],
                                [160.4,159.6,147.8,6.2],
                                [157.5,156.0,147.2,6.3],
                                [157.2,155.9,146.7,6.3],
                                [157.2,155.9,146.7,6.3],
                                [154.4,158.3,146.4,6.2],
                                [154.4,153.3,142.4,6.2],
                                [153.1,152.3,143.6,6.1],
                                [152.5,150.8,143.5,6.2],
                                [152.5,150.8,143.5,6.2],
                                [151.3,150.6,141.7,6.2],
                                [149.3,150.6,141.7,6.2],
                                [151.7,148.1,142.2,6.1],
                                [151.1,149.9,141.4,6.3],
                                [151.1,149.9,141.4,6.3],
                                [147.6,148.7,142.9,6.2],
                                [147.6,148.7,142.9,6.2],
                                [150.2,149.2,140.8,6.4],
                                [149.9,148.5,140.3,6.2],
                                [149.9,148.5,140.3,6.2],
                                [148.0,147.7,139.6,6.2],
                                [148.0,147.7,139.6,6.2],
                                [144.3,145.9,138.9,6.2],
                                [145.7,143.1,135.8,6.1],
                                [145.7,143.1,135.8,6.1],
                                [144.9,144.9,136.3,5.8],
                                [144.9,144.9,136.3,5.8],
                                [143.6,144.5,135.1,6.0],
                                [143.9,144.1,136.1,5.8],
                                [143.9,144.1,132.7,5.8],
                                [142.3,143.5,133.7,6.0],
                                [142.3,143.5,133.7,6.0],
                                [141.8,141.3,132.7,6.2],
                                [140.9,140.2,132.0,5.7],
                                [140.9,140.2,132.0,5.7],
                                [139.3,141.3,132.1,6.1],
                                [139.3,138.0,132.1,6.1],
                                [140.0,139.8,131.0,6.1],
                                [140.0,139.3,131.0,5.6],
                                [140.0,139.3,131.0,5.6],
                                [138.6,138.0,131.8,5.6],
                                [138.6,138.0,131.8,5.6],
                                [138.3,136.2,130.8,6.0],
                                [139.8,138.3,131.6,5.8],
                                [139.8,138.3,127.6,5.8],
                                [137.3,137.3,130.5,6.1],
                                [137.3,137.3,130.5,6.1],
                                [137.9,135.5,128.3,6.2],
                                [138.8,137.3,129.9,5.8],
                                [138.8,137.3,129.9,5.8],
                                [137.9,135.9,128.0,6.3],
                                [137.9,135.9,128.0,6.3],
                                [139.2,136.5,129.8,5.7],
                                [134.5,131.0,123.4,5.3],
                                [129.0,131.0,123.4,5.3],
                                [128.0,127.9,122.3,5.7],
                                [128.0,127.9,117.4,5.7],
                                [124.0,126.8,115.7,5.5],
                                [126.2,126.8,117.9,5.4],
                                [126.2,126.8,117.9,5.4],
                                [122.5,122.8,116.1,5.4],
                                [122.5,122.8,116.1,5.4],
                                [121.4,122.5,116.3,5.1],
                                [121.1,117.6,115.3,5.1],
                                [121.1,122.7,115.3,5.1],
                                [122.7,127.8,117.5,4.9],
                                [129.3,127.8,122.5,4.9],
                                [124.2,127.4,117.5,4.8],
                                [124.6,123.5,118.8,4.8],
                                [129.3,127.8,117.7,4.8],
                                [125.5,125.9,118.9,5.0],
                                [129.4,125.9,122.8,5.4],
                                [130.9,127.9,123.6,5.2],
                                [129.2,127.8,122.1,5.1],
                                [124.4,127.8,117.9,5.1],
                                [129.3,128.6,123.5,5.1],
                                [129.3,128.6,123.1,5.1],
                                [132.1,132.7,124.8,5.3],
                                [131.5,131.2,124.7,5.5],
                                [129.8,131.2,123.2,5.5],
                                [130.6,131.6,128.7,5.2],
                                [135.3,133.5,128.7,5.2],
                                [130.3,132.3,126.4,4.7],
                                [134.0,134.9,129.3,4.6],
                                [134.7,134.9,129.3,4.6],
                                [135.8,136.5,130.8,4.6],
                                [142.8,147.7,131.1,5.3],
                                [138.0,136.7,130.8,5.0],
                                [142.4,142.9,135.9,4.7],
                                [142.4,142.9,135.9,4.7],
                                [146.2,145.0,136.8,4.9],
                                [147.0,144.5,136.6,4.3],
                                [150.3,145.8,137.3,4.6],
                                [154.1,162.6,140.8,4.7],
                                [149.6,148.6,140.4,4.7],
                                [155.6,151.3,146.1,5.4],
                                [156.2,149.8,144.8,4.4],
                                [150.6,146.9,138.0,4.4],
                                [153.7,149.3,141.8,4.1],
                                [150.6,149.3,144.3,4.1],
                                [151.4,148.0,144.0,4.3],
                                [152.0,151.0,144.8,4.3],
                                [155.9,151.2,149.7,4.6],
                                [155.9,152.9,147.4,4.6],
                                [157.2,153.1,148.3,4.3],
                                [157.6,152.9,143.3,4.4],
                                [156.5,151.3,145.3,4.2],
                                [157.2,151.8,145.0,4.4],
                                [158.9,157.5,150.1,4.6],
                                [153.9,152.9,145.8,4.6],
                                [162.3,159.2,151.3,4.5],
                                [157.3,154.1,146.3,4.5],
                                [160.2,155.2,147.9,4.7],
                                [159.8,154.8,146.0,4.7],
                                [166.8,171.8,151.2,4.5],
                                [157.9,155.8,149.5,4.4],
                                [160.6,156.6,148.8,4.4],
                                [162.5,155.7,152.2,4.2],
                                [162.6,161.5,152.9,4.2],
                                [163.9,159.7,153.7,4.5],
                                [163.7,159.8,151.5,4.6],
                                [162.3,159.6,155.2,4.6],
                                [163.2,159.6,155.1,5.0],
                                [163.8,165.1,150.7,5.0],
                                [164.9,156.9,154.3,4.4],
                                [162.3,163.0,154.5,4.5],
                                [163.7,158.5,153.1,4.5],
                                [163.7,164.9,152.0,4.4],
                                [165.8,163.9,153.3,4.4],
                                [168.9,162.2,154.1,4.3],
                                [167.2,164.8,149.5,4.6],
                                [172.5,174.7,156.2,4.6],
                                [160.9,160.1,151.4,4.5],
                                [166.7,166.3,156.5,4.5],
                                [165.6,164.5,156.2,4.5],
                                [166.2,165.5,154.6,4.4],
                                [168.0,164.8,155.4,4.4],
                                [166.7,160.5,155.8,4.4],
                                [162.6,160.5,155.9,4.4],
                                [162.5,160.5,151.1,4.5],
                                [167.0,165.3,152.2,4.5],
                                [166.2,161.5,153.0,4.5],
                                [161.1,161.5,152.3,4.8],
                                [166.1,161.5,152.3,4.6],
                                [166.1,163.7,153.9,4.6],
                                [164.4,163.7,148.9,4.3],
                                [166.0,163.7,154.2,4.3],
                                [165.7,165.5,154.4,4.3],
                                [165.7,162.0,155.2,4.3],
                                [164.6,162.0,155.2,4.2],
                                [165.6,160.8,154.3,4.0],
                                [165.9,163.4,154.3,4.0],
                                [165.9,162.4,153.9,4.3],
                                [162.6,158.2,153.5,4.3],
                                [163.3,158.7,153.5,4.3],
                                [168.9,164.6,153.9,4.7],
                                [168.5,164.3,155.4,4.7],
                                [166.8,165.0,154.7,4.7],
                                [169.9,168.3,157.5,4.7],
                                [167.3,166.6,157.2,4.4],
                                [166.9,165.2,159.8,4.5],
                                [169.1,165.0,156.7,4.5],
                                [158.8,158.8,151.1,4.7],
                                [172.8,177.7,155.4,4.7],
                                [167.8,163.2,156.7,4.6],
                                [167.7,165.5,156.3,4.3],
                                [171.5,168.7,158.9,4.3],
                                [163.8,160.6,156.3,4.7],
                                [163.3,163.1,154.0,4.7],
                                [163.4,160.8,154.9,4.7],
                                [165.3,158.9,155.7,4.6],
                                [160.3,161.7,150.2,4.6],
                                [165.5,160.9,155.2,4.4],
                                [161.3,162.0,155.2,4.4],
                                [161.3,160.8,150.1,4.3],
                                [161.3,159.0,149.5,4.3],
                                [157.7,161.8,149.0,4.3],
                                [162.1,159.6,149.3,4.1],
                                [164.8,161.1,149.9,4.1],
                                [159.7,160.1,150.6,4.4],
                                [162.2,159.9,153.4,4.5],
                                [160.6,159.4,151.3,4.7],
                                [165.3,162.2,153.3,4.3],
                                [160.4,157.2,153.3,4.2],
                                [157.4,155.8,148.7,4.5],
                                [160.8,160.9,154.1,4.4],
                                [168.0,172.3,150.4,4.4],
                                [162.7,159.5,151.1,4.2],
                                [161.7,160.4,156.9,4.2],
                                [161.7,156.5,151.9,5.0],
                                [162.6,161.9,154.9,4.6],
                                [162.6,156.7,152.0,4.6],
                                [159.2,156.9,151.3,4.3],
                                [156.7,156.9,152.1,4.3],
                                [156.7,157.7,152.0,4.0],
                                [158.9,157.5,152.2,4.5],
                                [158.0,152.4,150.3,4.2],
                                [152.2,152.0,146.5,4.1],
                                [158.8,152.7,146.5,3.5],
                                [153.9,153.3,147.1,3.8],
                                [148.9,153.3,142.0,3.8],
                                [154.0,148.3,142.3,3.9],
                                [148.9,150.0,140.4,4.3],
                                [153.9,148.5,140.4,4.3],
                                [148.9,149.3,143.3,4.4],
                                [148.9,148.9,143.3,4.4],
                                [148.8,147.1,140.4,4.7],
                                [149.6,149.2,140.1,4.2],
                                [149.6,148.8,137.3,4.2],
                                [144.1,148.8,137.9,4.6],
                                [149.1,149.0,137.9,5.2],
                                [149.1,149.4,137.9,4.1],
                                [144.9,146.1,138.3,4.2],
                                [149.2,149.8,138.3,4.2],
                                [149.0,149.0,140.8,4.7],
                                [154.6,160.2,148.2,5.1],
                                [156.0,155.5,148.4,4.8],
                                [160.4,159.3,153.4,4.8],
                                [170.3,170.7,158.5,4.8],
                                [168.0,170.8,158.4,4.9],
                                [170.2,170.8,158.4,4.9],
                                [168.9,171.1,158.8,4.9],
                                [170.7,169.0,162.7,4.7],
                                [170.7,169.0,158.7,4.7],
                                [170.8,169.0,160.3,5.5],
                                [175.3,175.8,163.9,5.5],
                                [175.4,173.3,161.2,6.1],
                                [174.2,175.4,160.8,5.9],
                                [174.2,170.8,163.8,5.9],
                                [173.1,170.0,162.4,5.5],
                                [173.1,175.9,162.4,5.5],
                                [170.4,170.9,158.8,5.4],
                                [170.4,170.3,159.9,5.9],
                                [170.8,171.0,159.9,5.9],
                                [171.7,165.9,161.3,5.6],
                                [171.7,165.9,164.2,5.6],
                                [170.1,165.6,159.1,5.3],
                                [170.7,165.6,158.0,5.4],
                                [170.7,165.6,158.0,5.4],
                                [170.3,166.0,160.6,5.7],
                                [170.3,166.0,160.6,5.7],
                                [175.6,170.4,164.1,5.3],
                                [170.4,166.7,156.9,5.4],
                                [170.4,166.7,156.9,5.4],
                                [171.8,168.7,160.2,5.4],
                                [171.8,166.1,160.2,5.4],
                                [165.7,161.1,153.7,5.3],
                                [167.6,160.7,153.8,5.4],
                                [165.7,160.7,153.8,6.1],
                                [173.0,166.5,157.8,5.1],
                                [173.0,166.5,154.1,6.1],
                                [168.9,161.7,154.8,5.4],
                                [165.4,161.0,150.0,5.2],
                                [165.4,161.0,154.1,5.2],
                                [167.1,162.3,153.7,5.5],
                                [167.1,162.3,153.7,5.2],
                                [163.5,160.2,154.1,5.3],
                                [167.4,162.7,155.3,5.3],
                                [167.4,162.7,155.3,5.3],
                                [165.0,159.5,152.6,5.3],
                                [165.0,159.5,152.6,5.3],
                                [164.8,159.6,150.8,5.2],
                                [164.3,161.1,151.8,5.3],
                                [164.3,161.1,151.8,5.3],
                                [165.7,161.2,153.9,5.1],
                                [165.9,161.8,153.9,5.1],
                                [167.3,163.3,153.9,5.1],
                                [167.7,162.9,155.7,5.5],
                                [160.9,162.9,155.7,5.5],
                                [165.9,161.2,153.7,6.3],
                                [165.9,161.2,153.7,6.3],
                                [160.9,156.7,149.0,5.3],
                                [166.0,162.7,152.9,5.9],
                                [160.9,162.7,152.9,6.3],
                                [163.5,162.3,153.1,5.5],
                                [163.5,156.8,149.1,5.5],
                                [155.7,157.3,148.8,6.0],
                                [160.0,157.0,149.3,5.6],
                                [155.8,157.0,149.3,6.4],
                                [159.0,157.0,147.7,5.7],
                                [156.2,151.8,147.7,5.7],
                                [161.5,155.7,147.9,5.9],
                                [157.5,154.0,149.0,5.8],
                                [157.5,154.0,148.5,5.8],
                                [157.9,154.7,148.9,6.3],
                                [157.9,151.6,143.4,6.3],
                                [153.6,150.6,143.7,5.7],
                                [152.7,150.5,143.7,5.9],
                                [155.5,150.5,143.7,5.9],
                                [152.7,149.9,143.9,6.5],
                                [151.5,149.9,143.9,5.5],
                                [153.8,150.0,143.1,5.5],
                                [149.8,147.8,140.3,5.5],
                                [150.1,147.8,140.3,5.5],
                                [149.5,147.6,138.9,5.7],
                                [149.5,147.6,138.9,5.7],
                                [148.4,146.1,137.4,5.5],
                                [147.3,143.5,136.2,5.8],
                                [147.3,146.2,136.2,5.8],
                                [148.3,145.1,138.0,6.7],
                                [148.3,145.1,138.0,6.7],
                                [144.7,143.5,135.4,5.5],
                                [146.9,142.9,136.3,5.7],
                                [146.9,142.9,138.3,5.7],
                                [145.7,141.7,135.6,5.6],
                                [145.7,141.2,138.4,5.6],
                                [145.7,141.8,133.3,5.4],
                                [144.3,140.8,135.2,5.5],
                                [144.3,140.8,135.2,5.5],
                                [143.6,140.6,133.4,5.6],
                                [143.6,140.6,133.4,5.6],
                                [140.1,139.7,131.7,5.6],
                                [143.5,140.8,132.9,5.5],
                                [143.5,140.8,132.9,5.5],
                                [140.5,136.1,131.7,5.5],
                                [144.5,141.2,131.7,5.5],
                                [141.2,136.9,132.4,5.6],
                                [144.7,136.9,132.4,5.6],
                                [141.8,138.3,130.9,5.6],
                                [141.5,138.0,130.6,5.7],
                                [141.5,138.0,130.6,5.7],
                                [140.7,137.4,128.8,5.6],
                                [140.7,137.4,128.8,5.6],
                                [138.0,135.2,128.3,5.7],
                                [138.8,137.1,128.6,5.6],
                                [138.8,137.1,128.6,5.6],
                                [138.5,136.3,127.8,5.4],
                                [134.6,131.0,127.8,5.4],
                                [136.6,136.1,127.9,5.6],
                                [138.9,134.0,127.9,5.4],
                                [138.9,134.0,127.9,5.4],
                                [136.0,132.0,128.2,5.4],
                                [140.5,132.0,128.2,5.4],
                                [139.1,133.9,127.5,5.4],
                                [137.4,135.3,126.8,5.6],
                                [135.5,135.3,126.8,5.6],
                                [135.4,132.6,126.8,5.6],
                                [135.4,132.6,123.2,5.6],
                                [136.9,133.9,123.2,5.5],
                                [134.8,130.2,124.9,5.4],
                                [134.8,130.2,123.7,5.4],
                                [129.7,126.3,123.7,5.4],
                                [129.7,126.3,118.4,5.4],
                                [125.0,125.2,120.4,5.8],
                                [123.4,122.9,116.8,4.8],
                                [123.4,121.3,116.8,5.8],
                                [123.4,122.0,116.6,5.1],
                                [123.4,122.0,116.6,5.1],
                                [124.4,122.5,112.6,6.0],
                                [119.3,120.7,112.5,5.4],
                                [119.3,116.3,112.5,5.0],
                                [118.1,116.4,111.4,5.0],
                                [118.1,116.4,111.4,5.0],
                                [117.4,116.2,111.0,4.3],
                                [121.8,121.0,114.6,4.7],
                                [119.7,121.0,112.4,4.7],
                                [120.4,117.1,113.0,4.6],
                                [120.4,121.5,113.0,4.6],
                                [118.8,116.5,111.7,4.7],
                                [121.5,121.0,114.5,4.9],
                                [124.7,121.0,114.5,4.9],
                                [121.3,121.5,113.8,4.6],
                                [125.0,121.5,117.6,4.6],
                                [125.4,124.2,117.7,5.0],
                                [122.3,120.8,115.5,4.6],
                                [122.3,120.8,115.5,4.6],
                                [130.1,126.6,122.7,5.1],
                                [125.0,126.6,122.7,5.0],
                                [124.9,121.6,117.7,5.0],
                                [123.0,123.1,116.6,4.8],
                                [123.0,123.1,116.6,4.8],
                                [125.5,122.7,117.2,4.8],
                                [125.5,122.7,117.2,4.8],
                                [125.6,126.9,123.1,5.3],
                                [124.0,124.6,118.2,4.7],
                                [126.1,124.1,118.0,4.7],
                                [126.6,126.6,118.0,4.9],
                                [126.9,126.3,119.1,4.9],
                                [129.3,127.4,121.7,4.7],
                                [127.5,128.4,119.4,4.7],
                                [129.2,133.1,124.4,4.7],
                                [130.8,128.1,122.2,4.2],
                                [127.8,128.7,124.3,4.2],
                                [132.9,134.1,125.4,4.4],
                                [131.8,130.6,124.5,4.0],
                                [131.8,130.6,124.5,4.0],
                                [133.6,131.4,125.4,4.2],
                                [132.7,131.1,126.1,4.2],
                                [129.9,130.3,121.1,4.5],
                                [128.9,128.2,123.3,4.3],
                                [130.7,131.7,126.6,4.3],
                                [136.2,131.7,125.9,4.2],
                                [136.4,132.8,125.9,4.2],
                                [136.3,130.5,125.5,4.1],
                                [131.7,129.5,121.9,4.3],
                                [136.3,129.5,126.7,4.3],
                                [136.4,136.0,126.9,4.4],
                                [141.4,136.0,132.0,4.4],
                                [136.4,139.1,126.8,4.4],
                                [135.3,134.3,129.8,4.2],
                                [135.3,134.3,129.8,4.2],
                                [134.4,132.1,124.6,4.2],
                                [134.4,138.0,126.9,4.2],
                                [136.1,135.7,127.8,4.2],
                                [132.7,134.2,125.7,4.3],
                                [136.6,133.0,125.7,4.3],
                                [134.7,132.8,126.0,4.1],
                                [131.6,132.8,126.0,4.1],
                                [138.7,137.6,127.2,4.5],
                                [135.5,136.2,127.3,4.3],
                                [142.1,136.2,132.4,4.3],
                                [141.7,138.2,133.3,4.0],
                                [137.0,138.2,127.2,4.0],
                                [138.5,137.3,127.2,4.2],
                                [138.5,137.3,132.5,4.2],
                                [137.1,135.5,132.5,4.4],
                                [135.2,134.3,132.5,4.0],
                                [137.0,134.3,132.5,4.0],
                                [138.8,137.5,131.9,4.3],
                                [138.8,143.2,131.9,4.3],
                                [136.9,138.1,131.0,4.1],
                                [138.3,137.5,131.0,4.1],
                                [142.2,138.2,131.0,4.1],
                                [141.1,138.1,133.8,4.1],
                                [141.1,143.5,132.5,4.1],
                                [144.0,143.1,137.6,4.6],
                                [142.6,142.9,133.3,4.3],
                                [142.0,142.9,132.8,4.3],
                                [140.0,141.4,132.7,4.5],
                                [147.1,143.1,132.7,4.5],
                                [142.0,141.3,134.4,4.5],
                                [142.1,142.1,133.3,4.3],
                                [142.1,142.1,133.3,4.3],
                                [147.0,144.5,137.9,4.5],
                                [147.0,144.5,137.9,4.5],
                                [145.7,145.7,138.2,4.5],
                                [143.4,142.7,132.9,4.2],
                                [143.4,138.2,132.9,4.2],
                                [142.5,140.4,131.8,4.5],
                                [142.5,143.2,138.0,4.5],
                                [141.5,142.6,132.9,4.6],
                                [142.4,142.3,133.2,4.5],
                                [142.4,142.3,133.2,4.5],
                                [141.2,143.1,132.3,4.7],
                                [141.2,143.1,132.3,4.7],
                                [141.9,141.6,131.9,4.6],
                                [141.0,143.5,133.0,4.7],
                                [141.0,138.6,133.0,4.7],
                                [141.1,144.2,133.0,4.7],
                                [141.1,144.2,133.0,4.7],
                                [142.3,139.0,134.9,4.5],
                                [144.3,145.8,134.8,4.7],
                                [141.7,145.8,132.9,4.7],
                                [140.0,142.0,133.4,4.8],
                                [140.0,142.0,138.5,4.8],
                                [141.7,144.1,133.5,4.1],
                                [140.1,141.3,131.4,4.2],
                                [142.6,144.7,135.2,4.4],
                                [144.0,143.8,135.6,4.3],
                                [144.0,143.8,135.6,4.3],
                                [142.4,142.9,135.9,4.5],
                                [143.5,145.0,136.5,4.1],
                                [142.8,149.8,134.9,4.1],
                                [148.0,145.7,140.6,4.6],
                                [142.8,145.7,135.6,4.6],
                                [143.0,144.9,135.5,4.6],
                                [141.7,141.7,134.2,4.2],
                                [138.0,139.5,130.5,4.2],
                                [139.8,139.7,133.3,4.6],
                                [139.8,144.6,133.3,4.6],
                                [141.1,139.5,134.9,4.1],
                                [142.8,144.6,136.2,4.0],
                                [142.8,139.4,136.2,4.0],
                                [141.1,138.4,133.8,4.2],
                                [143.2,138.4,133.8,4.2],
                                [138.8,138.5,131.6,4.3],
                                [141.9,144.5,135.2,4.4],
                                [141.9,139.5,135.2,4.4],
                                [140.1,139.7,134.6,4.5],
                                [140.1,139.7,134.6,4.5],
                                [143.4,139.8,134.4,4.5],
                                [142.3,139.8,135.0,4.0],
                                [138.0,140.3,130.5,4.0],
                                [142.9,140.2,130.4,4.2],
                                [138.3,135.2,135.5,4.2],
                                [140.2,137.0,130.5,4.2],
                                [140.2,140.6,131.6,4.7],
                                [140.2,137.3,130.9,4.7],
                                [137.7,138.7,131.1,4.4],
                                [137.7,138.7,131.1,4.4],
                                [139.8,137.5,131.1,4.5],
                                [141.2,140.5,133.8,4.5],
                                [141.2,137.5,131.4,4.5],
                                [142.7,141.4,131.2,4.4],
                                [142.7,141.4,136.4,4.4],
                                [143.8,138.1,131.2,4.4],
                                [142.9,140.5,132.0,4.4],
                                [142.9,140.5,132.0,4.4],
                                [138.7,143.7,130.6,4.3],
                                [143.0,143.9,136.4,4.3],
                                [141.8,138.7,134.7,4.6],
                                [144.4,145.3,137.5,5.1],
                                [148.3,149.2,136.4,5.6],
                                [153.3,147.3,139.7,5.3],
                                [158.6,154.3,152.2,5.3],
                                [157.8,155.3,148.4,5.4],
                                [162.9,158.1,150.8,5.2],
                                [162.9,158.1,150.8,5.2],
                                [163.8,158.8,152.3,5.3],
                                [163.8,159.6,156.7,5.3],
                                [166.1,162.5,156.0,5.3],
                                [165.9,158.9,151.2,5.2],
                                [169.0,160.4,156.0,5.2],
                                [166.0,165.5,153.7,5.3],
                                [163.6,160.4,150.6,5.3],
                                [164.5,160.2,150.7,5.2],
                                [165.6,162.8,155.7,5.3],
                                [169.1,165.6,161.0,5.3],
                                [168.2,161.7,156.8,5.3],
                                [169.0,162.0,156.8,5.3],
                                [164.4,157.0,154.3,5.2],
                                [167.2,164.3,158.8,5.8],
                                [167.2,164.3,158.8,5.8],
                                [169.0,163.0,156.4,5.9],
                                [163.9,163.0,156.9,5.9],
                                [168.9,162.1,156.0,5.7],
                                [165.3,161.5,153.9,5.8],
                                [165.3,161.5,157.0,5.8],
                                [168.9,162.9,154.9,5.9],
                                [163.9,162.9,152.0,5.6],
                                [168.2,164.6,157.8,5.5],
                                [164.0,164.6,157.8,5.5],
                                [169.0,160.2,156.5,5.5],
                                [166.3,159.8,151.9,5.6],
                                [158.7,159.8,152.0,5.6],
                                [163.8,156.7,152.5,5.7],
                                [163.8,156.7,152.5,5.7],
                                [162.7,157.1,151.7,5.4],
                                [161.9,157.6,152.4,5.5],
                                [161.9,157.6,147.4,5.5],
                                [163.7,157.9,152.8,5.7],
                                [163.7,157.9,147.8,5.7],
                                [162.5,158.3,149.3,5.8],
                                [162.9,158.1,152.6,5.4],
                                [162.9,158.1,152.6,5.4],
                                [162.3,162.5,151.9,5.6],
                                [163.8,157.4,152.8,5.6],
                                [162.0,162.8,152.1,5.6],
                                [165.2,162.9,151.6,6.3],
                                [165.2,162.6,152.8,6.3],
                                [163.6,157.5,152.8,5.7],
                                [163.6,162.6,152.8,5.7],
                                [164.2,160.0,150.7,5.6],
                                [164.0,162.3,152.0,5.9],
                                [164.0,162.3,152.0,5.9],
                                [162.6,162.7,153.2,6.0],
                                [162.6,162.7,153.2,6.0],
                                [162.1,163.2,155.5,6.0],
                                [163.1,163.2,153.3,5.9],
                                [163.1,163.2,153.3,5.9],
                                [164.2,162.9,153.9,6.4],
                                [164.2,162.9,153.9,6.4],
                                [158.4,163.8,154.2,6.4],
                                [160.8,157.4,150.3,5.7],
                                [160.8,162.6,150.3,6.6],
                                [162.5,163.1,154.1,6.3],
                                [158.4,162.5,152.6,7.7],
                                [158.5,162.3,147.5,6.7],
                                [160.9,160.6,152.6,6.1],
                                [158.7,160.6,152.6,6.1],
                                [158.5,159.2,148.8,6.4],
                                [158.5,157.3,148.8,6.4],
                                [160.5,157.3,148.5,6.3],
                                [157.0,154.8,146.5,6.6],
                                [157.0,154.8,146.5,6.6],
                                [156.9,156.7,148.1,6.8],
                                [153.6,156.7,148.1,6.8],
                                [155.7,152.2,142.4,6.3],
                                [153.4,152.7,142.0,6.7],
                                [153.4,152.7,142.0,6.7],
                                [154.5,153.7,145.2,6.1],
                                [154.5,153.7,145.2,6.1],
                                [155.9,152.5,142.4,6.1],
                                [154.1,153.5,142.4,6.2],
                                [154.1,153.5,142.4,6.2],
                                [153.2,153.4,142.8,6.3],
                                [153.2,147.4,137.4,6.3],
                                [148.2,149.9,137.7,6.4],
                                [149.6,148.5,138.1,6.2],
                                [149.6,148.5,138.1,6.2],
                                [148.6,146.4,136.2,6.1],
                                [148.6,146.4,136.2,6.1],
                                [147.9,146.5,135.6,6.2],
                                [148.0,145.5,137.3,6.1],
                                [148.0,145.5,137.3,6.1],
                                [147.0,146.8,136.2,6.2],
                                [147.0,146.8,136.2,6.2],
                                [147.5,146.0,135.0,6.4],
                                [147.1,145.2,135.9,6.2],
                                [143.2,145.2,132.5,6.2],
                                [145.4,143.8,133.5,6.2],
                                [145.4,143.8,133.5,6.2],
                                [144.6,143.0,131.7,6.1],
                                [143.6,142.1,132.9,6.1],
                                [143.6,142.1,132.9,6.1],
                                [143.6,142.2,131.6,6.2],
                                [143.6,142.2,131.6,6.2],
                                [143.6,142.3,132.2,6.2],
                                [141.4,141.0,131.4,6.0],
                                [141.4,141.0,131.4,6.0],
                                [143.7,141.8,132.8,6.2],
                                [143.7,141.8,132.8,6.2],
                                [141.8,139.4,130.7,6.1],
                                [139.6,138.0,130.6,6.0],
                                [143.1,142.4,130.6,6.7],
                                [140.4,139.1,131.2,5.6],
                                [140.4,137.4,131.2,5.6],
                                [138.0,138.9,127.4,5.8],
                                [143.2,138.8,132.4,5.6],
                                [138.1,138.8,132.4,5.6],
                                [141.2,137.5,130.3,5.6],
                                [137.7,137.5,127.3,5.6],
                                [138.1,138.4,128.2,5.7],
                                [136.7,135.2,127.3,5.8],
                                [136.7,135.2,127.3,5.6],
                                [137.8,136.5,128.6,5.5],
                                [137.8,136.5,128.6,5.5],
                                [137.7,135.1,127.5,5.6],
                                [136.1,134.5,126.8,5.7],
                                [132.7,134.5,122.3,5.7],
                                [127.8,127.2,119.3,5.7],
                                [127.8,127.2,117.2,5.7],
                                [122.5,127.2,117.9,5.8],
                                [125.9,125.7,116.4,5.6],
                                [125.9,122.1,116.4,5.6],
                                [121.3,122.1,114.8,5.3],
                                [121.3,122.1,114.8,5.3],
                                [123.8,123.2,115.6,5.2],
                                [120.7,120.3,113.3,5.1],
                                [120.7,120.3,113.3,5.1],
                                [117.5,117.0,109.8,5.3],
                                [117.5,117.0,109.8,5.6],
                                [116.8,119.5,111.1,5.4],
                                [116.0,117.6,109.7,5.5],
                                [116.0,117.7,109.7,5.5],
                                [118.1,117.7,110.5,5.1],
                                [118.1,117.7,110.5,5.1],
                                [121.2,120.5,114.9,5.3],
                                [118.4,119.1,113.6,5.0],
                                [118.4,119.1,113.6,5.0],
                                [116.2,117.0,110.1,5.4],
                                [116.2,117.0,110.1,5.4],
                                [115.7,117.5,110.8,5.3],
                                [115.7,117.5,110.8,5.3],
                                [117.3,118.5,110.4,5.2],
                                [116.9,123.0,113.2,5.3],
                                [122.8,123.0,117.8,5.3],
                                [117.3,120.8,113.8,5.1],
                                [117.2,120.8,112.7,5.1],
                                [122.3,123.2,117.8,4.6],
                                [120.9,122.3,112.7,4.6],
                                [120.9,123.0,117.9,4.6],
                                [120.4,121.6,118.0,4.4],
                                [122.1,121.6,118.0,4.4],
                                [121.2,123.0,112.9,4.5],
                                [123.0,123.6,116.7,4.5],
                                [123.0,123.6,116.7,4.5],
                                [121.7,122.9,116.8,4.9],
                                [121.7,122.9,113.0,4.9],
                                [122.1,122.9,118.1,4.7],
                                [121.6,121.9,117.8,4.8],
                                [121.6,121.9,117.8,4.8],
                                [124.5,127.3,117.6,4.7],
                                [124.5,127.3,117.6,4.7],
                                [122.0,122.8,118.0,4.7],
                                [123.2,126.3,114.2,4.9],
                                [127.4,122.5,117.8,4.7],
                                [126.2,126.6,117.2,4.8],
                                [122.3,122.5,112.8,4.8],
                                [120.8,121.3,117.9,4.6],
                                [122.3,120.7,114.2,4.6],
                                [122.3,117.5,114.2,4.6],
                                [122.8,120.1,113.1,4.4],
                                [122.8,122.6,113.1,4.4],
                                [119.8,122.5,111.5,4.0],
                                [122.9,121.3,112.7,4.8],
                                [122.9,116.5,112.7,4.8],
                                [122.3,116.4,114.0,4.3],
                                [122.3,116.4,114.0,4.3],
                                [121.9,121.5,112.9,4.2],
                                [125.0,121.8,116.8,4.3],
                                [121.8,121.8,113.4,4.3],
                                [126.9,121.5,116.9,4.4],
                                [126.9,121.5,113.4,4.4],
                                [121.9,121.8,113.4,4.3],
                                [119.5,120.0,111.6,4.3],
                                [119.5,120.0,108.3,4.3],
                                [121.7,119.4,114.3,4.2],
                                [121.7,117.0,114.3,4.2],
                                [121.7,122.1,114.8,4.9],
                                [119.8,119.2,111.4,4.9],
                                [116.8,117.1,111.4,4.9],
                                [119.4,119.8,112.2,4.7],
                                [116.6,116.9,112.2,4.7],
                                [119.3,118.5,113.7,4.7],
                                [123.7,122.4,114.3,4.7],
                                [123.7,122.4,114.3,4.7],
                                [121.0,119.8,113.4,5.0],
                                [121.0,119.8,113.4,5.0],
                                [121.9,120.0,111.7,4.4],
                                [121.8,119.7,113.3,4.5],
                                [121.8,119.7,113.3,4.5],
                                [122.9,121.2,114.0,4.2],
                                [122.9,121.2,114.0,4.2],
                                [122.2,121.6,114.4,4.4],
                                [124.0,122.5,115.9,5.2],
                                [121.9,122.5,115.9,5.2],
                                [123.1,122.2,113.1,4.8],
                                [123.1,122.2,113.1,4.8],
                                [120.9,121.7,111.8,4.6],
                                [119.5,120.4,112.0,4.8],
                                [122.0,120.4,112.0,4.8],
                                [122.8,121.6,113.0,4.8],
                                [127.0,121.6,113.0,4.8],
                                [125.7,124.6,115.3,5.0],
                                [124.3,123.0,116.2,4.7],
                                [124.3,123.0,116.2,4.7],
                                [124.2,121.3,114.4,4.7],
                                [121.9,121.3,113.7,4.7],
                                [123.0,119.6,114.2,4.7],
                                [125.3,122.1,116.4,4.7],
                                [125.3,122.1,116.4,4.7],
                                [122.8,121.7,113.9,4.3],
                                [122.8,121.7,114.2,4.3],
                                [125.3,122.6,116.7,4.5],
                                [125.3,123.0,118.0,5.1],
                                [122.3,123.0,114.5,5.1],
                                [122.3,121.5,118.0,4.6],
                                [122.3,121.5,119.0,4.6],
                                [124.4,122.2,117.9,5.7],
                                [122.2,122.2,117.9,5.7],
                                [123.5,122.4,117.7,5.0],
                                [124.0,121.9,117.1,4.8],
                                [124.0,121.9,117.1,4.8],
                                [124.2,122.3,119.0,4.9],
                                [127.2,122.3,119.0,4.9],
                                [124.0,120.6,117.5,4.9],
                                [129.1,127.4,118.7,5.0],
                                [129.1,121.8,118.7,5.0],
                                [132.5,126.7,124.0,4.9],
                                [132.5,126.7,124.0,4.9],
                                [127.5,127.1,118.9,4.5],
                                [127.4,124.9,121.5,4.9],
                                [127.4,121.6,121.5,4.9],
                                [126.8,121.3,119.7,4.7],
                                [126.8,126.4,119.7,4.7],
                                [131.8,124.2,124.3,4.9],
                                [126.6,125.3,119.9,4.8],
                                [131.7,125.3,124.1,4.8],
                                [130.5,126.6,124.0,4.6],
                                [126.7,126.6,119.1,4.6],
                                [126.8,126.4,118.7,4.6],
                                [127.6,123.3,118.8,4.7],
                                [127.6,126.8,118.8,4.7],
                                [126.9,126.8,124.2,4.7],
                                [131.7,126.8,124.2,4.7],
                                [131.6,125.6,121.9,4.7],
                                [129.1,126.5,121.8,4.4],
                                [129.1,126.5,121.8,4.4],
                                [129.0,126.4,122.7,4.7],
                                [129.0,126.4,122.7,4.7],
                                [129.8,126.9,122.7,4.5],
                                [132.0,130.5,126.9,4.6],
                                [135.8,132.0,129.4,4.6],
                                [136.0,129.7,129.3,4.2],
                                [136.0,129.7,129.3,4.2],
                                [136.1,130.2,123.6,4.3],
                                [140.8,136.5,128.4,4.1],
                                [140.8,135.5,128.4,4.1],
                                [141.4,135.3,128.9,4.5],
                                [141.4,135.7,134.0,4.5],
                                [141.3,140.3,135.4,4.5],
                                [141.3,142.7,132.0,4.7],
                                [141.4,142.7,132.0,4.7],
                                [146.9,144.4,139.1,4.7],
                                [146.9,151.1,139.1,4.7],
                                [146.0,149.2,137.6,4.8],
                                [149.5,149.3,139.2,4.7],
                                [149.5,149.3,139.2,4.7],
                                [151.4,151.1,144.2,4.8],
                                [155.9,157.5,150.0,4.8],
                                [161.1,158.2,150.0,4.4],
                                [166.1,162.5,155.1,4.7],
                                [166.1,162.5,155.1,4.7],
                                [167.5,166.4,155.8,4.7],
                                [168.1,163.9,157.0,4.7],
                                [167.1,163.4,155.6,4.7],
                                [168.6,165.5,157.0,4.7],
                                [173.1,165.5,162.0,4.7],
                                [173.0,168.1,160.8,4.9],
                                [173.0,168.1,160.8,4.9],
                                [168.1,168.4,159.0,5.1],
                                [167.0,163.3,157.3,5.2],
                                [167.0,168.4,157.3,5.2],
                                [173.5,167.8,162.2,5.2],
                                [173.6,167.8,162.2,5.2],
                                [173.8,168.5,160.8,5.2],
                                [171.7,168.2,160.6,5.3],
                                [169.2,168.2,157.1,5.3],
                                [172.9,169.9,163.1,5.2],
                                [168.4,163.3,158.4,5.2],
                                [169.9,168.4,158.8,5.4],
                                [170.2,167.0,162.4,5.4],
                                [168.4,167.0,157.2,5.4],
                                [172.0,169.0,159.5,5.5],
                                [172.0,169.0,162.5,5.5],
                                [169.9,163.4,157.4,5.6],
                                [173.2,169.3,162.0,5.6],
                                [173.2,168.4,162.0,5.6],
                                [169.1,168.6,157.3,5.7],
                                [169.1,163.4,157.3,5.7],
                                [168.7,169.4,156.2,5.7],
                                [171.0,166.8,156.2,5.5],
                                [171.0,166.8,156.2,5.5],
                                [169.7,165.8,157.7,5.9],
                                [163.7,165.8,151.8,5.9],
                                [168.6,162.0,156.9,5.9],
                                [168.6,162.0,156.9,5.9],
                                [167.3,164.4,156.2,6.1],
                                [166.0,163.7,156.1,6.0],
                                [166.0,163.7,156.9,6.0],
                                [168.9,163.9,157.1,5.9],
                                [169.1,166.9,158.3,6.1],
                                [169.1,166.9,158.3,6.1],
                                [169.4,164.6,158.2,6.2],
                                [169.4,169.6,158.2,6.2],
                                [170.4,165.5,158.4,6.1],
                                [170.4,165.5,158.4,6.1],
                                [168.4,164.5,156.9,6.3],
                                [168.6,166.8,160.6,5.9],
                                [168.6,164.7,156.2,5.9],
                                [170.9,166.1,160.8,5.9],
                                [170.9,166.1,156.2,6.3],
                                [167.9,164.9,156.4,6.3],
                                [166.2,164.7,157.6,6.4],
                                [166.2,164.7,161.5,6.4],
                                [166.2,163.7,157.5,7.3],
                                [166.2,163.7,157.5,6.3],
                                [163.5,162.5,155.5,6.3],
                                [167.0,163.2,157.5,6.6],
                                [163.7,163.2,157.5,6.6],
                                [163.4,164.8,154.7,6.5],
                                [163.4,159.8,154.7,6.5],
                                [163.2,159.8,153.5,6.3],
                                [163.1,158.1,152.5,6.4],
                                [163.1,158.1,151.2,6.4],], columns=['IA', 'IB', 'IV', 'IN'])

 

    
    dummy= pandas.concat([dummy_week] * qty_weeks, ignore_index=True)
    
    end_date_dt = start_date_dt  + dt.timedelta(days=qty_weeks*7)
    
    dummy_idx = numpy.arange(start_date_dt, end_date_dt, numpy.timedelta64(5, 'm'), dtype='datetime64')
    
    dummy['timestamp'] = dummy_idx
    dummy.set_index('timestamp', inplace=True)    
    dummy.index = pandas.to_datetime(dummy.index)


    cycles = 0.7 * dummy.shape[0] / (365 * 24 * 12)  # how many sine cycles
    resolution = dummy.shape[0]  # how many datapoints to generate
    length = numpy.pi * 2 * cycles
    season_year = numpy.sin(numpy.arange(0, length, length / resolution))

    cycles = 12 * 4 * dummy.shape[0] / (365 * 24 * 12)  # how many sine cycles
    resolution = dummy.shape[0]  # how many datapoints to generate
    length = numpy.pi * 2 * cycles
    season_week = numpy.sin(numpy.arange(0, length, length / resolution))

    cycles = 12 * dummy.shape[0] / (365 * 24 * 12)  # how many sine cycles
    resolution = dummy.shape[0]  # how many datapoints to generate
    length = numpy.pi * 2 * cycles
    season_month = numpy.sin(numpy.arange(0, length, length / resolution))

    rand_year = random.randint(5, 10)
    rand_month = random.randint(1, 5)
    rand_week = random.randint(1, 3)

    rand_vet = numpy.random.randint(5, 10, size=dummy.shape[0])
    step_vet = numpy.zeros(dummy.shape[0])

    # Load transfer
    for i in range(0, random.randint(1, 4)):
        start = random.randint(0, dummy.shape[0])
        end = start + random.randint(1, 60) * 24 * 12

        if end >= len(step_vet):
            end = len(step_vet)

        step_vet[start:end] = random.randint(-50, -20)

    # Noise
    for i in range(0, random.randint(1, 40)):
        start = random.randint(0, dummy.shape[0])
        end = start + random.randint(1, 12 * 3)

        if end >= len(step_vet):
            end = len(step_vet)

        step_vet[start:end] = random.randint(-300, 300)

    dummy['IA'] = dummy['IA'].values + rand_year * season_year + rand_month * season_month \
                                       + rand_week * season_week + rand_vet + step_vet
    dummy['IB'] = dummy['IB'].values + rand_year * season_year + rand_month * season_month \
                                       + rand_week * season_week + rand_vet + step_vet
    dummy['IV'] = dummy['IV'].values + rand_year * season_year + rand_month * season_month \
                                       + rand_week * season_week + rand_vet + step_vet
    dummy['IN'] = dummy['IN'].values + (rand_year / 10) * season_year \
                                       + (rand_month / 10) * season_month + (rand_week / 10) \
                                                           * season_week + rand_vet / 10

    return dummy



def VoltageDummyData(qty_weeks:int = 12*4,start_date_dt:datetime = datetime(2023,1,1)):
    """
    Generate a DataFrame containing dummy voltage data over a specified number of weeks.

    This function creates a time series of voltage data, simulating variations in voltage values 
    over a given time period. The data includes random noise and step changes to mimic real-world 
    fluctuations in voltage readings.

    Parameters
    ----------
    qty_weeks : int, optional
        The number of weeks over which to generate the data (default is 48 weeks).
    start_date_dt : datetime, optional
        The start date for the data generation (default is January 1, 2023).

    Returns
    -------
    pandas.DataFrame
        A DataFrame with timestamps as index and columns 'VA', 'VB', and 'VV' representing 
        simulated voltage readings for three different phases or measurements. Each column 
        contains voltage values that are affected by random noise and step changes.

    Notes
    -----
    - The voltage values are simulated around a base value of 13.8, adjusted by a random noise 
      factor and step changes.
    - The step changes in voltage are randomly introduced at various points in the time series.
    - The timestamps are spaced 5 minutes apart.

    Examples
    --------
    >>> dummy_data = VoltageDummyData()
    >>> dummy_data.head()
    """
    
    
    end_date_dt = start_date_dt  + dt.timedelta(days=qty_weeks*7)

    dummy = numpy.arange(start_date_dt, end_date_dt, numpy.timedelta64(5, 'm'), dtype='datetime64')
    dummy = pandas.DataFrame(dummy, columns=['timestamp'])
    dummy.set_index('timestamp', inplace=True)

    rand_vet = 0.05 * 13.8 * numpy.random.rand(dummy.shape[0], 1) - 0.025 * 13.8

    step_vet = numpy.zeros((dummy.shape[0], 1))

    #  Noise
    for i in range(0, random.randint(1, 40)):
        start = random.randint(0, dummy.shape[0])
        end = start + random.randint(1, 12 * 3)

        if end >= len(step_vet):
            end = len(step_vet)

        step_vet[start:end] = random.randint(-1, 1)

    dummy['VA'] = 1.03 * 13.8 + rand_vet + step_vet
    dummy['VB'] = 1.03 * 13.8 + rand_vet + step_vet
    dummy['VV'] = 1.03 * 13.8 + rand_vet + step_vet

    return dummy


def PowerFactorDummyData(qty_weeks:int = 12*4,start_date_dt:datetime = datetime(2023,1,1)):
    
    """
    Generates dummy power factor data for a specified number of weeks starting from a given date.
    
    This function creates a pandas DataFrame containing simulated power factor data across three columns: 'FPA', 'FPB', and 'FPV'.
    Each row represents a 5-minute interval within the specified time frame. The data includes base values with added random 
    load transfer and noise effects to simulate real-world fluctuations in power factor measurements.
    
    Parameters
    ----------
    qty_weeks : int, optional
        The quantity of weeks to generate data for, defaults to 48 weeks (approximately one year).
    start_date_dt : datetime, optional
        The start date for the data generation, defaults to January 1, 2023.
    
    Returns
    -------
    pandas.DataFrame
        A DataFrame with a datetime index representing 5-minute intervals and columns 'FPA', 'FPB', and 'FPV' for power factor values.
        The data includes random variations to simulate realistic power factor changes over time.
    
    Notes
    -----
    - The function internally generates a dummy week of data and replicates it for the number of weeks specified.
    - Random load transfers and noise are added to the base values to create variability in the data.
    - The DataFrame's index is set to the timestamp of each record, making it suitable for time series analysis.
    
    Examples
    --------
    >>> import pandas
    >>> from datetime import datetime
    >>> dummy_data = PowerFactorDummyData(qty_weeks=12, start_date_dt=datetime(2023, 1, 1))
    >>> dummy_data.head()
    """

    dummy_week = pandas.DataFrame([[0.99,1.0,0.99],
                                        [0.99,1.0,0.99],
                                        [0.99,1.0,0.98],
                                        [0.99,1.0,0.98],
                                        [0.99,1.0,0.98],
                                        [0.99,1.0,0.98],
                                        [0.99,1.0,0.98],
                                        [0.99,1.0,0.98],
                                        [0.99,1.0,0.98],
                                        [0.99,1.0,0.98],
                                        [0.99,1.0,0.98],
                                        [0.99,1.0,0.98],
                                        [0.99,0.99,0.98],
                                        [0.99,0.99,0.98],
                                        [0.98,1.0,0.98],
                                        [0.98,0.99,0.98],
                                        [0.98,0.99,0.98],
                                        [0.98,0.99,0.97],
                                        [0.98,0.99,0.97],
                                        [0.98,0.99,0.98],
                                        [0.98,0.99,0.97],
                                        [0.98,0.99,0.97],
                                        [0.98,0.99,0.97],
                                        [0.98,0.99,0.97],
                                        [0.98,0.99,0.97],
                                        [0.98,0.99,0.97],
                                        [0.98,0.99,0.97],
                                        [0.98,0.99,0.98],
                                        [0.98,0.99,0.98],
                                        [0.99,1.0,0.98],
                                        [0.98,0.99,0.97],
                                        [0.98,0.99,0.97],
                                        [0.98,0.99,0.98],
                                        [0.98,0.99,0.98],
                                        [0.98,0.99,0.98],
                                        [0.98,0.99,0.98],
                                        [0.98,0.99,0.98],
                                        [0.98,0.99,0.97],
                                        [0.98,0.99,0.97],
                                        [0.98,0.99,0.97],
                                        [0.98,0.99,0.97],
                                        [0.98,0.99,0.97],
                                        [0.98,0.99,0.97],
                                        [0.98,0.99,0.97],
                                        [0.98,0.99,0.97],
                                        [0.98,0.99,0.97],
                                        [0.98,0.99,0.97],
                                        [0.98,0.99,0.97],
                                        [0.98,0.99,0.97],
                                        [0.98,0.99,0.97],
                                        [0.98,0.99,0.97],
                                        [0.98,0.99,0.97],
                                        [0.98,0.99,0.97],
                                        [0.98,0.99,0.97],
                                        [0.98,0.99,0.96],
                                        [0.98,0.99,0.97],
                                        [0.98,0.99,0.97],
                                        [0.98,0.99,0.97],
                                        [0.98,0.99,0.97],
                                        [0.97,0.99,0.97],
                                        [0.98,0.99,0.97],
                                        [0.98,0.99,0.97],
                                        [0.98,0.99,0.97],
                                        [0.98,0.99,0.97],
                                        [0.98,0.99,0.97],
                                        [0.98,0.99,0.97],
                                        [0.98,0.99,0.97],
                                        [0.98,0.99,0.97],
                                        [0.98,0.99,0.97],
                                        [0.98,0.99,0.97],
                                        [0.98,0.99,0.98],
                                        [0.98,0.99,0.98],
                                        [0.98,0.99,0.97],
                                        [0.98,0.99,0.97],
                                        [0.98,0.99,0.97],
                                        [0.97,0.99,0.97],
                                        [0.97,0.99,0.97],
                                        [0.97,0.99,0.97],
                                        [0.97,0.99,0.97],
                                        [0.97,0.99,0.97],
                                        [0.97,0.98,0.97],
                                        [0.97,0.98,0.97],
                                        [0.97,0.98,0.96],
                                        [0.97,0.98,0.96],
                                        [0.97,0.99,0.97],
                                        [0.98,0.99,0.97],
                                        [0.98,0.99,0.97],
                                        [0.97,0.99,0.98],
                                        [0.97,0.99,0.98],
                                        [0.98,0.99,0.98],
                                        [0.98,0.99,0.98],
                                        [0.98,0.99,0.98],
                                        [0.98,0.99,0.99],
                                        [0.98,0.99,0.99],
                                        [0.98,0.99,0.98],
                                        [0.98,0.99,0.98],
                                        [0.98,0.99,0.98],
                                        [0.98,0.99,0.98],
                                        [0.98,0.99,0.98],
                                        [0.98,0.99,0.98],
                                        [0.98,0.99,0.97],
                                        [0.98,0.99,0.97],
                                        [0.98,0.99,0.98],
                                        [0.98,0.99,0.98],
                                        [0.98,0.99,0.98],
                                        [0.99,0.99,0.98],
                                        [0.99,0.99,0.98],
                                        [0.98,0.99,0.98],
                                        [0.98,0.99,0.98],
                                        [0.98,0.99,0.98],
                                        [0.98,0.99,0.98],
                                        [0.98,0.99,0.98],
                                        [0.99,0.99,0.98],
                                        [0.99,0.99,0.98],
                                        [0.98,0.99,0.98],
                                        [0.98,0.99,0.98],
                                        [0.98,0.99,0.98],
                                        [0.98,0.99,0.98],
                                        [0.98,0.99,0.98],
                                        [0.98,0.99,0.98],
                                        [0.98,0.99,0.98],
                                        [0.98,0.99,0.98],
                                        [0.98,1.0,0.98],
                                        [0.98,1.0,0.98],
                                        [0.98,0.99,0.98],
                                        [0.98,0.99,0.98],
                                        [0.99,0.99,0.98],
                                        [0.99,0.99,0.98],
                                        [0.99,0.99,0.98],
                                        [0.99,1.0,0.99],
                                        [0.99,1.0,0.99],
                                        [0.99,1.0,0.98],
                                        [0.99,1.0,0.99],
                                        [0.99,1.0,0.99],
                                        [0.99,1.0,0.98],
                                        [0.99,1.0,0.98],
                                        [0.99,1.0,0.98],
                                        [0.99,1.0,0.99],
                                        [0.99,1.0,0.99],
                                        [0.99,1.0,0.99],
                                        [0.99,1.0,0.99],
                                        [0.99,1.0,0.98],
                                        [0.99,1.0,0.99],
                                        [0.99,1.0,0.99],
                                        [0.99,1.0,0.99],
                                        [0.99,1.0,0.99],
                                        [0.99,1.0,0.99],
                                        [0.99,1.0,0.99],
                                        [0.99,1.0,0.99],
                                        [0.99,1.0,0.99],
                                        [0.99,1.0,0.99],
                                        [0.99,1.0,0.99],
                                        [0.99,1.0,0.99],
                                        [0.99,1.0,0.99],
                                        [0.99,1.0,0.99],
                                        [0.99,1.0,0.99],
                                        [0.99,1.0,0.99],
                                        [0.99,1.0,0.99],
                                        [0.99,1.0,0.99],
                                        [1.0,1.0,0.99],
                                        [1.0,1.0,0.99],
                                        [0.99,1.0,0.99],
                                        [0.99,1.0,0.99],
                                        [0.99,1.0,0.99],
                                        [0.99,1.0,0.99],
                                        [0.99,1.0,0.99],
                                        [1.0,1.0,1.0],
                                        [1.0,1.0,0.99],
                                        [1.0,1.0,0.99],
                                        [1.0,1.0,0.99],
                                        [1.0,1.0,0.99],
                                        [1.0,1.0,0.99],
                                        [1.0,1.0,0.99],
                                        [1.0,1.0,0.99],
                                        [0.99,1.0,0.98],
                                        [0.99,1.0,0.98],
                                        [0.99,1.0,0.99],
                                        [0.99,1.0,0.99],
                                        [0.99,1.0,0.99],
                                        [1.0,1.0,0.99],
                                        [1.0,1.0,0.99],
                                        [0.99,1.0,0.99],
                                        [0.99,1.0,0.99],
                                        [0.99,1.0,0.99],
                                        [1.0,1.0,0.99],
                                        [1.0,1.0,0.99],
                                        [0.99,1.0,0.99],
                                        [0.99,1.0,0.99],
                                        [0.99,1.0,0.99],
                                        [0.99,1.0,0.99],
                                        [0.99,1.0,0.99],
                                        [0.99,1.0,0.99],
                                        [0.99,1.0,0.99],
                                        [0.99,1.0,0.99],
                                        [0.99,1.0,0.99],
                                        [0.99,1.0,0.99],
                                        [0.99,1.0,0.99],
                                        [0.99,1.0,0.99],
                                        [0.99,1.0,0.99],
                                        [0.99,1.0,0.99],
                                        [0.99,1.0,0.99],
                                        [0.99,1.0,0.99],
                                        [0.99,1.0,0.99],
                                        [0.99,1.0,0.99],
                                        [0.99,1.0,0.99],
                                        [0.99,1.0,0.99],
                                        [0.99,1.0,0.99],
                                        [0.99,0.99,0.99],
                                        [0.99,0.99,0.99],
                                        [0.99,1.0,0.99],
                                        [0.99,1.0,0.99],
                                        [0.99,1.0,0.99],
                                        [0.99,1.0,0.99],
                                        [0.99,1.0,0.99],
                                        [0.99,1.0,0.99],
                                        [0.99,1.0,0.99],
                                        [0.99,1.0,0.99],
                                        [1.0,1.0,0.99],
                                        [1.0,1.0,0.99],
                                        [0.99,1.0,0.99],
                                        [0.99,1.0,0.99],
                                        [0.99,1.0,0.99],
                                        [0.99,1.0,0.99],
                                        [0.99,1.0,0.99],
                                        [0.99,1.0,0.99],
                                        [0.99,1.0,0.99],
                                        [0.99,1.0,0.99],
                                        [1.0,1.0,0.99],
                                        [1.0,1.0,0.99],
                                        [0.99,1.0,0.99],
                                        [0.99,1.0,0.99],
                                        [0.99,1.0,0.99],
                                        [0.99,1.0,0.99],
                                        [0.99,1.0,0.99],
                                        [0.99,1.0,0.99],
                                        [0.99,1.0,0.99],
                                        [0.99,1.0,0.99],
                                        [0.99,1.0,0.99],
                                        [0.99,1.0,0.99],
                                        [0.99,1.0,0.99],
                                        [0.99,1.0,0.99],
                                        [0.99,1.0,0.99],
                                        [0.99,1.0,0.99],
                                        [0.99,1.0,0.99],
                                        [0.99,1.0,0.99],
                                        [0.99,1.0,0.99],
                                        [0.99,1.0,0.99],
                                        [0.99,1.0,0.99],
                                        [0.99,1.0,0.99],
                                        [0.99,1.0,0.99],
                                        [0.99,1.0,0.99],
                                        [0.99,1.0,0.98],
                                        [0.99,1.0,0.98],
                                        [0.99,1.0,0.98],
                                        [0.99,1.0,0.98],
                                        [0.99,1.0,0.98],
                                        [0.99,1.0,0.99],
                                        [0.99,1.0,0.99],
                                        [0.99,1.0,0.98],
                                        [0.99,1.0,0.98],
                                        [0.99,1.0,0.98],
                                        [0.99,1.0,0.99],
                                        [0.99,1.0,0.99],
                                        [0.99,1.0,0.99],
                                        [0.99,1.0,0.98],
                                        [0.99,1.0,0.98],
                                        [0.99,1.0,0.98],
                                        [0.99,1.0,0.98],
                                        [0.99,0.99,0.98],
                                        [0.99,1.0,0.98],
                                        [0.99,1.0,0.98],
                                        [0.99,1.0,0.98],
                                        [0.99,1.0,0.98],
                                        [0.99,1.0,0.98],
                                        [0.99,1.0,0.98],
                                        [0.99,1.0,0.98],
                                        [0.98,0.99,0.98],
                                        [0.98,0.99,0.98],
                                        [0.98,1.0,0.98],
                                        [0.98,0.99,0.98],
                                        [0.98,0.99,0.98],
                                        [0.98,1.0,0.97],
                                        [0.98,1.0,0.97],
                                        [0.99,1.0,0.98],
                                        [0.98,0.99,0.98],
                                        [0.98,0.99,0.98],
                                        [0.98,0.99,0.98],
                                        [0.98,1.0,0.98],
                                        [0.98,1.0,0.98],
                                        [0.98,0.99,0.97],
                                        [0.98,0.99,0.97],
                                        [0.99,0.99,0.97],
                                        [0.98,1.0,0.97],
                                        [0.98,1.0,0.97],
                                        [0.98,0.99,0.97],
                                        [0.98,0.99,0.97],
                                        [0.98,0.99,0.97],
                                        [0.98,0.99,0.97],
                                        [0.98,0.99,0.97],
                                        [0.98,0.99,0.97],
                                        [0.98,0.99,0.97],
                                        [0.98,0.99,0.97],
                                        [0.98,0.99,0.97],
                                        [0.98,0.99,0.97],
                                        [0.98,0.99,0.96],
                                        [0.98,0.99,0.96],
                                        [0.98,0.99,0.96],
                                        [0.97,0.99,0.96],
                                        [0.97,0.99,0.96],
                                        [0.97,0.99,0.96],
                                        [0.97,0.99,0.96],
                                        [0.98,0.99,0.97],
                                        [0.98,0.99,0.97],
                                        [0.98,0.99,0.97],
                                        [0.98,0.99,0.97],
                                        [0.98,0.99,0.97],
                                        [0.98,0.99,0.97],
                                        [0.98,0.99,0.97],
                                        [0.98,0.99,0.97],
                                        [0.98,0.99,0.97],
                                        [0.98,0.99,0.97],
                                        [0.97,0.99,0.96],
                                        [0.97,0.99,0.96],
                                        [0.97,0.99,0.96],
                                        [0.97,0.99,0.96],
                                        [0.97,0.99,0.96],
                                        [0.97,0.99,0.96],
                                        [0.97,0.99,0.96],
                                        [0.97,0.99,0.96],
                                        [0.97,0.99,0.96],
                                        [0.97,0.99,0.96],
                                        [0.97,0.99,0.96],
                                        [0.97,0.99,0.96],
                                        [0.97,0.99,0.96],
                                        [0.97,0.99,0.96],
                                        [0.97,0.99,0.96],
                                        [0.97,0.99,0.96],
                                        [0.97,0.99,0.96],
                                        [0.97,0.99,0.96],
                                        [0.97,0.99,0.96],
                                        [0.97,0.99,0.96],
                                        [0.97,0.99,0.97],
                                        [0.97,0.99,0.96],
                                        [0.97,0.99,0.96],
                                        [0.97,0.99,0.96],
                                        [0.97,0.99,0.96],
                                        [0.97,0.98,0.96],
                                        [0.97,0.99,0.96],
                                        [0.97,0.99,0.96],
                                        [0.97,0.99,0.96],
                                        [0.97,0.99,0.96],
                                        [0.97,0.99,0.96],
                                        [0.97,0.99,0.96],
                                        [0.97,0.99,0.96],
                                        [0.97,0.99,0.96],
                                        [0.97,0.99,0.96],
                                        [0.97,0.98,0.96],
                                        [0.98,0.99,0.97],
                                        [0.98,0.99,0.97],
                                        [0.98,0.99,0.97],
                                        [0.98,0.99,0.97],
                                        [0.97,0.99,0.97],
                                        [0.97,0.99,0.98],
                                        [0.97,0.99,0.98],
                                        [0.96,0.99,0.97],
                                        [0.96,0.99,0.97],
                                        [0.96,0.98,0.96],
                                        [0.97,0.99,0.96],
                                        [0.97,0.99,0.96],
                                        [0.96,0.99,0.96],
                                        [0.96,0.99,0.96],
                                        [0.96,0.98,0.96],
                                        [0.96,0.98,0.96],
                                        [0.96,0.98,0.96],
                                        [0.97,0.99,0.97],
                                        [0.97,0.99,0.97],
                                        [0.96,0.98,0.98],
                                        [0.97,0.98,0.97],
                                        [0.97,0.98,0.97],
                                        [0.97,0.98,0.97],
                                        [0.97,0.98,0.97],
                                        [0.97,0.99,0.97],
                                        [0.98,0.99,0.97],
                                        [0.98,0.99,0.97],
                                        [0.97,0.99,0.97],
                                        [0.97,0.99,0.97],
                                        [0.98,0.99,0.98],
                                        [0.98,0.99,0.97],
                                        [0.98,0.99,0.97],
                                        [0.98,0.99,0.98],
                                        [0.98,0.99,0.98],
                                        [0.98,1.0,0.98],
                                        [0.98,0.99,0.98],
                                        [0.98,0.99,0.98],
                                        [0.98,0.99,0.98],
                                        [0.98,0.99,0.98],
                                        [0.98,1.0,0.98],
                                        [0.98,1.0,0.98],
                                        [0.99,1.0,0.99],
                                        [0.98,1.0,0.99],
                                        [0.98,1.0,0.99],
                                        [0.99,0.99,0.99],
                                        [0.99,0.99,0.99],
                                        [0.98,1.0,0.99],
                                        [0.98,1.0,0.98],
                                        [0.98,1.0,0.98],
                                        [0.99,1.0,0.99],
                                        [0.99,1.0,0.99],
                                        [0.99,1.0,0.99],
                                        [0.99,1.0,0.98],
                                        [0.99,1.0,0.98],
                                        [0.99,1.0,0.99],
                                        [0.99,1.0,0.99],
                                        [0.99,1.0,0.99],
                                        [0.99,1.0,0.99],
                                        [0.99,1.0,0.99],
                                        [0.99,1.0,0.99],
                                        [0.99,1.0,0.99],
                                        [0.99,1.0,0.99],
                                        [0.99,1.0,0.99],
                                        [0.99,1.0,0.99],
                                        [0.99,1.0,0.99],
                                        [0.99,1.0,0.99],
                                        [0.99,1.0,0.99],
                                        [0.99,1.0,0.99],
                                        [0.99,1.0,0.99],
                                        [0.99,1.0,0.99],
                                        [0.99,1.0,0.99],
                                        [0.99,1.0,0.99],
                                        [0.99,1.0,0.99],
                                        [0.99,1.0,0.99],
                                        [0.99,1.0,0.99],
                                        [0.99,1.0,0.99],
                                        [0.99,1.0,0.99],
                                        [0.99,1.0,0.99],
                                        [0.99,1.0,0.99],
                                        [0.99,1.0,0.99],
                                        [0.99,1.0,0.99],
                                        [0.99,1.0,0.99],
                                        [0.99,1.0,0.99],
                                        [0.99,1.0,0.99],
                                        [0.99,1.0,0.99],
                                        [0.99,1.0,0.99],
                                        [0.99,1.0,0.99],
                                        [0.99,1.0,0.99],
                                        [0.99,1.0,0.99],
                                        [0.99,1.0,0.99],
                                        [0.99,1.0,0.99],
                                        [0.99,1.0,0.99],
                                        [0.99,1.0,0.99],
                                        [0.99,1.0,0.99],
                                        [0.99,1.0,0.99],
                                        [0.99,1.0,0.99],
                                        [0.99,1.0,0.99],
                                        [0.99,1.0,0.99],
                                        [0.99,1.0,0.99],
                                        [1.0,1.0,0.99],
                                        [1.0,1.0,0.99],
                                        [0.99,1.0,0.99],
                                        [0.99,1.0,0.99],
                                        [0.99,1.0,0.99],
                                        [0.99,1.0,0.99],
                                        [0.99,1.0,0.99],
                                        [1.0,1.0,0.99],
                                        [0.99,1.0,0.99],
                                        [0.99,1.0,0.99],
                                        [1.0,1.0,0.99],
                                        [1.0,1.0,0.99],
                                        [0.99,1.0,0.99],
                                        [0.99,1.0,0.99],
                                        [0.99,1.0,0.99],
                                        [0.99,1.0,0.99],
                                        [0.99,1.0,0.99],
                                        [0.99,1.0,0.99],
                                        [0.99,1.0,0.99],
                                        [0.99,1.0,0.99],
                                        [0.99,1.0,0.99],
                                        [0.99,1.0,0.99],
                                        [0.99,1.0,0.99],
                                        [0.99,1.0,0.99],
                                        [0.99,1.0,0.99],
                                        [0.99,1.0,0.99],
                                        [0.99,1.0,0.99],
                                        [0.99,1.0,0.99],
                                        [0.99,1.0,0.99],
                                        [0.99,1.0,0.99],
                                        [0.99,1.0,0.99],
                                        [0.99,1.0,0.99],
                                        [0.99,1.0,0.99],
                                        [0.99,1.0,0.99],
                                        [0.99,1.0,0.99],
                                        [0.99,1.0,0.99],
                                        [0.99,1.0,0.99],
                                        [0.99,1.0,0.99],
                                        [0.99,1.0,0.99],
                                        [0.99,1.0,0.99],
                                        [0.99,1.0,0.99],
                                        [0.99,1.0,0.99],
                                        [1.0,1.0,0.99],
                                        [0.99,1.0,0.99],
                                        [0.99,1.0,0.99],
                                        [0.99,1.0,0.99],
                                        [0.99,1.0,0.99],
                                        [0.99,1.0,0.99],
                                        [0.99,1.0,0.99],
                                        [0.99,1.0,0.99],
                                        [0.99,1.0,0.99],
                                        [0.99,1.0,0.99],
                                        [0.99,1.0,0.99],
                                        [0.99,1.0,0.99],
                                        [0.99,1.0,0.99],
                                        [0.99,1.0,0.99],
                                        [0.99,1.0,0.99],
                                        [0.99,1.0,0.99],
                                        [0.99,1.0,0.99],
                                        [0.99,1.0,0.99],
                                        [0.99,1.0,0.99],
                                        [0.99,1.0,0.99],
                                        [0.99,1.0,0.99],
                                        [0.99,1.0,0.99],
                                        [0.99,1.0,0.99],
                                        [0.99,1.0,0.99],
                                        [0.99,1.0,0.99],
                                        [0.99,1.0,0.99],
                                        [0.99,1.0,0.99],
                                        [0.99,1.0,0.99],
                                        [0.99,1.0,0.99],
                                        [0.99,1.0,0.99],
                                        [0.99,1.0,0.99],
                                        [0.99,1.0,0.99],
                                        [0.99,1.0,0.99],
                                        [0.99,1.0,0.99],
                                        [0.99,1.0,0.99],
                                        [0.99,1.0,0.99],
                                        [0.99,1.0,0.99],
                                        [0.99,1.0,0.99],
                                        [0.99,1.0,0.98],
                                        [0.99,1.0,0.98],
                                        [0.99,1.0,0.99],
                                        [0.99,1.0,0.99],
                                        [0.99,1.0,0.99],
                                        [0.99,1.0,0.99],
                                        [0.99,1.0,0.99],
                                        [0.99,1.0,0.99],
                                        [0.99,1.0,0.99],
                                        [0.99,1.0,0.99],
                                        [0.99,1.0,0.99],
                                        [0.99,1.0,0.99],
                                        [0.99,1.0,0.99],
                                        [0.99,1.0,0.99],
                                        [0.99,1.0,0.99],
                                        [0.99,1.0,0.98],
                                        [0.99,1.0,0.98],
                                        [0.99,1.0,0.98],
                                        [0.99,1.0,0.98],
                                        [0.99,1.0,0.98],
                                        [0.98,1.0,0.98],
                                        [0.98,1.0,0.98],
                                        [0.99,1.0,0.98],
                                        [0.99,1.0,0.98],
                                        [0.98,1.0,0.98],
                                        [0.99,1.0,0.98],
                                        [0.99,1.0,0.98],
                                        [0.99,1.0,0.98],
                                        [0.99,1.0,0.98],
                                        [0.99,1.0,0.98],
                                        [0.98,1.0,0.98],
                                        [0.98,1.0,0.98],
                                        [0.98,1.0,0.98],
                                        [0.98,1.0,0.98],
                                        [0.98,1.0,0.98],
                                        [0.98,0.99,0.98],
                                        [0.98,0.99,0.98],
                                        [0.98,0.99,0.98],
                                        [0.98,0.99,0.98],
                                        [0.99,1.0,0.98],
                                        [0.98,1.0,0.97],
                                        [0.98,1.0,0.97],
                                        [0.98,0.99,0.97],
                                        [0.98,0.99,0.97],
                                        [0.98,1.0,0.97],
                                        [0.98,1.0,0.97],
                                        [0.98,1.0,0.97],
                                        [0.98,0.99,0.97],
                                        [0.98,0.99,0.97],
                                        [0.98,0.99,0.97],
                                        [0.98,0.99,0.97],
                                        [0.98,0.99,0.97],
                                        [0.98,0.99,0.97],
                                        [0.98,0.99,0.97],
                                        [0.98,0.99,0.97],
                                        [0.98,0.99,0.97],
                                        [0.98,0.99,0.97],
                                        [0.98,0.99,0.97],
                                        [0.98,0.99,0.97],
                                        [0.98,0.99,0.96],
                                        [0.97,0.99,0.96],
                                        [0.97,0.99,0.96],
                                        [0.98,0.99,0.96],
                                        [0.98,0.99,0.96],
                                        [0.97,0.99,0.96],
                                        [0.97,0.99,0.96],
                                        [0.97,0.99,0.96],
                                        [0.97,0.99,0.96],
                                        [0.97,0.99,0.96],
                                        [0.97,0.99,0.96],
                                        [0.97,0.99,0.96],
                                        [0.97,0.99,0.96],
                                        [0.97,0.99,0.96],
                                        [0.97,0.99,0.96],
                                        [0.97,0.99,0.96],
                                        [0.97,0.99,0.96],
                                        [0.97,0.99,0.96],
                                        [0.97,0.99,0.96],
                                        [0.97,0.99,0.96],
                                        [0.97,0.99,0.96],
                                        [0.97,0.99,0.96],
                                        [0.97,0.99,0.96],
                                        [0.97,0.99,0.96],
                                        [0.97,0.99,0.96],
                                        [0.97,0.99,0.96],
                                        [0.97,0.99,0.96],
                                        [0.97,0.99,0.96],
                                        [0.96,0.98,0.96],
                                        [0.96,0.98,0.96],
                                        [0.97,0.99,0.96],
                                        [0.97,0.99,0.96],
                                        [0.97,0.99,0.96],
                                        [0.96,0.99,0.96],
                                        [0.96,0.99,0.96],
                                        [0.96,0.99,0.95],
                                        [0.97,0.99,0.95],
                                        [0.97,0.99,0.95],
                                        [0.97,0.99,0.96],
                                        [0.97,0.99,0.96],
                                        [0.97,0.99,0.96],
                                        [0.97,0.99,0.96],
                                        [0.97,0.99,0.96],
                                        [0.97,0.99,0.97],
                                        [0.97,0.99,0.97],
                                        [0.98,0.99,0.96],
                                        [0.98,0.99,0.96],
                                        [0.97,0.99,0.96],
                                        [0.97,0.99,0.96],
                                        [0.97,0.99,0.96],
                                        [0.98,0.99,0.96],
                                        [0.98,0.99,0.96],
                                        [0.97,0.99,0.96],
                                        [0.97,0.99,0.96],
                                        [0.97,0.99,0.96],
                                        [0.96,0.99,0.96],
                                        [0.96,0.99,0.96],
                                        [0.97,0.98,0.96],
                                        [0.97,0.99,0.96],
                                        [0.97,0.99,0.96],
                                        [0.96,0.98,0.96],
                                        [0.96,0.98,0.96],
                                        [0.96,0.99,0.96],
                                        [0.96,0.98,0.95],
                                        [0.96,0.98,0.95],
                                        [0.96,0.98,0.96],
                                        [0.96,0.98,0.96],
                                        [0.96,0.98,0.96],
                                        [0.96,0.98,0.96],
                                        [0.96,0.98,0.96],
                                        [0.97,0.98,0.96],
                                        [0.97,0.98,0.96],
                                        [0.97,0.99,0.97],
                                        [0.96,0.98,0.96],
                                        [0.96,0.98,0.96],
                                        [0.98,0.99,0.97],
                                        [0.98,0.99,0.97],
                                        [0.97,0.99,0.97],
                                        [0.98,0.99,0.98],
                                        [0.98,0.99,0.98],
                                        [0.98,0.99,0.98],
                                        [0.98,0.99,0.98],
                                        [0.97,0.99,0.97],
                                        [0.97,0.99,0.97],
                                        [0.97,0.99,0.97],
                                        [0.97,0.99,0.96],
                                        [0.97,0.99,0.96],
                                        [0.97,0.99,0.97],
                                        [0.98,0.99,0.98],
                                        [0.98,0.99,0.98],
                                        [0.99,1.0,0.98],
                                        [0.99,1.0,0.98],
                                        [0.98,0.99,0.97],
                                        [0.98,0.99,0.98],
                                        [0.98,0.99,0.98],
                                        [0.98,0.99,0.98],
                                        [0.98,0.99,0.98],
                                        [0.99,1.0,0.99],
                                        [0.99,1.0,0.99],
                                        [0.99,1.0,0.99],
                                        [0.99,1.0,0.99],
                                        [0.99,1.0,0.99],
                                        [0.99,1.0,0.99],
                                        [0.99,1.0,0.98],
                                        [0.99,1.0,0.98],
                                        [0.99,1.0,0.99],
                                        [0.99,1.0,0.99],
                                        [0.99,1.0,0.99],
                                        [0.99,1.0,0.99],
                                        [0.99,1.0,0.99],
                                        [0.99,1.0,0.98],
                                        [0.99,1.0,0.98],
                                        [0.99,1.0,0.99],
                                        [1.0,1.0,0.99],
                                        [1.0,1.0,0.99],
                                        [1.0,1.0,0.99],
                                        [1.0,1.0,0.99],
                                        [0.99,1.0,0.99],
                                        [1.0,1.0,0.99],
                                        [1.0,1.0,0.99],
                                        [1.0,1.0,0.99],
                                        [1.0,1.0,0.99],
                                        [0.99,1.0,0.99],
                                        [0.99,1.0,0.99],
                                        [0.99,1.0,0.99],
                                        [0.99,1.0,0.99],
                                        [0.99,1.0,0.99],
                                        [0.99,1.0,0.99],
                                        [1.0,1.0,0.99],
                                        [1.0,1.0,0.99],
                                        [0.99,1.0,0.99],
                                        [0.99,1.0,0.99],
                                        [0.99,1.0,0.99],
                                        [0.99,1.0,0.99],
                                        [0.99,1.0,0.99],
                                        [0.99,1.0,0.99],
                                        [0.99,1.0,0.99],
                                        [0.99,1.0,0.98],
                                        [0.99,1.0,0.98],
                                        [0.99,1.0,0.99],
                                        [0.99,1.0,0.99],
                                        [0.99,1.0,0.99],
                                        [0.99,1.0,0.99],
                                        [0.99,1.0,0.99],
                                        [1.0,1.0,0.99],
                                        [0.99,1.0,0.99],
                                        [0.99,1.0,0.99],
                                        [1.0,1.0,0.99],
                                        [1.0,1.0,0.99],
                                        [1.0,1.0,0.99],
                                        [1.0,1.0,0.99],
                                        [1.0,1.0,0.99],
                                        [1.0,1.0,0.99],
                                        [1.0,1.0,0.99],
                                        [0.99,1.0,0.99],
                                        [0.99,1.0,0.99],
                                        [0.99,1.0,0.99],
                                        [0.99,1.0,0.99],
                                        [0.99,1.0,0.99],
                                        [0.99,1.0,0.99],
                                        [1.0,1.0,0.99],
                                        [1.0,1.0,0.99],
                                        [1.0,1.0,1.0],
                                        [1.0,1.0,1.0],
                                        [0.99,1.0,0.99],
                                        [0.99,1.0,0.99],
                                        [0.99,1.0,0.99],
                                        [0.99,1.0,0.99],
                                        [0.99,1.0,0.99],
                                        [0.99,1.0,0.99],
                                        [1.0,1.0,0.99],
                                        [1.0,1.0,0.99],
                                        [0.99,1.0,0.99],
                                        [0.99,1.0,0.99],
                                        [0.99,1.0,0.99],
                                        [0.99,1.0,0.99],
                                        [0.99,1.0,0.99],
                                        [0.99,1.0,0.99],
                                        [0.99,1.0,0.99],
                                        [0.99,1.0,0.99],
                                        [0.99,1.0,0.99],
                                        [0.99,1.0,0.99],
                                        [0.99,0.99,0.99],
                                        [0.99,0.99,0.99],
                                        [0.99,1.0,0.98],
                                        [0.99,1.0,0.99],
                                        [0.99,1.0,0.99],
                                        [0.99,1.0,0.99],
                                        [0.99,1.0,0.99],
                                        [0.99,1.0,0.99],
                                        [0.99,1.0,0.99],
                                        [0.99,1.0,0.99],
                                        [0.99,1.0,0.99],
                                        [0.99,1.0,0.99],
                                        [0.99,1.0,0.99],
                                        [0.99,1.0,0.99],
                                        [0.99,1.0,0.99],
                                        [0.99,1.0,0.99],
                                        [0.99,1.0,0.99],
                                        [0.99,1.0,0.99],
                                        [0.99,1.0,0.99],
                                        [0.99,1.0,0.99],
                                        [1.0,1.0,0.99],
                                        [1.0,1.0,0.99],
                                        [0.99,1.0,0.99],
                                        [0.99,1.0,0.99],
                                        [0.99,1.0,0.99],
                                        [0.99,1.0,0.99],
                                        [0.99,1.0,0.99],
                                        [0.99,1.0,0.99],
                                        [0.99,1.0,0.99],
                                        [0.99,1.0,0.99],
                                        [0.99,1.0,0.99],
                                        [0.99,1.0,0.99],
                                        [0.99,1.0,0.99],
                                        [0.99,1.0,0.99],
                                        [0.99,1.0,0.99],
                                        [0.99,1.0,0.99],
                                        [0.99,1.0,0.99],
                                        [0.99,1.0,0.99],
                                        [0.99,1.0,0.99],
                                        [0.99,1.0,0.99],
                                        [0.99,1.0,0.98],
                                        [0.99,1.0,0.98],
                                        [1.0,1.0,0.99],
                                        [1.0,1.0,0.99],
                                        [0.99,1.0,0.99],
                                        [0.99,1.0,0.99],
                                        [0.99,1.0,0.99],
                                        [0.99,1.0,0.99],
                                        [0.99,1.0,0.99],
                                        [0.99,1.0,0.99],
                                        [0.99,1.0,0.99],
                                        [0.99,1.0,0.99],
                                        [0.99,1.0,0.99],
                                        [0.99,1.0,0.99],
                                        [0.99,1.0,0.99],
                                        [0.99,1.0,0.99],
                                        [0.99,1.0,0.99],
                                        [0.99,1.0,0.99],
                                        [0.99,1.0,0.99],
                                        [0.99,1.0,0.98],
                                        [0.99,1.0,0.99],
                                        [0.99,1.0,0.99],
                                        [0.99,1.0,0.98],
                                        [0.99,1.0,0.98],
                                        [0.99,0.99,0.98],
                                        [0.99,0.99,0.98],
                                        [0.99,0.99,0.98],
                                        [0.99,1.0,0.99],
                                        [0.99,1.0,0.99],
                                        [0.99,1.0,0.99],
                                        [0.99,1.0,0.98],
                                        [0.99,1.0,0.98],
                                        [0.99,1.0,0.98],
                                        [0.99,1.0,0.98],
                                        [0.99,1.0,0.98],
                                        [0.99,1.0,0.98],
                                        [0.99,1.0,0.98],
                                        [0.99,1.0,0.98],
                                        [0.99,1.0,0.98],
                                        [0.99,1.0,0.98],
                                        [0.99,0.99,0.98],
                                        [0.99,0.99,0.98],
                                        [0.99,0.99,0.98],
                                        [0.99,0.99,0.98],
                                        [0.99,0.99,0.98],
                                        [0.99,0.99,0.99],
                                        [0.99,0.99,0.99],
                                        [0.98,0.99,0.98],
                                        [0.98,0.99,0.98],
                                        [0.98,0.99,0.98],
                                        [0.98,0.99,0.98],
                                        [0.98,0.99,0.98],
                                        [0.98,0.99,0.97],
                                        [0.98,0.99,0.97],
                                        [0.98,0.99,0.97],
                                        [0.98,0.99,0.97],
                                        [0.98,0.99,0.97],
                                        [0.98,0.99,0.97],
                                        [0.98,0.99,0.97],
                                        [0.98,0.99,0.97],
                                        [0.98,0.99,0.97],
                                        [0.98,0.99,0.97],
                                        [0.97,0.99,0.97],
                                        [0.97,0.99,0.97],
                                        [0.98,0.99,0.97],
                                        [0.98,0.99,0.97],
                                        [0.98,0.99,0.97],
                                        [0.97,0.99,0.97],
                                        [0.97,0.99,0.97],
                                        [0.98,0.99,0.97],
                                        [0.98,0.99,0.97],
                                        [0.98,0.99,0.97],
                                        [0.98,0.99,0.97],
                                        [0.98,0.99,0.97],
                                        [0.97,0.99,0.97],
                                        [0.98,0.99,0.97],
                                        [0.98,0.99,0.97],
                                        [0.97,0.99,0.97],
                                        [0.97,0.99,0.97],
                                        [0.98,0.99,0.97],
                                        [0.97,0.99,0.97],
                                        [0.97,0.99,0.97],
                                        [0.98,0.99,0.97],
                                        [0.98,0.99,0.97],
                                        [0.97,0.99,0.96],
                                        [0.97,0.99,0.96],
                                        [0.97,0.99,0.96],
                                        [0.97,0.99,0.97],
                                        [0.97,0.99,0.97],
                                        [0.97,0.98,0.96],
                                        [0.98,0.99,0.97],
                                        [0.98,0.99,0.97],
                                        [0.97,0.99,0.97],
                                        [0.97,0.99,0.97],
                                        [0.97,0.99,0.97],
                                        [0.97,0.99,0.96],
                                        [0.97,0.99,0.96],
                                        [0.97,0.99,0.97],
                                        [0.97,0.99,0.97],
                                        [0.97,0.99,0.97],
                                        [0.97,0.99,0.97],
                                        [0.97,0.99,0.97],
                                        [0.97,0.99,0.97],
                                        [0.97,0.99,0.97],
                                        [0.98,0.99,0.97],
                                        [0.97,0.99,0.97],
                                        [0.97,0.99,0.97],
                                        [0.97,0.99,0.96],
                                        [0.97,0.99,0.96],
                                        [0.97,0.99,0.96],
                                        [0.97,0.99,0.96],
                                        [0.97,0.99,0.96],
                                        [0.97,0.99,0.96],
                                        [0.97,0.99,0.96],
                                        [0.97,0.99,0.97],
                                        [0.97,0.99,0.96],
                                        [0.97,0.99,0.96],
                                        [0.97,0.99,0.96],
                                        [0.97,0.99,0.96],
                                        [0.97,0.99,0.96],
                                        [0.97,0.99,0.96],
                                        [0.97,0.99,0.96],
                                        [0.96,0.98,0.95],
                                        [0.96,0.98,0.95],
                                        [0.96,0.98,0.95],
                                        [0.96,0.98,0.95],
                                        [0.96,0.98,0.95],
                                        [0.96,0.98,0.96],
                                        [0.96,0.98,0.96],
                                        [0.96,0.98,0.95],
                                        [0.96,0.98,0.95],
                                        [0.95,0.97,0.94],
                                        [0.96,0.98,0.95],
                                        [0.96,0.98,0.95],
                                        [0.95,0.98,0.95],
                                        [0.95,0.98,0.95],
                                        [0.96,0.98,0.96],
                                        [0.96,0.98,0.96],
                                        [0.96,0.98,0.96],
                                        [0.96,0.98,0.97],
                                        [0.96,0.98,0.97],
                                        [0.96,0.98,0.96],
                                        [0.96,0.99,0.97],
                                        [0.96,0.99,0.97],
                                        [0.97,0.99,0.97],
                                        [0.97,0.99,0.97],
                                        [0.97,0.98,0.96],
                                        [0.97,0.98,0.96],
                                        [0.97,0.98,0.96],
                                        [0.98,0.98,0.97],
                                        [0.98,0.98,0.97],
                                        [0.97,0.99,0.96],
                                        [0.98,0.99,0.96],
                                        [0.98,0.99,0.96],
                                        [0.98,0.99,0.97],
                                        [0.98,0.99,0.97],
                                        [0.97,0.99,0.96],
                                        [0.97,0.99,0.97],
                                        [0.97,0.99,0.97],
                                        [0.97,0.99,0.97],
                                        [0.97,0.99,0.97],
                                        [0.98,0.99,0.97],
                                        [0.98,0.99,0.97],
                                        [0.98,0.99,0.97],
                                        [0.98,0.99,0.98],
                                        [0.98,0.99,0.98],
                                        [0.98,1.0,0.98],
                                        [0.98,0.99,0.98],
                                        [0.98,0.99,0.98],
                                        [0.98,0.99,0.98],
                                        [0.98,0.99,0.98],
                                        [0.98,0.99,0.98],
                                        [0.98,0.99,0.98],
                                        [0.98,0.99,0.98],
                                        [0.99,0.99,0.98],
                                        [0.99,0.99,0.98],
                                        [0.99,1.0,0.99],
                                        [0.99,1.0,0.99],
                                        [0.99,1.0,0.99],
                                        [0.99,1.0,0.99],
                                        [0.99,1.0,0.99],
                                        [0.99,1.0,0.99],
                                        [0.99,1.0,0.98],
                                        [0.99,1.0,0.98],
                                        [0.99,1.0,0.98],
                                        [0.99,1.0,0.98],
                                        [0.98,1.0,0.99],
                                        [0.98,1.0,0.99],
                                        [0.98,1.0,0.99],
                                        [0.99,1.0,0.99],
                                        [0.99,1.0,0.99],
                                        [0.99,1.0,0.99],
                                        [0.99,1.0,0.99],
                                        [0.99,1.0,0.99],
                                        [0.99,1.0,0.99],
                                        [0.99,1.0,0.99],
                                        [0.99,1.0,0.99],
                                        [0.99,1.0,0.98],
                                        [0.99,1.0,0.98],
                                        [0.99,1.0,0.99],
                                        [0.99,1.0,0.99],
                                        [0.99,1.0,0.99],
                                        [0.99,1.0,0.99],
                                        [0.99,1.0,0.99],
                                        [0.99,1.0,0.99],
                                        [0.99,1.0,0.99],
                                        [0.99,1.0,0.99],
                                        [0.99,1.0,0.99],
                                        [0.99,1.0,0.99],
                                        [0.99,1.0,0.99],
                                        [0.99,1.0,0.99],
                                        [0.99,1.0,0.99],
                                        [0.99,1.0,0.99],
                                        [0.99,1.0,0.99],
                                        [0.99,1.0,0.98],
                                        [0.99,1.0,0.98],
                                        [0.99,1.0,0.98],
                                        [0.99,1.0,0.98],
                                        [0.99,1.0,0.98],
                                        [0.99,1.0,0.98],
                                        [0.99,1.0,0.98],
                                        [0.99,1.0,0.99],
                                        [0.99,1.0,0.98],
                                        [0.99,1.0,0.98],
                                        [0.99,0.99,0.98],
                                        [0.99,0.99,0.98],
                                        [0.99,1.0,0.98],
                                        [0.98,1.0,0.98],
                                        [0.98,1.0,0.98],
                                        [0.99,1.0,0.98],
                                        [0.99,1.0,0.98],
                                        [0.99,1.0,0.98],
                                        [0.99,0.99,0.98],
                                        [0.99,0.99,0.98],
                                        [0.99,0.99,0.98],
                                        [0.99,0.99,0.98],
                                        [0.99,1.0,0.99],
                                        [0.99,1.0,0.99],
                                        [0.99,1.0,0.99],
                                        [0.99,1.0,0.98],
                                        [0.99,1.0,0.98],
                                        [0.98,0.99,0.98],
                                        [0.98,0.99,0.98],
                                        [0.99,1.0,0.98],
                                        [0.99,1.0,0.99],
                                        [0.99,1.0,0.99],
                                        [0.99,1.0,0.98],
                                        [0.99,1.0,0.98],
                                        [0.99,1.0,0.98],
                                        [0.99,1.0,0.98],
                                        [0.99,1.0,0.98],
                                        [0.99,1.0,0.99],
                                        [0.99,1.0,0.99],
                                        [0.99,0.99,0.99],
                                        [0.99,1.0,0.99],
                                        [0.99,1.0,0.99],
                                        [0.99,1.0,0.99],
                                        [0.99,1.0,0.99],
                                        [0.99,1.0,0.99],
                                        [0.99,1.0,0.99],
                                        [0.99,1.0,0.99],
                                        [0.99,1.0,0.99],
                                        [0.99,1.0,0.99],
                                        [0.99,1.0,0.99],
                                        [0.99,1.0,0.99],
                                        [0.99,1.0,0.99],
                                        [0.99,1.0,0.99],
                                        [0.99,1.0,0.99],
                                        [0.99,1.0,0.99],
                                        [0.99,1.0,0.98],
                                        [0.99,1.0,0.98],
                                        [0.99,1.0,0.99],
                                        [0.99,1.0,0.99],
                                        [0.99,1.0,0.99],
                                        [0.99,1.0,0.99],
                                        [0.99,1.0,0.99],
                                        [0.99,1.0,0.99],
                                        [0.99,1.0,0.99],
                                        [0.99,1.0,0.99],
                                        [0.99,1.0,0.99],
                                        [0.99,1.0,0.99],
                                        [0.99,1.0,0.99],
                                        [0.99,1.0,0.99],
                                        [0.99,1.0,0.99],
                                        [0.99,1.0,0.99],
                                        [0.99,1.0,0.99],
                                        [0.99,1.0,0.99],
                                        [0.99,1.0,0.99],
                                        [0.99,1.0,0.99],
                                        [0.99,1.0,0.99],
                                        [0.99,1.0,0.99],
                                        [0.99,1.0,0.99],
                                        [0.99,1.0,0.99],
                                        [0.99,1.0,0.99],
                                        [0.99,1.0,0.99],
                                        [0.99,1.0,0.99],
                                        [0.99,1.0,0.99],
                                        [0.99,1.0,0.99],
                                        [0.99,1.0,0.99],
                                        [0.99,1.0,0.99],
                                        [0.99,1.0,0.99],
                                        [0.99,1.0,0.99],
                                        [0.99,1.0,0.99],
                                        [0.99,1.0,0.99],
                                        [0.99,1.0,0.98],
                                        [0.99,1.0,0.98],
                                        [0.99,1.0,0.98],
                                        [0.99,1.0,0.98],
                                        [0.99,1.0,0.98],
                                        [0.99,1.0,0.99],
                                        [0.99,1.0,0.99],
                                        [0.99,1.0,0.99],
                                        [0.99,1.0,0.99],
                                        [0.99,1.0,0.99],
                                        [0.99,1.0,0.99],
                                        [0.99,1.0,0.99],
                                        [0.99,1.0,0.98],
                                        [0.99,1.0,0.98],
                                        [0.99,1.0,0.99],
                                        [0.99,1.0,0.98],
                                        [0.99,1.0,0.98],
                                        [0.98,1.0,0.98],
                                        [0.98,1.0,0.98],
                                        [0.99,1.0,0.98],
                                        [0.98,1.0,0.98],
                                        [0.98,1.0,0.98],
                                        [0.99,1.0,0.98],
                                        [0.99,1.0,0.98],
                                        [0.98,1.0,0.98],
                                        [0.98,1.0,0.98],
                                        [0.98,1.0,0.98],
                                        [0.98,1.0,0.98],
                                        [0.98,1.0,0.98],
                                        [0.98,0.99,0.98],
                                        [0.98,0.99,0.98],
                                        [0.98,0.99,0.98],
                                        [0.98,0.99,0.97],
                                        [0.98,0.99,0.97],
                                        [0.98,0.99,0.98],
                                        [0.98,0.99,0.98],
                                        [0.98,0.99,0.98],
                                        [0.98,1.0,0.98],
                                        [0.98,1.0,0.98],
                                        [0.98,0.99,0.98],
                                        [0.98,0.99,0.98],
                                        [0.98,0.99,0.98],
                                        [0.98,0.99,0.97],
                                        [0.98,0.99,0.97],
                                        [0.98,0.99,0.97],
                                        [0.98,0.99,0.98],
                                        [0.98,0.99,0.98],
                                        [0.98,0.99,0.97],
                                        [0.98,0.99,0.97],
                                        [0.98,0.99,0.97],
                                        [0.98,0.99,0.97],
                                        [0.98,0.99,0.97],
                                        [0.97,0.99,0.97],
                                        [0.97,0.99,0.97],
                                        [0.98,0.99,0.97],
                                        [0.97,0.99,0.97],
                                        [0.97,0.99,0.97],
                                        [0.98,0.99,0.97],
                                        [0.98,0.99,0.97],
                                        [0.97,0.99,0.97],
                                        [0.98,0.99,0.97],
                                        [0.98,0.99,0.97],
                                        [0.97,0.99,0.97],
                                        [0.97,0.99,0.97],
                                        [0.97,0.98,0.96],
                                        [0.97,0.99,0.97],
                                        [0.97,0.99,0.97],
                                        [0.97,0.99,0.97],
                                        [0.97,0.99,0.97],
                                        [0.97,0.99,0.96],
                                        [0.97,0.99,0.97],
                                        [0.97,0.99,0.97],
                                        [0.97,0.99,0.96],
                                        [0.97,0.99,0.96],
                                        [0.97,0.99,0.96],
                                        [0.97,0.98,0.96],
                                        [0.97,0.98,0.96],
                                        [0.97,0.99,0.96],
                                        [0.97,0.99,0.96],
                                        [0.97,0.99,0.96],
                                        [0.97,0.99,0.96],
                                        [0.97,0.99,0.96],
                                        [0.97,0.99,0.96],
                                        [0.97,0.99,0.96],
                                        [0.97,0.98,0.96],
                                        [0.97,0.99,0.96],
                                        [0.97,0.99,0.96],
                                        [0.96,0.98,0.96],
                                        [0.96,0.98,0.96],
                                        [0.97,0.99,0.95],
                                        [0.97,0.98,0.96],
                                        [0.97,0.98,0.96],
                                        [0.97,0.98,0.96],
                                        [0.97,0.98,0.96],
                                        [0.97,0.99,0.96],
                                        [0.97,0.99,0.96],
                                        [0.97,0.99,0.97],
                                        [0.97,0.99,0.97],
                                        [0.97,0.99,0.97],
                                        [0.97,0.98,0.97],
                                        [0.97,0.98,0.97],
                                        [0.97,0.99,0.96],
                                        [0.97,0.98,0.96],
                                        [0.97,0.98,0.96],
                                        [0.97,0.99,0.96],
                                        [0.97,0.99,0.96],
                                        [0.97,0.98,0.97],
                                        [0.96,0.98,0.96],
                                        [0.96,0.98,0.96],
                                        [0.97,0.99,0.96],
                                        [0.97,0.99,0.96],
                                        [0.96,0.98,0.96],
                                        [0.95,0.98,0.95],
                                        [0.95,0.98,0.95],
                                        [0.96,0.98,0.96],
                                        [0.96,0.98,0.96],
                                        [0.96,0.98,0.96],
                                        [0.96,0.98,0.97],
                                        [0.96,0.98,0.97],
                                        [0.97,0.98,0.97],
                                        [0.97,0.98,0.97],
                                        [0.96,0.98,0.97],
                                        [0.96,0.98,0.97],
                                        [0.96,0.98,0.97],
                                        [0.97,0.99,0.97],
                                        [0.97,0.99,0.97],
                                        [0.97,0.98,0.97],
                                        [0.97,0.99,0.97],
                                        [0.97,0.99,0.97],
                                        [0.97,0.99,0.97],
                                        [0.97,0.99,0.97],
                                        [0.97,0.99,0.97],
                                        [0.98,0.99,0.97],
                                        [0.98,0.99,0.97],
                                        [0.98,0.99,0.97],
                                        [0.98,0.99,0.97],
                                        [0.97,0.99,0.96],
                                        [0.97,0.99,0.98],
                                        [0.97,0.99,0.98],
                                        [0.98,0.99,0.98],
                                        [0.98,0.99,0.98],
                                        [0.98,0.99,0.98],
                                        [0.99,0.99,0.99],
                                        [0.99,0.99,0.99],
                                        [0.98,0.99,0.98],
                                        [0.98,0.99,0.98],
                                        [0.98,0.99,0.97],
                                        [0.99,1.0,0.99],
                                        [0.99,1.0,0.99],
                                        [0.98,0.99,0.98],
                                        [0.98,0.99,0.98],
                                        [0.98,0.99,0.98],
                                        [0.99,1.0,0.98],
                                        [0.99,1.0,0.98],
                                        [0.99,1.0,0.99],
                                        [0.99,1.0,0.99],
                                        [0.98,1.0,0.99],
                                        [0.98,1.0,0.98],
                                        [0.98,1.0,0.98],
                                        [0.98,1.0,0.98],
                                        [0.98,1.0,0.98],
                                        [0.99,1.0,0.99],
                                        [0.99,1.0,0.99],
                                        [0.99,1.0,0.99],
                                        [0.99,1.0,0.99],
                                        [0.99,1.0,0.99],
                                        [1.0,1.0,0.99],
                                        [0.99,1.0,0.99],
                                        [0.99,1.0,0.99],
                                        [0.99,1.0,0.99],
                                        [0.99,1.0,0.99],
                                        [0.99,1.0,0.99],
                                        [1.0,1.0,0.99],
                                        [1.0,1.0,0.99],
                                        [1.0,1.0,0.99],
                                        [1.0,1.0,0.99],
                                        [0.99,1.0,0.99],
                                        [0.99,1.0,0.99],
                                        [0.99,1.0,0.99],
                                        [0.99,1.0,0.99],
                                        [0.99,1.0,0.99],
                                        [0.99,1.0,0.99],
                                        [0.99,1.0,0.99],
                                        [0.99,1.0,0.99],
                                        [0.99,1.0,0.99],
                                        [0.99,1.0,0.99],
                                        [0.99,1.0,0.99],
                                        [0.99,1.0,0.99],
                                        [0.99,1.0,0.99],
                                        [0.99,1.0,0.99],
                                        [0.99,1.0,0.99],
                                        [1.0,1.0,0.99],
                                        [1.0,1.0,0.99],
                                        [1.0,1.0,0.99],
                                        [1.0,1.0,0.99],
                                        [1.0,1.0,0.99],
                                        [1.0,1.0,1.0],
                                        [1.0,1.0,0.99],
                                        [1.0,1.0,0.99],
                                        [1.0,1.0,1.0],
                                        [1.0,1.0,1.0],
                                        [1.0,1.0,0.99],
                                        [1.0,1.0,1.0],
                                        [1.0,1.0,1.0],
                                        [0.99,1.0,1.0],
                                        [0.99,1.0,1.0],
                                        [1.0,1.0,1.0],
                                        [1.0,1.0,1.0],
                                        [1.0,1.0,0.99],
                                        [1.0,1.0,1.0],
                                        [1.0,1.0,1.0],
                                        [1.0,1.0,1.0],
                                        [1.0,1.0,1.0],
                                        [1.0,1.0,1.0],
                                        [1.0,1.0,1.0],
                                        [1.0,1.0,1.0],
                                        [1.0,1.0,1.0],
                                        [1.0,1.0,1.0],
                                        [1.0,1.0,0.99],
                                        [1.0,1.0,0.99],
                                        [1.0,1.0,0.99],
                                        [1.0,1.0,0.99],
                                        [1.0,1.0,0.99],
                                        [1.0,1.0,0.99],
                                        [0.99,1.0,0.99],
                                        [0.99,1.0,0.99],
                                        [1.0,1.0,0.99],
                                        [1.0,1.0,0.99],
                                        [1.0,1.0,0.99],
                                        [0.99,1.0,0.99],
                                        [0.99,1.0,0.99],
                                        [1.0,1.0,0.99],
                                        [1.0,1.0,0.99],
                                        [0.99,1.0,1.0],
                                        [1.0,1.0,1.0],
                                        [1.0,1.0,1.0],
                                        [1.0,1.0,0.99],
                                        [1.0,1.0,0.99],
                                        [1.0,1.0,0.99],
                                        [0.99,1.0,1.0],
                                        [0.99,1.0,1.0],
                                        [1.0,1.0,1.0],
                                        [1.0,1.0,1.0],
                                        [1.0,1.0,1.0],
                                        [1.0,1.0,1.0],
                                        [1.0,1.0,1.0],
                                        [1.0,1.0,1.0],
                                        [1.0,1.0,1.0],
                                        [1.0,1.0,1.0],
                                        [1.0,1.0,1.0],
                                        [1.0,1.0,1.0],
                                        [1.0,1.0,0.99],
                                        [1.0,1.0,0.99],
                                        [1.0,1.0,1.0],
                                        [1.0,1.0,1.0],
                                        [1.0,1.0,1.0],
                                        [1.0,1.0,1.0],
                                        [1.0,1.0,1.0],
                                        [1.0,1.0,0.99],
                                        [0.99,1.0,0.99],
                                        [0.99,1.0,0.99],
                                        [0.99,1.0,0.99],
                                        [0.99,1.0,0.99],
                                        [0.99,1.0,0.99],
                                        [1.0,1.0,0.99],
                                        [1.0,1.0,0.99],
                                        [0.99,1.0,0.99],
                                        [0.99,1.0,0.99],
                                        [1.0,1.0,0.99],
                                        [0.99,1.0,0.99],
                                        [0.99,1.0,0.99],
                                        [0.99,1.0,0.99],
                                        [0.99,1.0,0.99],
                                        [1.0,1.0,0.99],
                                        [1.0,1.0,0.99],
                                        [1.0,1.0,0.99],
                                        [1.0,1.0,0.99],
                                        [1.0,1.0,0.99],
                                        [0.99,1.0,0.99],
                                        [0.99,1.0,0.99],
                                        [0.99,1.0,0.99],
                                        [0.99,1.0,0.99],
                                        [0.99,1.0,0.99],
                                        [0.99,1.0,0.99],
                                        [1.0,1.0,0.99],
                                        [1.0,1.0,0.99],
                                        [0.99,1.0,0.99],
                                        [0.99,1.0,0.99],
                                        [0.99,1.0,0.99],
                                        [0.99,1.0,0.99],
                                        [0.99,1.0,0.99],
                                        [0.99,1.0,0.99],
                                        [0.99,1.0,0.99],
                                        [0.99,1.0,0.99],
                                        [0.99,1.0,0.99],
                                        [0.99,1.0,0.99],
                                        [0.99,1.0,0.99],
                                        [0.99,1.0,0.99],
                                        [0.99,1.0,0.99],
                                        [0.99,1.0,0.99],
                                        [0.99,1.0,0.99],
                                        [0.99,1.0,0.99],
                                        [0.99,1.0,0.99],
                                        [0.99,1.0,0.98],
                                        [0.99,1.0,0.98],
                                        [0.99,1.0,0.98],
                                        [0.99,1.0,0.99],
                                        [0.99,1.0,0.99],
                                        [0.99,1.0,0.99],
                                        [0.99,1.0,0.99],
                                        [0.99,1.0,0.99],
                                        [0.99,1.0,0.99],
                                        [0.99,1.0,0.99],
                                        [0.99,1.0,0.99],
                                        [0.99,1.0,0.99],
                                        [0.99,1.0,0.98],
                                        [0.99,1.0,0.98],
                                        [0.99,1.0,0.98],
                                        [0.99,1.0,0.98],
                                        [0.99,1.0,0.98],
                                        [0.98,0.99,0.98],
                                        [0.98,0.99,0.98],
                                        [0.98,0.99,0.98],
                                        [0.98,0.99,0.98],
                                        [0.98,0.99,0.98],
                                        [0.98,0.99,0.98],
                                        [0.98,1.0,0.98],
                                        [0.98,1.0,0.98],
                                        [0.98,0.99,0.98],
                                        [0.98,0.99,0.98],
                                        [0.98,1.0,0.98],
                                        [0.98,1.0,0.98],
                                        [0.98,1.0,0.98],
                                        [0.98,0.99,0.98],
                                        [0.98,0.99,0.98],
                                        [0.98,0.99,0.98],
                                        [0.98,0.99,0.98],
                                        [0.98,0.99,0.98],
                                        [0.98,0.99,0.98],
                                        [0.98,0.99,0.98],
                                        [0.98,0.99,0.97],
                                        [0.98,0.99,0.97],
                                        [0.98,0.99,0.97],
                                        [0.98,0.99,0.97],
                                        [0.98,0.99,0.97],
                                        [0.98,0.99,0.97],
                                        [0.97,0.99,0.97],
                                        [0.97,0.99,0.97],
                                        [0.98,0.99,0.97],
                                        [0.98,0.99,0.97],
                                        [0.98,0.99,0.97],
                                        [0.98,0.99,0.97],
                                        [0.98,0.99,0.97],
                                        [0.98,0.99,0.97],
                                        [0.98,0.99,0.97],
                                        [0.98,0.99,0.97],
                                        [0.97,0.99,0.97],
                                        [0.98,0.99,0.97],
                                        [0.98,0.99,0.97],
                                        [0.97,0.99,0.97],
                                        [0.97,0.99,0.97],
                                        [0.98,0.99,0.97],
                                        [0.97,0.99,0.97],
                                        [0.97,0.99,0.97],
                                        [0.97,0.99,0.96],
                                        [0.97,0.99,0.96],
                                        [0.97,0.99,0.97],
                                        [0.98,0.99,0.96],
                                        [0.98,0.99,0.96],
                                        [0.97,0.99,0.96],
                                        [0.97,0.99,0.96],
                                        [0.97,0.99,0.96],
                                        [0.97,0.99,0.96],
                                        [0.97,0.99,0.96],
                                        [0.97,0.99,0.97],
                                        [0.97,0.99,0.97],
                                        [0.97,0.99,0.96],
                                        [0.97,0.99,0.96],
                                        [0.97,0.99,0.96],
                                        [0.98,0.99,0.97],
                                        [0.98,0.99,0.97],
                                        [0.98,0.99,0.97],
                                        [0.97,0.99,0.96],
                                        [0.97,0.99,0.96],
                                        [0.97,0.99,0.97],
                                        [0.97,0.99,0.97],
                                        [0.97,0.99,0.96],
                                        [0.97,0.99,0.96],
                                        [0.97,0.99,0.96],
                                        [0.98,0.99,0.96],
                                        [0.98,0.99,0.96],
                                        [0.98,0.99,0.97],
                                        [0.97,0.99,0.96],
                                        [0.97,0.99,0.96],
                                        [0.96,0.98,0.96],
                                        [0.96,0.98,0.96],
                                        [0.96,0.98,0.96],
                                        [0.97,0.98,0.96],
                                        [0.97,0.98,0.96],
                                        [0.96,0.98,0.95],
                                        [0.96,0.98,0.95],
                                        [0.96,0.98,0.95],
                                        [0.96,0.99,0.96],
                                        [0.96,0.99,0.96],
                                        [0.96,0.98,0.96],
                                        [0.96,0.98,0.96],
                                        [0.97,0.98,0.96],
                                        [0.97,0.98,0.97],
                                        [0.97,0.98,0.97],
                                        [0.96,0.98,0.96],
                                        [0.96,0.98,0.96],
                                        [0.97,0.99,0.96],
                                        [0.97,0.99,0.97],
                                        [0.97,0.99,0.97],
                                        [0.97,0.99,0.97],
                                        [0.97,0.99,0.97],
                                        [0.97,0.99,0.97],
                                        [0.97,0.98,0.96],
                                        [0.97,0.98,0.96],
                                        [0.98,0.99,0.98],
                                        [0.98,0.99,0.98],
                                        [0.98,0.99,0.97],
                                        [0.99,0.99,0.98],
                                        [0.99,0.99,0.98],
                                        [0.98,0.99,0.98],
                                        [0.98,0.99,0.98],
                                        [0.98,0.99,0.98],
                                        [0.98,0.99,0.98],
                                        [0.98,1.0,0.98],
                                        [0.99,1.0,0.98],
                                        [0.99,1.0,0.98],
                                        [0.99,1.0,0.98],
                                        [0.99,1.0,0.98],
                                        [0.98,1.0,0.98],
                                        [0.99,1.0,0.99],
                                        [0.99,1.0,0.99],
                                        [0.99,1.0,0.99],
                                        [0.99,1.0,0.99],
                                        [0.99,1.0,0.99],
                                        [0.99,1.0,0.99],
                                        [0.99,1.0,0.99],
                                        [0.99,1.0,0.99],
                                        [0.99,1.0,0.99],
                                        [0.99,1.0,0.99],
                                        [0.99,1.0,0.99],
                                        [0.99,1.0,0.99],
                                        [0.99,1.0,0.99],
                                        [0.99,1.0,0.99],
                                        [0.99,1.0,0.99],
                                        [0.99,1.0,0.99],
                                        [0.99,1.0,0.99],
                                        [1.0,1.0,0.99],
                                        [1.0,1.0,0.99],
                                        [1.0,1.0,0.99],
                                        [1.0,1.0,0.99],
                                        [1.0,1.0,0.99],
                                        [0.99,1.0,0.99],
                                        [0.99,1.0,0.99],
                                        [0.99,1.0,1.0],
                                        [1.0,1.0,1.0],
                                        [1.0,1.0,1.0],
                                        [0.99,1.0,0.99],
                                        [0.99,1.0,0.99],
                                        [0.99,1.0,0.99],
                                        [1.0,1.0,0.99],
                                        [1.0,1.0,0.99],
                                        [1.0,1.0,0.99],
                                        [1.0,1.0,0.99],
                                        [1.0,1.0,1.0],
                                        [1.0,1.0,1.0],
                                        [1.0,1.0,1.0],
                                        [1.0,1.0,1.0],
                                        [1.0,1.0,1.0],
                                        [1.0,1.0,1.0],
                                        [1.0,1.0,1.0],
                                        [1.0,1.0,1.0],
                                        [1.0,1.0,1.0],
                                        [1.0,1.0,1.0],
                                        [1.0,1.0,1.0],
                                        [1.0,1.0,1.0],
                                        [1.0,1.0,1.0],
                                        [1.0,1.0,1.0],
                                        [1.0,1.0,1.0],
                                        [1.0,1.0,1.0],
                                        [1.0,1.0,1.0],
                                        [1.0,1.0,1.0],
                                        [1.0,1.0,1.0],
                                        [1.0,1.0,1.0],
                                        [1.0,1.0,1.0],
                                        [1.0,1.0,1.0],
                                        [1.0,1.0,1.0],
                                        [1.0,1.0,1.0],
                                        [1.0,1.0,1.0],
                                        [1.0,1.0,1.0],
                                        [1.0,1.0,1.0],
                                        [1.0,1.0,1.0],
                                        [1.0,1.0,1.0],
                                        [1.0,1.0,1.0],
                                        [1.0,1.0,1.0],
                                        [1.0,1.0,1.0],
                                        [1.0,1.0,1.0],
                                        [1.0,1.0,1.0],
                                        [1.0,1.0,1.0],
                                        [1.0,1.0,1.0],
                                        [1.0,1.0,1.0],
                                        [1.0,1.0,1.0],
                                        [1.0,1.0,1.0],
                                        [1.0,1.0,1.0],
                                        [1.0,1.0,1.0],
                                        [1.0,1.0,1.0],
                                        [1.0,1.0,1.0],
                                        [1.0,1.0,1.0],
                                        [1.0,1.0,1.0],
                                        [1.0,1.0,1.0],
                                        [1.0,1.0,0.99],
                                        [1.0,1.0,0.99],
                                        [1.0,1.0,1.0],
                                        [1.0,1.0,1.0],
                                        [1.0,1.0,1.0],
                                        [1.0,1.0,0.99],
                                        [1.0,1.0,0.99],
                                        [1.0,1.0,1.0],
                                        [1.0,1.0,1.0],
                                        [1.0,1.0,0.99],
                                        [1.0,1.0,0.99],
                                        [1.0,1.0,0.99],
                                        [0.99,1.0,1.0],
                                        [0.99,1.0,1.0],
                                        [1.0,1.0,1.0],
                                        [1.0,1.0,1.0],
                                        [1.0,1.0,1.0],
                                        [1.0,1.0,1.0],
                                        [1.0,1.0,1.0],
                                        [1.0,1.0,1.0],
                                        [1.0,1.0,1.0],
                                        [1.0,1.0,1.0],
                                        [1.0,1.0,1.0],
                                        [1.0,1.0,1.0],
                                        [1.0,1.0,1.0],
                                        [1.0,1.0,1.0],
                                        [1.0,1.0,1.0],
                                        [1.0,1.0,1.0],
                                        [1.0,1.0,1.0],
                                        [1.0,1.0,1.0],
                                        [1.0,1.0,1.0],
                                        [1.0,1.0,1.0],
                                        [1.0,1.0,1.0],
                                        [1.0,1.0,1.0],
                                        [1.0,1.0,1.0],
                                        [1.0,1.0,1.0],
                                        [1.0,1.0,1.0],
                                        [1.0,1.0,1.0],
                                        [1.0,1.0,1.0],
                                        [1.0,1.0,1.0],
                                        [1.0,1.0,1.0],
                                        [1.0,1.0,1.0],
                                        [1.0,1.0,1.0],
                                        [1.0,1.0,1.0],
                                        [1.0,1.0,1.0],
                                        [1.0,1.0,1.0],
                                        [1.0,1.0,1.0],
                                        [1.0,1.0,1.0],
                                        [1.0,1.0,1.0],
                                        [1.0,1.0,1.0],
                                        [1.0,1.0,1.0],
                                        [1.0,1.0,1.0],
                                        [1.0,1.0,1.0],
                                        [1.0,1.0,1.0],
                                        [1.0,1.0,1.0],
                                        [1.0,1.0,1.0],
                                        [1.0,1.0,0.99],
                                        [1.0,1.0,1.0],
                                        [1.0,1.0,1.0],
                                        [1.0,1.0,0.99],
                                        [1.0,1.0,0.99],
                                        [0.99,1.0,0.99],
                                        [0.99,1.0,0.99],
                                        [0.99,1.0,0.99],
                                        [0.99,1.0,1.0],
                                        [0.99,1.0,1.0],
                                        [0.99,1.0,0.99],
                                        [0.99,1.0,0.99],
                                        [0.99,1.0,0.99],
                                        [0.99,1.0,0.99],
                                        [0.99,1.0,0.99],
                                        [0.99,1.0,0.99],
                                        [0.99,1.0,0.99],
                                        [0.99,1.0,0.99],
                                        [0.99,1.0,0.99],
                                        [0.99,1.0,0.99],
                                        [0.99,1.0,0.99],
                                        [0.99,1.0,0.99],
                                        [0.99,1.0,0.99],
                                        [1.0,1.0,0.99],
                                        [1.0,1.0,0.99],
                                        [0.99,1.0,0.99],
                                        [0.99,1.0,0.99],
                                        [0.99,1.0,0.99],
                                        [0.99,1.0,0.99],
                                        [0.99,1.0,0.99],
                                        [0.99,1.0,0.99],
                                        [0.99,1.0,0.99],
                                        [0.99,1.0,0.99],
                                        [0.99,1.0,0.99],
                                        [0.99,1.0,0.99],
                                        [0.99,1.0,0.99],
                                        [0.99,1.0,0.99],
                                        [0.99,1.0,0.99],
                                        [0.99,1.0,0.99],
                                        [0.99,1.0,0.99],
                                        [0.99,1.0,0.99],
                                        [0.99,1.0,0.99],
                                        [0.99,1.0,0.99],
                                        [0.99,1.0,0.99],
                                        [0.99,1.0,0.99],
                                        [0.99,1.0,0.99],
                                        [0.99,1.0,0.99],
                                        [0.99,1.0,0.99],
                                        [0.99,1.0,0.99],
                                        [0.99,1.0,0.99],
                                        [0.99,1.0,0.99],
                                        [0.99,1.0,0.99],
                                        [0.99,1.0,0.99],
                                        [0.99,1.0,0.99],
                                        [0.99,1.0,0.99],
                                        [0.99,1.0,0.99],
                                        [0.99,1.0,0.98],
                                        [0.99,1.0,0.98],
                                        [0.99,1.0,0.98],
                                        [0.99,1.0,0.98],
                                        [0.99,1.0,0.99],
                                        [0.99,1.0,0.98],
                                        [0.99,1.0,0.98],
                                        [0.99,1.0,0.99],
                                        [0.99,1.0,0.99],
                                        [0.98,1.0,0.98],
                                        [0.99,1.0,0.98],
                                        [0.99,1.0,0.98],
                                        [0.99,1.0,0.98],
                                        [0.99,1.0,0.98],
                                        [0.99,1.0,0.98],
                                        [0.98,1.0,0.98],
                                        [0.98,1.0,0.98],
                                        [0.99,1.0,0.98],
                                        [0.99,1.0,0.98],
                                        [0.99,1.0,0.98],
                                        [0.99,1.0,0.98],
                                        [0.99,1.0,0.98],
                                        [0.99,1.0,0.98],
                                        [0.99,1.0,0.98],
                                        [0.99,1.0,0.98],
                                        [0.99,1.0,0.98],
                                        [0.99,1.0,0.98],
                                        [0.98,1.0,0.98],
                                        [0.98,1.0,0.98],
                                        [0.98,1.0,0.98],
                                        [0.99,1.0,0.98],
                                        [0.99,1.0,0.98],
                                        [0.98,1.0,0.98],
                                        [0.98,1.0,0.98],
                                        [0.98,1.0,0.98],
                                        [0.98,1.0,0.98],
                                        [0.98,1.0,0.98],
                                        [0.98,1.0,0.98],
                                        [0.98,1.0,0.98],
                                        [0.98,1.0,0.98],
                                        [0.98,1.0,0.98],
                                        [0.98,1.0,0.98],
                                        [0.99,1.0,0.98],
                                        [0.99,1.0,0.98],
                                        [0.98,1.0,0.98],
                                        [0.98,1.0,0.98],
                                        [0.98,1.0,0.98],
                                        [0.98,1.0,0.98],
                                        [0.98,1.0,0.98],
                                        [0.98,1.0,0.98],
                                        [0.98,1.0,0.98],
                                        [0.98,1.0,0.98],
                                        [0.98,1.0,0.98],
                                        [0.98,1.0,0.98],
                                        [0.98,1.0,0.97],
                                        [0.97,0.99,0.97],
                                        [0.97,0.99,0.97],
                                        [0.97,0.99,0.97],
                                        [0.97,0.99,0.97],
                                        [0.97,0.99,0.97],
                                        [0.97,0.99,0.97],
                                        [0.97,0.99,0.97],
                                        [0.97,0.99,0.97],
                                        [0.97,0.99,0.97],
                                        [0.97,0.99,0.97],
                                        [0.97,0.99,0.97],
                                        [0.97,0.99,0.97],
                                        [0.97,0.99,0.97],
                                        [0.97,0.99,0.97],
                                        [0.97,0.99,0.97],
                                        [0.97,0.99,0.97],
                                        [0.97,0.99,0.97],
                                        [0.97,0.99,0.97],
                                        [0.97,0.99,0.97],
                                        [0.97,0.99,0.97],
                                        [0.97,0.99,0.97],
                                        [0.97,0.99,0.98],
                                        [0.97,0.99,0.97],
                                        [0.97,0.99,0.97],
                                        [0.97,0.99,0.97],
                                        [0.97,0.99,0.97],
                                        [0.98,0.99,0.97],
                                        [0.98,0.99,0.98],
                                        [0.98,0.99,0.98],
                                        [0.98,0.99,0.98],
                                        [0.98,0.99,0.98],
                                        [0.98,1.0,0.98],
                                        [0.98,0.99,0.98],
                                        [0.98,0.99,0.98],
                                        [0.98,1.0,0.98],
                                        [0.98,1.0,0.98],
                                        [0.98,1.0,0.98],
                                        [0.98,1.0,0.98],
                                        [0.98,1.0,0.98],
                                        [0.98,1.0,0.98],
                                        [0.98,1.0,0.98],
                                        [0.98,1.0,0.98],
                                        [0.98,1.0,0.98],
                                        [0.98,1.0,0.98],
                                        [0.98,1.0,0.98],
                                        [0.98,1.0,0.98],
                                        [0.99,1.0,0.98],
                                        [0.98,1.0,0.99],
                                        [0.98,1.0,0.99],
                                        [0.99,1.0,0.98],
                                        [0.99,1.0,0.98],
                                        [0.98,1.0,0.99],
                                        [0.98,1.0,0.99],
                                        [0.98,1.0,0.99],
                                        [0.99,1.0,0.98],
                                        [0.99,1.0,0.98],
                                        [0.99,1.0,0.99],
                                        [0.99,1.0,0.99],
                                        [0.99,1.0,0.99],
                                        [0.98,1.0,0.98],
                                        [0.98,1.0,0.98],
                                        [0.99,1.0,0.99],
                                        [0.99,1.0,0.99],
                                        [0.99,1.0,0.99],
                                        [0.99,1.0,0.99],
                                        [0.99,1.0,0.99],
                                        [0.99,1.0,0.99],
                                        [0.99,1.0,0.99],
                                        [0.99,1.0,0.99],
                                        [0.99,1.0,0.99],
                                        [0.99,1.0,0.99],
                                        [0.99,1.0,0.99],
                                        [0.99,1.0,0.99],
                                        [0.99,1.0,0.99],
                                        [0.99,1.0,0.99],
                                        [0.99,1.0,0.99],
                                        [0.99,1.0,0.99],
                                        [0.99,1.0,0.99],
                                        [0.99,1.0,0.99],
                                        [0.99,1.0,0.99],
                                        [0.99,1.0,0.99],
                                        [0.99,1.0,0.99],
                                        [0.99,1.0,0.99],
                                        [0.99,1.0,0.99],
                                        [0.99,1.0,0.99],
                                        [0.99,1.0,0.99],
                                        [0.99,1.0,0.99],
                                        [0.98,1.0,0.99],
                                        [0.98,1.0,0.99],
                                        [0.99,1.0,0.99],
                                        [0.99,1.0,0.99],
                                        [0.99,1.0,0.99],
                                        [0.99,1.0,0.98],
                                        [0.99,1.0,0.98],
                                        [0.98,1.0,0.98],
                                        [0.98,1.0,0.98],
                                        [0.98,1.0,0.98],
                                        [0.99,1.0,0.99],
                                        [0.99,1.0,0.99],
                                        [0.99,1.0,0.98],
                                        [0.99,1.0,0.98],
                                        [0.99,0.99,0.99],
                                        [0.99,1.0,0.98],
                                        [0.99,1.0,0.98],
                                        [0.99,1.0,0.98],
                                        [0.99,1.0,0.98],
                                        [0.98,1.0,0.98],
                                        [0.98,1.0,0.98],
                                        [0.98,1.0,0.98],
                                        [0.98,0.99,0.97],
                                        [0.98,0.99,0.97],
                                        [0.98,0.99,0.98],
                                        [0.98,0.99,0.97],
                                        [0.98,0.99,0.97],
                                        [0.98,0.99,0.98],
                                        [0.98,0.99,0.98],
                                        [0.98,0.99,0.98],
                                        [0.98,0.99,0.98],
                                        [0.98,0.99,0.97],
                                        [0.98,0.99,0.98],
                                        [0.98,0.99,0.98],
                                        [0.98,0.99,0.98],
                                        [0.98,0.99,0.98],
                                        [0.99,0.99,0.98],
                                        [0.98,1.0,0.98],
                                        [0.98,1.0,0.98],
                                        [0.98,1.0,0.98],
                                        [0.98,1.0,0.98],
                                        [0.98,0.99,0.98],
                                        [0.99,0.99,0.98],
                                        [0.99,0.99,0.98],
                                        [0.99,1.0,0.99],
                                        [0.99,1.0,0.99],
                                        [0.99,1.0,0.99],
                                        [0.99,1.0,0.99],
                                        [0.99,1.0,0.99],
                                        [0.99,1.0,0.99],
                                        [0.99,1.0,0.99],
                                        [0.99,1.0,0.99],
                                        [0.99,1.0,0.99],
                                        [0.99,1.0,0.99],
                                        [0.99,1.0,0.99],
                                        [0.99,1.0,0.99],
                                        [0.99,1.0,0.99],
                                        [0.99,1.0,0.99],
                                        [0.99,1.0,0.99],
                                        [0.99,1.0,0.99],
                                        [0.99,1.0,0.99],
                                        [0.99,1.0,0.99],
                                        [0.99,1.0,0.99],
                                        [0.99,1.0,0.99],
                                        [0.99,1.0,0.99],
                                        [0.99,1.0,0.99],
                                        [0.99,1.0,0.99],
                                        [0.99,1.0,0.99],
                                        [0.99,1.0,0.99],
                                        [0.99,1.0,0.99],
                                        [0.99,1.0,0.99],
                                        [0.99,1.0,0.99],
                                        [0.99,1.0,0.99],
                                        [0.99,1.0,0.99],
                                        [0.99,1.0,0.99],
                                        [0.99,1.0,0.99],
                                        [0.99,1.0,0.99],
                                        [0.99,1.0,0.99],
                                        [0.99,1.0,0.99],
                                        [0.99,1.0,0.98],
                                        [0.99,1.0,0.98],
                                        [0.99,1.0,0.99],
                                        [0.99,1.0,0.99],
                                        [0.99,1.0,0.99],
                                        [0.99,1.0,0.99],
                                        [0.99,1.0,0.99],
                                        [0.99,1.0,0.99],
                                        [0.99,1.0,0.99],
                                        [0.99,1.0,0.99],
                                        [0.99,1.0,0.99],
                                        [0.99,1.0,0.99],
                                        [0.99,1.0,0.99],
                                        [0.99,1.0,0.98],
                                        [0.99,1.0,0.98],
                                        [0.99,1.0,0.99],
                                        [0.99,1.0,0.99],
                                        [0.99,1.0,0.99],
                                        [0.99,1.0,0.98],
                                        [0.99,1.0,0.98],
                                        [0.99,1.0,0.98],
                                        [0.99,1.0,0.98],
                                        [0.99,1.0,0.98],
                                        [0.99,1.0,0.99],
                                        [0.99,1.0,0.99],
                                        [0.99,1.0,0.99],
                                        [0.99,1.0,0.99],
                                        [0.99,1.0,0.98],
                                        [0.99,1.0,0.99],
                                        [0.99,1.0,0.99],
                                        [0.99,1.0,0.99],
                                        [0.99,1.0,0.99],
                                        [0.99,1.0,0.98],
                                        [0.99,1.0,0.99],
                                        [0.99,1.0,0.99],
                                        [0.99,1.0,0.99],
                                        [0.99,1.0,0.99],
                                        [0.98,1.0,0.98],
                                        [0.99,1.0,0.99],
                                        [0.99,1.0,0.99],
                                        [0.99,1.0,0.98],
                                        [0.99,1.0,0.98],
                                        [0.99,1.0,0.98],
                                        [0.99,0.99,0.98],
                                        [0.99,0.99,0.98],
                                        [0.98,1.0,0.98],
                                        [0.98,1.0,0.98],
                                        [0.99,0.99,0.98]], columns=['FPA', 'FPB', 'FPV'])

    
    
    dummy= pandas.concat([dummy_week] * qty_weeks, ignore_index=True)
    
    end_date_dt = start_date_dt  + dt.timedelta(days=qty_weeks*7)
    
    dummy_idx = numpy.arange(start_date_dt, end_date_dt, numpy.timedelta64(5, 'm'), dtype='datetime64')
    
    dummy['timestamp'] = dummy_idx
    dummy.set_index('timestamp', inplace=True)    
    dummy.index = pandas.to_datetime(dummy.index)
    

    step_vet = numpy.zeros(dummy.shape[0])

    # Load transfer
    for i in range(0, random.randint(1, 4)):
        start = random.randint(0, dummy.shape[0])
        end = start + random.randint(1, 60) * 24 * 12

        if end >= len(step_vet):
            end = len(step_vet)

        step_vet[start:end] = -0.2 * random.random()

    # Noise
    for i in range(0, random.randint(1, 40)):
        start = random.randint(0, dummy.shape[0])
        end = start + random.randint(1, 12 * 3)

        if end >= len(step_vet):
            end = len(step_vet)

        step_vet[start:end] = random.randint(-10,2)*0.07

    dummy['FPA'] = dummy['FPA'].values + step_vet
    dummy['FPB'] = dummy['FPB'].values + step_vet
    dummy['FPV'] = dummy['FPV'].values + step_vet

    return dummy


def PowerDummyData(qty_weeks:int = 12*4,start_date_dt:datetime = datetime(2023,1,1)):
    """
    Generates dummy power data for a specified number of weeks from a start date.
    
    This function calculates the apparent power (S), active power (P), and reactive power (Q)
    for a given number of weeks starting from a specified date. It uses the CurrentDummyData, 
    VoltageDummyData, and PowerFactorDummyData functions to generate current (I), voltage (V), 
    and power factor (pf) data, respectively. The final DataFrame includes columns for S, P, and Q.
    
    Parameters:
    qty_weeks (int): The quantity of weeks for which to generate data. Default is 48 weeks.
    start_date_dt (datetime): The start date for data generation. Default is January 1, 2023.
    
    Returns:
    pandas.DataFrame: A DataFrame containing the columns 'S' (apparent power), 
                      'P' (active power), and 'Q' (reactive power).
    
    Example:
    >>> PowerDummyData(4, datetime(2023, 1, 1))
    [Output will be a DataFrame with the calculated power data for 4 weeks starting from January 1, 2023]
    """
    
    end_date_dt = start_date_dt  + dt.timedelta(days=qty_weeks*7)
    
    I = CurrentDummyData(qty_weeks, start_date_dt)
    V = VoltageDummyData(qty_weeks, start_date_dt)
    pf = PowerFactorDummyData(qty_weeks, start_date_dt)

    I = I.iloc[:, :-1]

    dummy = pandas.DataFrame([])

    dummy['S'] = V['VA'] / numpy.sqrt(3) * I['IA'] + V['VB'] / numpy.sqrt(3) * I['IB'] \
                                                   + V['VV'] / numpy.sqrt(3) * I['IV']
    dummy['P'] = V['VA'] / numpy.sqrt(3) * I['IA'] * pf['FPA'] + V['VB'] / numpy.sqrt(3) * I['IB'] * pf['FPB'] \
                                                               + V['VV'] / numpy.sqrt(3) * I['IV'] * pf['FPV']
    dummy['Q'] = dummy['S'].pow(2) - dummy['P'].pow(2)
    dummy['Q'] = numpy.sqrt(dummy['Q'])

    return dummy


def EnergyDummyData(qty_weeks:int = 12*4,start_date_dt:datetime = datetime(2023,1,1)):
    """
    Generate a dummy pandas DataFrame containing cumulative energy data.
    
    This function creates a DataFrame with two columns: 'Eactive' and 'Ereactive'.
    'Eactive' is the cumulative sum of the 'P' column from the PowerDummyData function,
    and 'Ereactive' is the absolute cumulative sum of the 'Q' column from the same function.
    
    Parameters
    ----------
    qty_weeks : int, optional
        The quantity of weeks for which to generate the data, default is 48 weeks (12*4).
    start_date_dt : datetime, optional
        The starting date for the data generation, default is January 1, 2023.
    
    Returns
    -------
    pandas.DataFrame
        A DataFrame with two columns 'Eactive' and 'Ereactive' representing the cumulative
        active and reactive energy data respectively.
    
    Examples
    --------
    >>> EnergyDummyData(4, datetime(2023, 1, 1))
    DataFrame with the cumulative energy data for 4 weeks starting from January 1, 2023.
    
    Notes
    -----
    The function relies on PowerDummyData function to generate initial power data
    which is then cumulatively summed to generate energy data.
    """

    dummy_s = PowerDummyData(qty_weeks, start_date_dt)

    dummy = pandas.DataFrame([])

    dummy['Eactive'] = dummy_s['P'].cumsum(skipna=True)

    dummy['Ereactive'] = dummy_s['Q'].abs().cumsum(skipna=True)

    return dummy


def ShowExampleSimpleProcess(plot: bool = True):
    
    """
    Demonstrates a simple data processing workflow using various functions to handle, analyze, and visualize data.
    
    The function executes a sequence of operations on dummy data, including data synchronization, outlier removal, 
    and data processing. It uses matplotlib to plot the results at each step. Additionally, it tracks the execution
    time for each step using the TimeProfile function and the number of missing data samples, outputting this 
    information to the console along with some explanation.
    
    
    Steps involved:
    - Close all existing matplotlib plots.
    - Generate dummy data and plot it.
    - Synchronize data with specified start and end dates.
    - Remove outliers using various methods (Hard Threshold, MMADMM, Quantile, Histogram).
    - Execute a simple data fill processing operation.
    - Plot the final output.
    - Display a time profile of the entire process.
    
        
    :param plot: Plot data for each step of the process. Disables the time profile.
    :type plot: bool,optional
    
    Returns:
        None: This function does not return any value.
    """
    
    #-------INPUT-------#
    if(plot):
        matplotlib.pyplot.close('all')
    
    
    start_date_dt = datetime(2023,1,1)
    end_date_dt = datetime(2025,1,1)
    
    print("In this example, the data loss and outliers are exaggerated for demonstration purposes.")
    print(f"The data will be processed between {start_date_dt} and {end_date_dt}")

    dummy = CurrentDummyData(qty_weeks=90)
    dummy.drop(columns = ['IN'],inplace=True)    
    
    if(plot):
        ax = dummy.plot(title = "1 - Three phase current")
        ax.set_ylabel("Current [A]")

    time_stopper = [['time_init', time.perf_counter()]]   
    
    #-------DATA SYNCHRONIZATION-------#
    
    output = DataSynchronization(dummy, start_date_dt, end_date_dt, sample_freq=5, sample_time_base='m')
    
    if(plot):
        ax = output.plot(title = "2 - Three phase current Sync")
        ax.set_ylabel("Current [A]")    

    print("Data Synchronization process done.")
    CountMissingData(output, show=True)
    time_stopper.append(['DataSynchronization', time.perf_counter()])
    
    #-------OUTLIER REMOVAL-------#
    output = RemoveOutliersHardThreshold(output, hard_max=500, hard_min=0)

    if(plot):
        ax = output.plot(title = "3 - Remove Outliers Hard Threshold")
        ax.set_ylabel("Current [A]")       
        
    print("Outliers removed using Hard Threshold.")
    CountMissingData(output, show=True)
    time_stopper.append(['RemoveOutliersHardThreshold', time.perf_counter()])
    
    
    output = RemoveOutliersMMADMM(output, len_mov_avg=3, std_def=4, plot=False)

    if(plot):
        ax = output.plot(title = "4 - Remove Outliers MMADMM")
        ax.set_ylabel("Current [A]")    

    print("Outliers removed using MMADMM.")
    CountMissingData(output, show=True)
    time_stopper.append(['RemoveOutliersMMADMM', time.perf_counter()])
    
    
    output = RemoveOutliersQuantile(output)

    if(plot):
        ax = output.plot(title = "5 - Remove Outliers Quantile")
        ax.set_ylabel("Current [A]")    

    print("Outliers removed using Quantile.")
    CountMissingData(output, show=True)
    time_stopper.append(['RemoveOutliersQuantile', time.perf_counter()])
    
    
    output = RemoveOutliersHistogram(output, min_number_of_samples_limit=12 * 5)
    
    if(plot):
        ax = output.plot(title = "6 - Remove Outliers Histogram")
        ax.set_ylabel("Current [A]")    

    print("Outliers removed using Histogram.")
    CountMissingData(output, show=True)
    time_stopper.append(['RemoveOutliersHistoGram', time.perf_counter()])

    #-------SIMPLE INPUTATION PROCESS-------#

    output = SimpleProcess(output, start_date_dt, end_date_dt,
                           sample_freq=5,
                           sample_time_base='m',
                           pre_interpol=1,
                           pos_interpol=6,
                           prop_phases=True,
                           integrate=True,
                           interpol_integrate=100)
    
    if(plot):
        ax = output.plot(title = "7 - Simple process data fill")
        ax.set_ylabel("Current [A]") 
        
        
    print("Simple pre-made imputation process.\nORGANIZE->INTERPOLATE->PHASE_PROPORTION->INTERPOLATE->INTEGRATE->INTERPOLATE")
    CountMissingData(output, show=True)
    time_stopper.append(['SimpleProcessInput', time.perf_counter()])
    
    if(plot):
        matplotlib.pyplot.show()
    
    TimeProfile(time_stopper, name='Simple Process', show=~plot, estimate_for=1000 * 5)

    return



def ShowExampleNSSCProcess(plot: bool = True):
    """
    Demonstrates the normalized scaled standard weekday curve inputation method.

    This function goes through various steps of data processing including synchronization, outlier removal, data filling, and NSSC application. Each step is demonstrated with optional plotting for visual analysis. The data loss and outliers are exaggerated for demonstration purposes. The process is applied between a predefined start and end date, with multiple methods applied to handle missing data, outliers, and to predict and replace data in the final output.

    Parameters
    ----------
    plot : bool, optional
        If True, the function will plot the data at various stages of processing for visualization. 
        Defaults to True.

    Returns
    -------
    None
        The function does not return any value but optionally displays plots and prints information about the processing steps if `plot` is True.

    Notes
    -----
    - The function is meant for demonstration and educational purposes, showing various stages in data processing.
    - The process is specifically tailored for current data and may need adjustments for other types of data.
    - The example dates and parameters are hardcoded for demonstration and should be adapted for practical use.

    """
    
    if(plot):
        matplotlib.pyplot.close('all')
    
    #-----INPUT-----#
    start_date_dt = datetime(2023,1,1)  
    end_date_dt = datetime(2025,1,1)
    
    print("In this example, the data loss and outliers are exaggerated for demonstration purposes.")

    print(f"The data will be processed between {start_date_dt} and {end_date_dt}")

    dummy = CurrentDummyData(qty_weeks=90)
    dummy.drop(columns = ['IN'],inplace=True)
    
    #CREATE MISSING DATA ON ONE PHASE FOR A PERIOD AND THREE PHASE FOR ANOTHER
    dummy.iloc[12*24*7*40:12*24*7*50,2] = numpy.nan    
    dummy.iloc[12*24*7*60:12*24*7*64,:] = numpy.nan
    
    if(plot):
        ax = dummy.plot(title = "1 - Three phase current")
        ax.set_ylabel("Current [A]") 
    
    time_stopper = [['time_init', time.perf_counter()]]    
    
    #-----DATA SYNC-----#
    
    output = DataSynchronization(dummy, start_date_dt, end_date_dt, sample_freq=5, sample_time_base='m')     
        
    if(plot):
        ax = dummy.plot(title = "2 - Three phase Sync")
        ax.set_ylabel("Current [A]") 
    
    print("Data Synchronization process done.")
    CountMissingData(output, show=True)
    time_stopper.append(['DataSynchronization', time.perf_counter()])
    
        
    #-----OUTLIER REMOVAL-----#
    
    output = RemoveOutliersHardThreshold(output, hard_max=500, hard_min=0)
    
    if(plot):
        ax = output.plot(title = "3 - Remove Outliers Hard Threshold")
        ax.set_ylabel("Current [A]")   
    
    print("Outliers removed using Hard Threshold.")
    CountMissingData(output, show=True)    
    time_stopper.append(['RemoveOutliersHardThreshold', time.perf_counter()])
       
    
    output = RemoveOutliersMMADMM(output, len_mov_avg=25, std_def=3, plot=False,min_var_def=3)#, remove_from_process=['IN'])

    if(plot):
        ax = output.plot(title = "4 - Remove Outliers MMADMM")
        ax.set_ylabel("Current [A]")    

    print("Outliers removed using MMADMM.")
    CountMissingData(output, show=True)
    time_stopper.append(['RemoveOutliersMMADMM', time.perf_counter()])

    
    output = RemoveOutliersQuantile(output)
    
    if(plot):
        ax = output.plot(title = "5 - Remove Outliers Quantile")
        ax.set_ylabel("Current [A]")    
    
    print("Outliers removed using Quantile.")
    CountMissingData(output, show=True)
    time_stopper.append(['RemoveOutliersQuantile', time.perf_counter()])
    

    output = RemoveOutliersHistogram(output, min_number_of_samples_limit=12 * 5)
    
    if(plot):
        ax = output.plot(title = "6 - Remove Outliers Histogram")
        ax.set_ylabel("Current [A]")   
    
    print("Outliers removed using Histogram.")
    CountMissingData(output, show=True)
    time_stopper.append(['RemoveOutliersHistoGram', time.perf_counter()])
    

    #-----DATA FILL-----#

    output = PhaseProportionInput(output, threshold_accept=0.60,plot =False,apply_filter=True)        
    
    if(plot):
        ax = output.plot(title = "7 - Phase proportion fill")
        ax.set_ylabel("Current [A]") 

    print("Processes input DataFrame to compute phase proportion based on various time frames and criteria.")    
    CountMissingData(output, show=True)
    time_stopper.append(['PhaseProportionInput', time.perf_counter()])
 
    
    #----NSSC----#
 
    print("In this example the NSSC is implemented in parts to be able to print and demonstrate each step. In production use the NSSCInput.")    
    
    print("Calculate the day max/min values")
    # Get day max/min values
    max_vet,max_vet_idx = GetDayMaxMin(output,start_date_dt,end_date_dt,sample_freq=5,threshold_accept=0.5,exe_param='max')         
    min_vet,min_vet_idx = GetDayMaxMin(output,start_date_dt,end_date_dt,sample_freq=5,threshold_accept=0.5,exe_param='min') 
    
    print("Get standard WeekDay Curve")
    weekday_curve = GetWeekDayCurve(output, sample_freq=5, threshold_accept=0.8,min_sample_per_day=3,min_sample_per_workday=9)

    #For plot purpose
    list_df_print = []
    for col in max_vet.columns:        
        df_print = pandas.DataFrame(index=max_vet_idx[col].values, dtype=object)
        df_print[col+'_max'] = max_vet[col].values        
        df_print = DataSynchronization(df_print, start_date_dt, end_date_dt, sample_freq=5, sample_time_base='m')             
        list_df_print.append(df_print)
        
    
    for col in min_vet.columns:        
        df_print = pandas.DataFrame(index=min_vet_idx[col].values, dtype=object)
        df_print[col+'_min'] = min_vet[col].values        
        df_print = DataSynchronization(df_print, start_date_dt, end_date_dt, sample_freq=5, sample_time_base='m')             
        list_df_print.append(df_print)
            
    df_print = pandas.DataFrame([])
    for col in list_df_print:  
        if(df_print.size==0):
            df_print = col                    
        else:
            df_print = pandas.concat((df_print,col),axis=1)
            
    
    if(plot):
        ax = output.plot(title = "8 - Min/Max value for each valid day")
        ax.set_ylabel("Current [A]") 
        df_print.plot.line(ax=ax,color='red',style='.')    
    
    
    print("Makes the prediction.")
    X_pred = GetNSSCPredictedSamples(max_vet, min_vet, weekday_curve,start_date_dt,end_date_dt, sample_freq=5,sample_time_base='m')
    
    if(plot):
        ax = X_pred.plot(title = "9 - The predicted time series based on the weekday curve and min/max values")
        ax.set_ylabel("Current [A]") 
    
    time_stopper.append(['X_pred', time.perf_counter()])
    

    output = ReplaceData(output,
                X_pred,
                start_date_dt,
                end_date_dt,   
                num_samples_day = 12*24,
                day_threshold=0.5,
                patamar_threshold = 0.5,
                num_samples_patamar = 12*6,                
                sample_freq= 5,
                sample_time_base = 'm')

    if(plot):
        ax = output.plot(title = "10 - Output (NSSC)")
        ax.set_ylabel("Current [A]") 
    

    print("NSSC Method applied.")    
    CountMissingData(output, show=True)
    time_stopper.append(['Output (NSSC)', time.perf_counter()])    

    if(plot):
        matplotlib.pyplot.show()
    else:
        TimeProfile(time_stopper, name='NSSC', show=~plot, estimate_for=1000 * 5)
    
    return