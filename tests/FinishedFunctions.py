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

    if show:
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
                4: 1, 5: 1, 6: 1, 7: 1, 9: 1}

    season = month.map(map_dict)

    return season


def PhaseProportionInput(x_in: pandas.core.frame.DataFrame,
                         threshold_accept: float = 0.75,
                         remove_from_process: list = []) -> pandas.core.frame.DataFrame:
    """
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


    :param x_in: A pandas.core.frame.DataFrame where the index is of type "pandas.core.indexes.datetime.DatetimeIndex"
    and each column contain an electrical quantity time series.
    :type x_in: pandas.core.frame.DataFrame

    :param threshold_accept: The minimum amount of samples to accept. Defaults to 0.75 (75%).
    :type threshold_accept: float,optional

    :param remove_from_process: Columns to be kept off the process;
    :type remove_from_process: list,optional

    :raises Exception: if x_in has less than two columns to process.
    :raises Exception: if x_in has no DatetimeIndex.

    :return: Y: The pandas.core.frame.DataFrame with samples filled based on the proportion between time series.
    :rtype: Y: pandas.core.frame.DataFrame

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

    # make output vector
    Y = X.copy(deep=True)

    time_stopper.append(['Copy', time.perf_counter()])
    # -------------------------#
    #          HOUR           #
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

    time_stopper.append(['Hour-Y', time.perf_counter()])

    time_stopper.append(['Hour', time.perf_counter()])

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

    time_stopper.append(['Patamar', time.perf_counter()])
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

    time_stopper.append(['Day', time.perf_counter()])
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

    time_stopper.append(['Month', time.perf_counter()])
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

    time_stopper.append(['Season', time.perf_counter()])

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

    time_stopper.append(['Year', time.perf_counter()])

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

    time_stopper.append(['AllTimeSeries', time.perf_counter()])

    # return the keep out columns
    if len(remove_from_process) > 0:
        Y = pandas.concat([Y, x_in.loc[:, remove_from_process]], axis=1)

    time_stopper.append(['Final', time.perf_counter()])

    TimeProfile(time_stopper, name='Phase', show=False)

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
        ax = X_moving_median.plot()
        x_in.plot(ax=ax)
        X_mad.plot(ax=ax)
        X_moving_down.plot(ax=ax)
        X_moving_up.plot(ax=ax)
        Y.plot()

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


def RemoveOutliersHistoGram(x_in: pandas.core.frame.DataFrame,
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
                Y = pandas.core.frame.concat((Y, Y_day_aux))
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
    Generate predicted samples for NS-SSC using maximum and minimum vectors, 
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
    weekday_curve_vet_mid = pandas.core.frame.concat([weekday_curve] * num_mid_weeks)
    num_end_days = num_days - num_mid_weeks * 7 - (7 - first_day)
    weekday_curve_vet_end = weekday_curve.iloc[:num_end_days * (12 * 24), :].reset_index(drop=True)

    weekday_curve_vet = pandas.core.frame.concat([weekday_curve_vet_begin, weekday_curve_vet_mid, weekday_curve_vet_end])

    weekday_curve_vet = weekday_curve_vet.reset_index(drop=True)

    print(weekday_curve_vet)
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
    mark_substitute = pandas.core.frame.merge(mark_substitute, output_isnull_patamar,left_on='index_patamar',right_index=True,how='left').fillna(False)
    for col in x_in.columns.values:
        mark_substitute[col] = mark_substitute[col+'_mark']
        mark_substitute.drop(columns=[col+'_mark'],axis=1,inplace=True)
        
    mark_substitute.drop(columns=['index_patamar'],axis=1,inplace=True)
    
    mark_substitute['index_day'] = index_day.values
    mark_substitute = pandas.core.frame.merge(mark_substitute, output_isnull_day,left_on='index_day',right_index=True,how='left').fillna(False)    
    
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

def CurrentDummyData(start_date: str = '2021-01-01', final_date: str = '2023-01-01'):
    dummy_day = pandas.DataFrame([[129.2, 122.5, 118.8, 4.7],
                                  [126.3, 122.5, 118.8, 4.7],
                                  [131.3, 123, 120.9, 5.6],
                                  [126.9, 121.5, 117.8, 5],
                                  [126.9, 121.5, 117.8, 5],
                                  [125.4, 118.8, 116.6, 5],
                                  [125.4, 118.8, 118.5, 4.6],
                                  [124.8, 117.1, 116.6, 4.6],
                                  [125.7, 118.8, 116.1, 5],
                                  [125.7, 118.8, 113.5, 4.6],
                                  [125.1, 117.4, 115, 4.5],
                                  [125.1, 117.4, 115, 4.5],
                                  [122.8, 115.3, 113.5, 4.7],
                                  [122.4, 113.8, 114.7, 4.7],
                                  [122.4, 113.8, 114.7, 4.7],
                                  [120.9, 114.9, 111.5, 4.6],
                                  [120.9, 111.9, 107.8, 4.6],
                                  [120.2, 114.1, 107.8, 4.6],
                                  [116.2, 114.1, 107.8, 4.6],
                                  [119.1, 114.5, 108.3, 4.8],
                                  [119.3, 112.2, 110.5, 4.9],
                                  [116.3, 112.2, 110.5, 4.9],
                                  [117.7, 110.7, 109.1, 4.6],
                                  [117.7, 110.7, 109.1, 4.7],
                                  [116.9, 109.4, 107.6, 4.7],
                                  [116.3, 110.5, 109.8, 4.5],
                                  [116.3, 110.5, 109.8, 4.5],
                                  [117, 110.8, 107.5, 4.3],
                                  [117, 110.8, 107.5, 4.3],
                                  [116.3, 110.3, 107.5, 4.6],
                                  [116, 106.7, 108, 4.2],
                                  [116, 106.7, 108, 4.2],
                                  [113.7, 108.1, 105.5, 4.5],
                                  [113.7, 108.1, 105.5, 4.5],
                                  [115.9, 106.8, 106.6, 4.7],
                                  [113.8, 108.6, 105.7, 4.9],
                                  [116.4, 108.6, 105.7, 4.8],
                                  [114.1, 107.5, 107.8, 4.6],
                                  [114.1, 107.5, 107.8, 4.6],
                                  [114.7, 108.4, 105.6, 4.5],
                                  [111.3, 108.1, 105.4, 4.1],
                                  [111.3, 108.1, 102.8, 4.1],
                                  [112.5, 105.7, 103.9, 4.4],
                                  [112.5, 105.7, 103.9, 4.4],
                                  [111.1, 106.5, 103.5, 4.2],
                                  [112, 106, 104, 4.2],
                                  [112, 106, 104, 4.2],
                                  [111.5, 106.5, 102.5, 4.3],
                                  [111.5, 106.5, 102.5, 4.3],
                                  [112.3, 106.5, 104.4, 4.5],
                                  [112, 107.1, 104.7, 4.5],
                                  [112, 107.1, 104.7, 4.5],
                                  [113.9, 107.5, 104.6, 4.4],
                                  [113.9, 107.5, 104.6, 4.4],
                                  [110.9, 105.3, 101.9, 4.5],
                                  [111.4, 105.7, 103.2, 4.4],
                                  [111.4, 105.7, 103.2, 4.4],
                                  [111.4, 106.2, 107.8, 4.5],
                                  [111.4, 106.2, 102.7, 4.5],
                                  [110.6, 108.3, 101.8, 4.3],
                                  [109.6, 105, 102.6, 4.7],
                                  [106.3, 105, 102.6, 4.7],
                                  [111.3, 105.2, 102.5, 4.6],
                                  [111.3, 105.2, 102.5, 4.6],
                                  [110.1, 101.8, 102.5, 4.7],
                                  [110.3, 105.8, 102, 4.3],
                                  [110.3, 101.9, 102, 4.3],
                                  [107.3, 101.3, 98.2, 3.7],
                                  [107.3, 101.3, 102.7, 4.8],
                                  [105.6, 103.2, 97.7, 4.2],
                                  [107.4, 103.7, 100.8, 4.4],
                                  [107.4, 103.5, 104.9, 4.4],
                                  [103.5, 100.1, 100.8, 4.2],
                                  [103.5, 100.1, 100.8, 3.8],
                                  [113.6, 113, 105.4, 5.1],
                                  [112.2, 110.1, 105.9, 4.8],
                                  [112, 111.9, 111.2, 4],
                                  [108.4, 110.1, 100.8, 4.2],
                                  [114.3, 110.3, 105.9, 4.2],
                                  [109.2, 111.1, 106.1, 3.8],
                                  [112.7, 105.7, 105.4, 3.9],
                                  [113.4, 107.7, 105.6, 3.9],
                                  [112.6, 107.6, 105.6, 3.6],
                                  [113.7, 111.5, 106, 3.6],
                                  [115.3, 111.5, 104.5, 3.8],
                                  [114.3, 110.9, 108.2, 3.9],
                                  [120.4, 114.1, 107.2, 3.9],
                                  [115, 112.3, 109.7, 3.9],
                                  [120.2, 117.6, 109.7, 5.1],
                                  [117.3, 115.8, 111.3, 4.1],
                                  [115, 112.5, 108.9, 3.8],
                                  [117.7, 112.5, 113.8, 3.8],
                                  [119.3, 115.6, 110.3, 3.9],
                                  [124.5, 115.6, 112.2, 3.9],
                                  [121.3, 118, 114.5, 3.8],
                                  [125.9, 120.1, 115.7, 3.7],
                                  [125.9, 120.9, 119.7, 3.7],
                                  [127.3, 123, 121.1, 3.6],
                                  [129.8, 123.6, 121.3, 3.6],
                                  [128.8, 122.4, 120.8, 3.7],
                                  [127, 123.4, 121.4, 3.6],
                                  [128.5, 118.8, 121.4, 3.6],
                                  [129.4, 124.4, 122.3, 3.5],
                                  [130.2, 123.2, 121.3, 3.5],
                                  [134.5, 128.9, 129.2, 3.4],
                                  [128.6, 121, 124.1, 3.4],
                                  [133.1, 126.8, 128.1, 2.9],
                                  [133.4, 123.5, 122, 4],
                                  [137.9, 125.3, 127.2, 4],
                                  [133.9, 124.1, 126.6, 3.5],
                                  [133.3, 129.6, 126.6, 2.9],
                                  [134.2, 131.5, 130.8, 3.9],
                                  [135.9, 135.3, 129.7, 3.9],
                                  [135.4, 133.5, 129.7, 3.9],
                                  [138.9, 134.4, 128.5, 3.4],
                                  [134.6, 133.6, 128.9, 3.7],
                                  [134.5, 129, 129, 3.7],
                                  [134.8, 132, 129.6, 3.2],
                                  [140.3, 135.5, 130.4, 4.1],
                                  [141.3, 132.8, 131.2, 3.8],
                                  [137.8, 130.3, 129, 3.8],
                                  [143.1, 135.8, 134, 5.2],
                                  [138, 129.8, 127.6, 4.2],
                                  [137.4, 132.1, 129.3, 4.2],
                                  [139.7, 132.6, 125.3, 3.8],
                                  [140.5, 137.4, 135, 3],
                                  [143, 140.3, 136.3, 4.1],
                                  [142.2, 137.9, 134.9, 3.6],
                                  [144.5, 136.9, 135, 3.6],
                                  [144.6, 144.8, 135.5, 3.7],
                                  [142.8, 138.9, 134.4, 3.7],
                                  [145.5, 146.9, 138.7, 3.7],
                                  [145.3, 141.8, 138.7, 4.2],
                                  [147.4, 142.4, 138.1, 4.5],
                                  [143.7, 143.4, 136.9, 3.5],
                                  [142, 144.7, 141.2, 4.1],
                                  [151.1, 149.6, 135.9, 3.8],
                                  [147.2, 138.7, 129.7, 3.8],
                                  [150.8, 144.8, 139.8, 3.8],
                                  [146.9, 141.5, 138.6, 3.7],
                                  [157.3, 153.7, 140.5, 3.7],
                                  [146.6, 141.2, 146.7, 3.3],
                                  [147.3, 142.6, 141.1, 3.8],
                                  [148.9, 142.1, 142.7, 3.8],
                                  [147.2, 139.6, 140.7, 3.5],
                                  [160.6, 157.9, 143.6, 3.5],
                                  [155.6, 149.2, 143, 3.6],
                                  [149.2, 146.5, 140.3, 4],
                                  [156.5, 159.6, 139.5, 4],
                                  [151.1, 146.3, 143.4, 3.7],
                                  [146.7, 142.1, 143.7, 3.7],
                                  [149.5, 143.4, 140.8, 3.9],
                                  [145.2, 138.3, 135.6, 3.6],
                                  [146.8, 138.2, 138.6, 3.6],
                                  [148.4, 140.4, 137.6, 3.5],
                                  [144.7, 140.8, 137.7, 3.5],
                                  [145.5, 142.1, 138.4, 3.9],
                                  [143.3, 143.1, 138.9, 3.5],
                                  [151.5, 143.6, 140.8, 3.5],
                                  [148, 145.9, 141.8, 3.3],
                                  [148.3, 146.7, 144.7, 3.3],
                                  [149.3, 143.8, 139.9, 3.9],
                                  [149.2, 144.9, 140.1, 3.4],
                                  [150, 149.4, 141.8, 3.4],
                                  [147, 143.3, 137.9, 3.6],
                                  [146.2, 144.9, 139, 3.6],
                                  [148.8, 146.1, 138.9, 3.7],
                                  [147.5, 143, 140.7, 3.7],
                                  [149.9, 143.5, 136.1, 3.7],
                                  [149.9, 144.3, 137.1, 3.4],
                                  [146.2, 144.2, 137.3, 3.4],
                                  [154, 150.9, 142.1, 3.3],
                                  [147, 143.3, 141.3, 3.3],
                                  [148.3, 149.3, 139.6, 3.8],
                                  [149.4, 146.8, 140.5, 3.5],
                                  [150.4, 148, 142.3, 3.5],
                                  [149.3, 144.9, 142.4, 3.7],
                                  [149.7, 146.6, 140.3, 3.4],
                                  [147.1, 146.6, 140.2, 4.3],
                                  [148.5, 146.6, 137.7, 3.7],
                                  [145.6, 142.9, 138.6, 3.7],
                                  [147, 145.4, 140.7, 2.9],
                                  [148.4, 145.4, 141.1, 2.8],
                                  [148.6, 144.8, 141.1, 3.7],
                                  [143.5, 139.8, 137.4, 3.4],
                                  [149, 145.2, 141.3, 3.4],
                                  [149.3, 150.8, 141.8, 3.9],
                                  [146.6, 146.4, 139.4, 3.9],
                                  [149.5, 146.4, 142.4, 3.8],
                                  [149.7, 142.8, 142.1, 3.5],
                                  [149.7, 146.6, 142.1, 3.8],
                                  [150, 151.6, 147.2, 3.2],
                                  [150.6, 147.1, 142.8, 4],
                                  [149.8, 151.9, 142, 3.6],
                                  [150.2, 149.1, 142.5, 3.6],
                                  [149.8, 147.7, 142.3, 3.9],
                                  [150.5, 150.2, 147.1, 3.9],
                                  [150.4, 147.2, 145.2, 4.1],
                                  [146.9, 146.6, 142.9, 4.1],
                                  [148.8, 141.5, 142.9, 3.9],
                                  [147, 146.6, 142.9, 3.9],
                                  [146.8, 147.2, 142.8, 3.6],
                                  [144.7, 142.4, 138.6, 3.5],
                                  [152.4, 152.2, 143.1, 3.5],
                                  [153.2, 152.4, 148.2, 4.5],
                                  [148.1, 147.4, 143.1, 4.5],
                                  [148.3, 147.5, 142.7, 3.8],
                                  [148.1, 147.7, 142.7, 3.8],
                                  [148.4, 147.2, 142.9, 3.7],
                                  [155.2, 153, 148.2, 3.8],
                                  [148.4, 147.7, 143.3, 3.8],
                                  [148.7, 148.6, 143.9, 4.1],
                                  [159.1, 152.7, 154.2, 4.2],
                                  [159.5, 159.3, 154.5, 5.2],
                                  [161.6, 159.6, 155.1, 5],
                                  [159.2, 159.5, 153.8, 4.2],
                                  [159.2, 159.3, 149, 4.3],
                                  [158.7, 154.2, 149.3, 4.3],
                                  [164.4, 159.3, 154.6, 4.5],
                                  [159.5, 154.2, 149, 4.5],
                                  [164.7, 154.2, 154.5, 4.5],
                                  [159.3, 154.3, 149.4, 5],
                                  [159.4, 154.3, 149.4, 5],
                                  [164.7, 159.2, 154.8, 4.5],
                                  [159.7, 155.8, 152.2, 4.2],
                                  [164.9, 159, 154.7, 4.2],
                                  [165, 159.1, 155, 5.6],
                                  [159.5, 153.9, 149.9, 5.6],
                                  [164.6, 155, 155.3, 5.3],
                                  [159.3, 159.4, 155.3, 4.6],
                                  [159.5, 155.8, 155.2, 5.5],
                                  [159.1, 160.5, 157, 5.3],
                                  [164.5, 160.5, 159.9, 4.5],
                                  [163.4, 160.6, 149.7, 5.1],
                                  [164.4, 160, 155, 5.6],
                                  [165.1, 160, 155, 4.5],
                                  [160.1, 160.3, 154.7, 5],
                                  [159.7, 160.4, 149.4, 5],
                                  [164.7, 160.3, 154.6, 4.3],
                                  [166.6, 160.9, 158.8, 4.6],
                                  [164.7, 155.2, 159.9, 4.6],
                                  [159.7, 155.6, 149.6, 5.5],
                                  [159.3, 155.4, 154.7, 4.5],
                                  [160.2, 154.3, 149.6, 5.6],
                                  [158.8, 155.5, 148.2, 5.1],
                                  [155.2, 155.5, 144.3, 5.1],
                                  [158.9, 155.6, 155.1, 5.3],
                                  [158.9, 155.6, 150, 5.3],
                                  [155.5, 155, 150.1, 5.6],
                                  [159.3, 155.7, 147.1, 5.5],
                                  [155.8, 149.6, 147.1, 5.5],
                                  [150.7, 151.3, 143.9, 5],
                                  [150.8, 144.5, 139.5, 5.5],
                                  [155.1, 149.6, 144.7, 6.7],
                                  [153.6, 147.7, 143.1, 5.4],
                                  [150.2, 149.8, 139.7, 5.4],
                                  [149.6, 149.7, 140.8, 5.1],
                                  [149.6, 149.7, 140.8, 5.8],
                                  [155.3, 150.1, 143.4, 5.8],
                                  [148.1, 145, 140.1, 5.5],
                                  [145.7, 145.4, 140.9, 4.8],
                                  [145.8, 140.2, 135.8, 5.3],
                                  [145.8, 140.2, 135.8, 4.9],
                                  [147.9, 134.9, 134, 5.2],
                                  [144.2, 138.2, 134.9, 5.3],
                                  [144.2, 138.2, 134.9, 5.3],
                                  [145.7, 140.8, 135.5, 4.9],
                                  [145.7, 140.9, 135.5, 4.9],
                                  [147.6, 145.9, 136.7, 6],
                                  [144, 140.1, 133.6, 4.9],
                                  [144, 140.6, 133.6, 4.9],
                                  [147, 141.7, 135.5, 6],
                                  [147, 141.7, 130.4, 5],
                                  [145.7, 140.8, 134.9, 4.9],
                                  [139.6, 135.5, 128.2, 4.9],
                                  [145.7, 135.5, 134.9, 4.9],
                                  [140.7, 134.3, 131, 5],
                                  [140.7, 134.3, 131, 5],
                                  [135.7, 129.5, 124.8, 5.1],
                                  [136.7, 132.7, 126, 5.7],
                                  [136.7, 135.6, 126, 5.7],
                                  [136.5, 130.3, 125.4, 5.3],
                                  [135.9, 130.3, 125.4, 6],
                                  [135.8, 125.2, 125.9, 5],
                                  [135.2, 130, 128.3, 5.8],
                                  [135.2, 130, 128.3, 5],
                                  [134.1, 129.4, 126, 5.3],
                                  [131.1, 125.1, 119.6, 5.3]], columns=['IA', 'IB', 'IV', 'IN'])

    start_date_dt = datetime(int(start_date.split("-")[0]), int(start_date.split("-")[1]),
                             int(start_date.split("-")[2]))
    end_date_dt = datetime(int(final_date.split("-")[0]), int(final_date.split("-")[1]), int(final_date.split("-")[2]))

    dummy = numpy.arange(start_date_dt, end_date_dt, numpy.timedelta64(5, 'm'), dtype='datetime64')
    dummy = pandas.DataFrame(dummy, columns=['timestamp'])
    dummy.set_index('timestamp', inplace=True)

    aux_day = pandas.concat([dummy_day] * int(dummy.shape[0] / dummy_day.shape[0]), ignore_index=True)

    cycles = 0.7 * dummy.shape[0] / (365 * 24 * 12)  # how many sine cycles
    resolution = aux_day.shape[0]  # how many datapoints to generate
    length = numpy.pi * 2 * cycles
    season_year = numpy.sin(numpy.arange(0, length, length / resolution))

    cycles = 12 * 4 * dummy.shape[0] / (365 * 24 * 12)  # how many sine cycles
    resolution = aux_day.shape[0]  # how many datapoints to generate
    length = numpy.pi * 2 * cycles
    season_week = numpy.sin(numpy.arange(0, length, length / resolution))

    cycles = 12 * dummy.shape[0] / (365 * 24 * 12)  # how many sine cycles
    resolution = aux_day.shape[0]  # how many datapoints to generate
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

    dummy['IA'] = aux_day['IA'].values + rand_year * season_year + rand_month * season_month \
                                       + rand_week * season_week + rand_vet + step_vet
    dummy['IB'] = aux_day['IB'].values + rand_year * season_year + rand_month * season_month \
                                       + rand_week * season_week + rand_vet + step_vet
    dummy['IV'] = aux_day['IV'].values + rand_year * season_year + rand_month * season_month \
                                       + rand_week * season_week + rand_vet + step_vet
    dummy['IN'] = aux_day['IN'].values + (rand_year / 10) * season_year \
                                       + (rand_month / 10) * season_month + (rand_week / 10) \
                                                           * season_week + rand_vet / 10

    return dummy


def VoltageDummyData(start_date: str = '2021-01-01', final_date: str = '2023-01-01'):
    start_date_dt = datetime(int(start_date.split("-")[0]), int(start_date.split("-")[1]),
                             int(start_date.split("-")[2]))
    end_date_dt = datetime(int(final_date.split("-")[0]), int(final_date.split("-")[1]), int(final_date.split("-")[2]))

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

        step_vet[start:end] = random.randint(-300, 300)

    dummy['VA'] = 1.03 * 13.8 + rand_vet + step_vet
    dummy['VB'] = 1.03 * 13.8 + rand_vet + step_vet
    dummy['VV'] = 1.03 * 13.8 + rand_vet + step_vet

    return dummy


def PowerFactorDummyData(start_date: str = '2021-01-01', final_date: str = '2023-01-01'):
    dummy_day = pandas.DataFrame([[0.85, 0.88, 0.9],
                                  [0.86, 0.88, 0.9],
                                  [0.84, 0.88, 0.89],
                                  [0.86, 0.88, 0.9],
                                  [0.86, 0.88, 0.9],
                                  [0.87, 0.9, 0.91],
                                  [0.87, 0.9, 0.9],
                                  [0.87, 0.91, 0.91],
                                  [0.86, 0.9, 0.91],
                                  [0.86, 0.9, 0.92],
                                  [0.87, 0.9, 0.92],
                                  [0.87, 0.9, 0.92],
                                  [0.88, 0.91, 0.92],
                                  [0.88, 0.92, 0.92],
                                  [0.88, 0.92, 0.92],
                                  [0.89, 0.92, 0.93],
                                  [0.89, 0.93, 0.95],
                                  [0.89, 0.92, 0.95],
                                  [0.91, 0.92, 0.95],
                                  [0.9, 0.92, 0.95],
                                  [0.9, 0.93, 0.94],
                                  [0.91, 0.93, 0.94],
                                  [0.9, 0.94, 0.94],
                                  [0.9, 0.94, 0.94],
                                  [0.91, 0.94, 0.95],
                                  [0.91, 0.94, 0.94],
                                  [0.91, 0.94, 0.94],
                                  [0.91, 0.94, 0.95],
                                  [0.91, 0.94, 0.95],
                                  [0.91, 0.94, 0.95],
                                  [0.91, 0.96, 0.95],
                                  [0.91, 0.96, 0.95],
                                  [0.92, 0.95, 0.96],
                                  [0.92, 0.95, 0.96],
                                  [0.91, 0.96, 0.96],
                                  [0.92, 0.95, 0.96],
                                  [0.91, 0.95, 0.96],
                                  [0.92, 0.95, 0.95],
                                  [0.92, 0.95, 0.95],
                                  [0.92, 0.95, 0.96],
                                  [0.93, 0.95, 0.96],
                                  [0.93, 0.95, 0.98],
                                  [0.93, 0.96, 0.97],
                                  [0.93, 0.96, 0.97],
                                  [0.94, 0.96, 0.97],
                                  [0.93, 0.96, 0.97],
                                  [0.93, 0.96, 0.97],
                                  [0.93, 0.96, 0.98],
                                  [0.93, 0.96, 0.98],
                                  [0.93, 0.96, 0.97],
                                  [0.93, 0.95, 0.97],
                                  [0.93, 0.95, 0.97],
                                  [0.92, 0.95, 0.97],
                                  [0.92, 0.95, 0.97],
                                  [0.94, 0.96, 0.98],
                                  [0.93, 0.96, 0.97],
                                  [0.93, 0.96, 0.97],
                                  [0.93, 0.96, 0.95],
                                  [0.93, 0.96, 0.98],
                                  [0.94, 0.95, 0.98],
                                  [0.94, 0.96, 0.98],
                                  [0.96, 0.96, 0.98],
                                  [0.93, 0.96, 0.98],
                                  [0.93, 0.96, 0.98],
                                  [0.94, 0.98, 0.98],
                                  [0.94, 0.96, 0.98],
                                  [0.94, 0.98, 0.98],
                                  [0.95, 0.98, 1],
                                  [0.95, 0.98, 0.98],
                                  [0.96, 0.97, 1],
                                  [0.95, 0.97, 0.99],
                                  [0.95, 0.97, 0.97],
                                  [0.97, 0.99, 0.99],
                                  [0.97, 0.99, 0.99],
                                  [0.92, 0.93, 0.96],
                                  [0.93, 0.94, 0.96],
                                  [0.93, 0.93, 0.93],
                                  [0.95, 0.94, 0.99],
                                  [0.92, 0.94, 0.96],
                                  [0.94, 0.94, 0.96],
                                  [0.93, 0.96, 0.96],
                                  [0.92, 0.95, 0.96],
                                  [0.93, 0.95, 0.96],
                                  [0.92, 0.93, 0.96],
                                  [0.91, 0.93, 0.97],
                                  [0.92, 0.94, 0.95],
                                  [0.89, 0.92, 0.95],
                                  [0.92, 0.93, 0.94],
                                  [0.89, 0.9, 0.94],
                                  [0.91, 0.91, 0.93],
                                  [0.92, 0.93, 0.95],
                                  [0.9, 0.93, 0.92],
                                  [0.9, 0.91, 0.94],
                                  [0.87, 0.91, 0.93],
                                  [0.89, 0.9, 0.92],
                                  [0.86, 0.89, 0.91],
                                  [0.86, 0.89, 0.89],
                                  [0.86, 0.88, 0.89],
                                  [0.84, 0.87, 0.89],
                                  [0.85, 0.88, 0.89],
                                  [0.86, 0.88, 0.89],
                                  [0.85, 0.9, 0.89],
                                  [0.85, 0.87, 0.88],
                                  [0.84, 0.88, 0.89],
                                  [0.82, 0.85, 0.85],
                                  [0.85, 0.89, 0.87],
                                  [0.83, 0.86, 0.85],
                                  [0.83, 0.88, 0.88],
                                  [0.81, 0.87, 0.86],
                                  [0.82, 0.87, 0.86],
                                  [0.83, 0.85, 0.86],
                                  [0.82, 0.84, 0.84],
                                  [0.82, 0.82, 0.85],
                                  [0.82, 0.83, 0.85],
                                  [0.8, 0.82, 0.85],
                                  [0.82, 0.83, 0.85],
                                  [0.82, 0.85, 0.85],
                                  [0.82, 0.83, 0.85],
                                  [0.79, 0.82, 0.84],
                                  [0.79, 0.83, 0.84],
                                  [0.81, 0.84, 0.85],
                                  [0.78, 0.82, 0.82],
                                  [0.81, 0.84, 0.86],
                                  [0.81, 0.83, 0.85],
                                  [0.8, 0.83, 0.87],
                                  [0.79, 0.81, 0.82],
                                  [0.78, 0.79, 0.81],
                                  [0.78, 0.81, 0.82],
                                  [0.77, 0.81, 0.82],
                                  [0.77, 0.77, 0.82],
                                  [0.78, 0.8, 0.82],
                                  [0.77, 0.76, 0.8],
                                  [0.77, 0.79, 0.8],
                                  [0.76, 0.78, 0.8],
                                  [0.78, 0.78, 0.81],
                                  [0.79, 0.77, 0.79],
                                  [0.74, 0.75, 0.82],
                                  [0.76, 0.8, 0.85],
                                  [0.74, 0.77, 0.8],
                                  [0.76, 0.79, 0.8],
                                  [0.71, 0.73, 0.79],
                                  [0.76, 0.79, 0.76],
                                  [0.76, 0.78, 0.79],
                                  [0.75, 0.79, 0.78],
                                  [0.76, 0.8, 0.79],
                                  [0.7, 0.71, 0.78],
                                  [0.72, 0.75, 0.78],
                                  [0.75, 0.76, 0.79],
                                  [0.72, 0.7, 0.8],
                                  [0.74, 0.76, 0.78],
                                  [0.76, 0.79, 0.78],
                                  [0.75, 0.78, 0.79],
                                  [0.77, 0.8, 0.82],
                                  [0.76, 0.8, 0.8],
                                  [0.75, 0.79, 0.81],
                                  [0.77, 0.79, 0.81],
                                  [0.77, 0.79, 0.8],
                                  [0.78, 0.78, 0.8],
                                  [0.74, 0.78, 0.79],
                                  [0.76, 0.77, 0.79],
                                  [0.76, 0.76, 0.77],
                                  [0.75, 0.78, 0.8],
                                  [0.75, 0.77, 0.79],
                                  [0.75, 0.75, 0.79],
                                  [0.76, 0.78, 0.81],
                                  [0.77, 0.77, 0.8],
                                  [0.75, 0.77, 0.8],
                                  [0.76, 0.78, 0.79],
                                  [0.75, 0.78, 0.81],
                                  [0.75, 0.77, 0.81],
                                  [0.77, 0.78, 0.81],
                                  [0.73, 0.74, 0.79],
                                  [0.76, 0.78, 0.79],
                                  [0.76, 0.75, 0.8],
                                  [0.75, 0.76, 0.79],
                                  [0.75, 0.76, 0.78],
                                  [0.75, 0.77, 0.78],
                                  [0.75, 0.76, 0.79],
                                  [0.76, 0.76, 0.79],
                                  [0.75, 0.76, 0.81],
                                  [0.77, 0.78, 0.8],
                                  [0.76, 0.77, 0.79],
                                  [0.75, 0.77, 0.79],
                                  [0.75, 0.77, 0.79],
                                  [0.78, 0.8, 0.81],
                                  [0.75, 0.77, 0.79],
                                  [0.75, 0.74, 0.79],
                                  [0.76, 0.76, 0.8],
                                  [0.75, 0.76, 0.78],
                                  [0.75, 0.78, 0.79],
                                  [0.75, 0.76, 0.79],
                                  [0.75, 0.74, 0.76],
                                  [0.74, 0.76, 0.78],
                                  [0.75, 0.74, 0.79],
                                  [0.75, 0.75, 0.78],
                                  [0.75, 0.76, 0.78],
                                  [0.74, 0.75, 0.76],
                                  [0.75, 0.76, 0.77],
                                  [0.76, 0.76, 0.78],
                                  [0.75, 0.79, 0.78],
                                  [0.76, 0.76, 0.78],
                                  [0.76, 0.76, 0.78],
                                  [0.77, 0.78, 0.8],
                                  [0.74, 0.74, 0.78],
                                  [0.73, 0.74, 0.76],
                                  [0.76, 0.76, 0.78],
                                  [0.76, 0.76, 0.78],
                                  [0.76, 0.76, 0.78],
                                  [0.75, 0.76, 0.78],
                                  [0.72, 0.73, 0.76],
                                  [0.75, 0.76, 0.78],
                                  [0.75, 0.75, 0.78],
                                  [0.7, 0.73, 0.73],
                                  [0.7, 0.7, 0.73],
                                  [0.69, 0.7, 0.72],
                                  [0.7, 0.7, 0.73],
                                  [0.7, 0.7, 0.75],
                                  [0.7, 0.73, 0.75],
                                  [0.68, 0.7, 0.72],
                                  [0.7, 0.73, 0.75],
                                  [0.68, 0.73, 0.73],
                                  [0.7, 0.73, 0.75],
                                  [0.7, 0.73, 0.75],
                                  [0.68, 0.7, 0.72],
                                  [0.7, 0.72, 0.74],
                                  [0.67, 0.7, 0.72],
                                  [0.67, 0.7, 0.72],
                                  [0.7, 0.73, 0.75],
                                  [0.68, 0.72, 0.72],
                                  [0.7, 0.7, 0.72],
                                  [0.7, 0.72, 0.72],
                                  [0.7, 0.7, 0.71],
                                  [0.68, 0.7, 0.7],
                                  [0.68, 0.7, 0.75],
                                  [0.68, 0.7, 0.72],
                                  [0.67, 0.7, 0.72],
                                  [0.7, 0.7, 0.72],
                                  [0.7, 0.7, 0.75],
                                  [0.68, 0.7, 0.72],
                                  [0.67, 0.69, 0.7],
                                  [0.68, 0.72, 0.7],
                                  [0.7, 0.72, 0.75],
                                  [0.7, 0.72, 0.72],
                                  [0.7, 0.73, 0.75],
                                  [0.7, 0.72, 0.76],
                                  [0.72, 0.72, 0.77],
                                  [0.7, 0.72, 0.72],
                                  [0.7, 0.72, 0.75],
                                  [0.72, 0.72, 0.75],
                                  [0.7, 0.72, 0.76],
                                  [0.72, 0.75, 0.76],
                                  [0.74, 0.74, 0.78],
                                  [0.74, 0.77, 0.8],
                                  [0.72, 0.75, 0.77],
                                  [0.73, 0.76, 0.78],
                                  [0.75, 0.75, 0.8],
                                  [0.75, 0.75, 0.79],
                                  [0.75, 0.75, 0.79],
                                  [0.72, 0.75, 0.78],
                                  [0.76, 0.77, 0.79],
                                  [0.77, 0.77, 0.79],
                                  [0.77, 0.79, 0.82],
                                  [0.77, 0.79, 0.82],
                                  [0.76, 0.82, 0.82],
                                  [0.78, 0.8, 0.82],
                                  [0.78, 0.8, 0.82],
                                  [0.77, 0.79, 0.82],
                                  [0.77, 0.79, 0.82],
                                  [0.76, 0.77, 0.81],
                                  [0.78, 0.79, 0.83],
                                  [0.78, 0.79, 0.83],
                                  [0.76, 0.79, 0.82],
                                  [0.76, 0.79, 0.84],
                                  [0.77, 0.79, 0.82],
                                  [0.8, 0.82, 0.85],
                                  [0.77, 0.82, 0.82],
                                  [0.79, 0.82, 0.84],
                                  [0.79, 0.82, 0.84],
                                  [0.82, 0.85, 0.87],
                                  [0.81, 0.83, 0.86],
                                  [0.81, 0.82, 0.86],
                                  [0.81, 0.84, 0.87],
                                  [0.82, 0.84, 0.87],
                                  [0.82, 0.87, 0.86],
                                  [0.82, 0.84, 0.85],
                                  [0.82, 0.84, 0.85],
                                  [0.82, 0.85, 0.86],
                                  [0.84, 0.87, 0.89]], columns=['FPA', 'FPB', 'FPV'])

    start_date_dt = datetime(int(start_date.split("-")[0]), int(start_date.split("-")[1]),
                             int(start_date.split("-")[2]))
    end_date_dt = datetime(int(final_date.split("-")[0]), int(final_date.split("-")[1]), int(final_date.split("-")[2]))

    dummy = numpy.arange(start_date_dt, end_date_dt, numpy.timedelta64(5, 'm'), dtype='datetime64')
    dummy = pandas.DataFrame(dummy, columns=['timestamp'])
    dummy.set_index('timestamp', inplace=True)

    aux_day = pandas.concat([dummy_day] * int(dummy.shape[0] / dummy_day.shape[0]), ignore_index=True)

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

        step_vet[start:end] = random.randint(-300, 300)

    dummy['FPA'] = aux_day['FPA'].values + step_vet
    dummy['FPB'] = aux_day['FPB'].values + step_vet
    dummy['FPV'] = aux_day['FPV'].values + step_vet

    return dummy


def PowerDummyData(start_date: str = '2021-01-01', final_date: str = '2023-01-01'):
    I = CurrentDummyData(start_date, final_date)
    V = VoltageDummyData(start_date, final_date)
    pf = PowerFactorDummyData(start_date, final_date)

    I = I.iloc[:, :-1]

    dummy = pandas.DataFrame([])

    dummy['S'] = V['VA'] / numpy.sqrt(3) * I['IA'] + V['VB'] / numpy.sqrt(3) * I['IB'] \
                                                   + V['VV'] / numpy.sqrt(3) * I['IV']
    dummy['P'] = V['VA'] / numpy.sqrt(3) * I['IA'] * pf['FPA'] + V['VB'] / numpy.sqrt(3) * I['IB'] * pf['FPB'] \
                                                               + V['VV'] / numpy.sqrt(3) * I['IV'] * pf['FPV']
    dummy['Q'] = dummy['S'].pow(2) - dummy['P'].pow(2)
    dummy['Q'] = numpy.sqrt(dummy['Q'])

    return dummy


def EnergyDummyData(start_date: str = '2021-01-01', final_date: str = '2023-01-01'):
    dummy_s = PowerDummyData(start_date, final_date)

    dummy = pandas.DataFrame([])

    dummy['Eactive'] = dummy_s['P'].cumsum(skipna=True)

    dummy['Ereactive'] = dummy_s['Q'].abs().cumsum(skipna=True)

    return dummy


def ShowExampleSimpleProcess():
    
    data_inicio = '2021-01-01'
    data_final = '2023-01-01'

    start_date_dt = datetime(int(data_inicio.split("-")[0]), int(data_inicio.split("-")[1]),
                             int(data_inicio.split("-")[2]))

    end_date_dt = datetime(int(data_final.split("-")[0]), int(data_final.split("-")[1]),
                           int(data_final.split("-")[2]))

    dummy = CurrentDummyData()
    dummy.plot(title="Current Input (with outliers [A]")

    time_stopper = [['time_init', time.perf_counter()]]
    output = DataSynchronization(dummy, start_date_dt, end_date_dt, sample_freq=5, sample_time_base='m')
    
    fig, ax = matplotlib.pyplot.subplots()
    ax.plot(output.values)
    ax.set_title('Input')

    CountMissingData(output, show=True)
    time_stopper.append(['DataSynchronization', time.perf_counter()])
    output = RemoveOutliersHardThreshold(output, hard_max=500, hard_min=0)

    fig, ax = matplotlib.pyplot.subplots()
    ax.plot(output.values)
    ax.set_title('Whitout Outliers (RemoveOutliersHardThreshold)')

    CountMissingData(output, show=True)
    time_stopper.append(['RemoveOutliersHardThreshold', time.perf_counter()])
    output = RemoveOutliersMMADMM(output, len_mov_avg=3, std_def=4, plot=False, remove_from_process=['IN'])

    fig, ax = matplotlib.pyplot.subplots()
    ax.plot(output.values)
    ax.set_title('Whitout Outliers (+RemoveOutliersMMADMM)')

    CountMissingData(output, show=True)
    time_stopper.append(['RemoveOutliersMMADMM', time.perf_counter()])
    output = RemoveOutliersQuantile(output)

    fig, ax = matplotlib.pyplot.subplots()
    ax.plot(output.values)
    ax.set_title('Whitout Outliers (+RemoveOutliersQuantile)')

    CountMissingData(output, show=True)
    time_stopper.append(['RemoveOutliersQuantile', time.perf_counter()])
    output = RemoveOutliersHistoGram(output, min_number_of_samples_limit=12 * 5)

    fig, ax = matplotlib.pyplot.subplots()
    ax.plot(output.values)
    ax.set_title('Whitout Outliers (+RemoveOutliersHistoGram)')

    CountMissingData(output, show=True)
    time_stopper.append(['RemoveOutliersHistoGram', time.perf_counter()])

    

    output = SimpleProcess(output, start_date_dt, end_date_dt,
                           remove_from_process=['IN'],
                           sample_freq=5,
                           sample_time_base='m',
                           pre_interpol=1,
                           pos_interpol=6,
                           prop_phases=True,
                           integrate=True,
                           interpol_integrate=100)

    time_stopper.append(['SimpleProcessInput', time.perf_counter()])
    CountMissingData(output, show=True)

    fig, ax = matplotlib.pyplot.subplots()
    ax.plot(output.values)
    ax.set_title('Output (SimpleProcess)')
    matplotlib.pyplot.show()
    
    TimeProfile(time_stopper, name='Main', show=True, estimate_for=1000 * 5)

    return

