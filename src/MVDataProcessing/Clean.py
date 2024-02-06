import pandas
import numpy
import matplotlib.pyplot
from .Util import SavePeriod,IntegrateHour


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

    #X_moving_median.iloc[-2 * len_mov_avg:, :] = X_moving_median.iloc[-2 * len_mov_avg:, :].fillna(method='ffill')
    #X_moving_median.iloc[:2 * len_mov_avg, :] = X_moving_median.iloc[:2 * len_mov_avg, :].fillna(method='bfill')
    
    X_moving_median.iloc[-2 * len_mov_avg:, :] = X_moving_median.iloc[-2 * len_mov_avg:, :].ffill()
    X_moving_median.iloc[:2 * len_mov_avg, :] = X_moving_median.iloc[:2 * len_mov_avg, :].bfill() 
    
    # ------------ Computa MAD Móvel ------------#
    X_mad = X - X_moving_median
    X_mad = X_mad.rolling(len_mov_avg).median().shift(-int(len_mov_avg / 2))
    #X_mad.iloc[-2 * len_mov_avg:, :] = X_mad.iloc[-2 * len_mov_avg:, :].fillna(method='ffill')
    #X_mad.iloc[:2 * len_mov_avg, :] = X_mad.iloc[:2 * len_mov_avg, :].fillna(method='bfill')

    X_mad.iloc[-2 * len_mov_avg:, :] = X_mad.iloc[-2 * len_mov_avg:, :].ffill()
    X_mad.iloc[:2 * len_mov_avg, :] = X_mad.iloc[:2 * len_mov_avg, :].bfill() 
    
    
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
        Y_int.loc[:,col] = Y_int.loc[:,col].sort_values(ascending=False, ignore_index=True)

    if Y_int.shape[0] < min_number_of_samples_limit:
        min_number_of_samples_limit = Y_int.shape[0]

    threshold_max = Y_int.iloc[min_number_of_samples_limit + 1, :]
    threshold_min = Y_int.iloc[-min_number_of_samples_limit - 1, :]
    
    threshold_max = threshold_max.fillna(9999999999)
    threshold_min = threshold_min.fillna(0)

    for col in Y:
        Y.loc[numpy.logical_or(Y[col] > threshold_max[col], Y[col] < threshold_min[col]), col] = numpy.nan

    if df_avoid_periods.shape[0] != 0:
        df_values, index_return = SavePeriod(X, df_avoid_periods)
        Y.loc[index_return, :] = df_values

    # return the keep out columns
    if len(remove_from_process) > 0:
        Y = pandas.concat([Y, x_in.loc[:, remove_from_process]], axis=1)

    return Y
