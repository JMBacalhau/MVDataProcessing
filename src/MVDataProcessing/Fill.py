import pandas
import time
import numpy
from itertools import permutations
import matplotlib.pyplot
import datetime

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
