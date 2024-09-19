#  Copyright (c) 2022. Terrier Team at University of Glasgow, http://http://terrierteam.dcs.gla.ac.uk
#
#  This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
#  If a copy of the MPL was not distributed with this  file, you can obtain one at
#  http://mozilla.org/MPL/2.0/.
#
#  This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
#  If a copy of the MPL was not distributed with this  file, you can obtain one at
#  http://mozilla.org/MPL/2.0/.

import datetime as dt
import os
import statistics
import sys

import numpy as np
import pandas as pd
from beta_rec.utils.constants import DEFAULT_RATING_COL, DEFAULT_ITEM_COL, DEFAULT_TIMESTAMP_COL, DEFAULT_USER_COL

from algorithms.kpi_gen.profit_vol_ma_kpi_generator import ProfitabilityVolatilityMAKPIGenerator
from data.financial_asset_time_series import FinancialAssetTimeSeries
from data.financial_data_continuous import FinancialContinuousData
from data.financial_interaction_data import FinancialInteractionData

import argparse

pd.options.mode.chained_assignment = None  # default='warn'

if __name__ == "__main__":

    timea = dt.datetime.now()
    parser = argparse.ArgumentParser(
        prog="dataset_analysis",
        description="Analyzes the dataset from a general asset point of view.",
        epilog="Developed by University of Glasgow"
    )

    parser.add_argument("interactions", help="Customer-asset transaction data file.")
    parser.add_argument("time_series", help="Asset pricing data file.")
    subparsers = parser.add_subparsers(title='date_format', help='Data choice format.', dest='date_format')

    parser_range = subparsers.add_parser('range', help='Range of dates to use. This mode divides the dataset as '
                                                       'follows:\n'
                                                       '- First, divide the period between min_date and max_date into'
                                                       'num_splits + num_future dates\n'
                                                       '- Second, the first num_split dates are considered the split '
                                                       'dates (everything before them is the training set).\n'
                                                       '- The test set contains the data between the split date and '
                                                       '  num_future dates in the list afterwards.')
    parser_range.add_argument("min_date", help='Date of the first split. Format: %Y-%m-%d')
    parser_range.add_argument("max_date", help='End date of the last test set. Format: %Y-%m-%d')
    parser_range.add_argument("num_splits", help='Number of splits to consider.', type=int)
    parser_range.add_argument("num_future", help='Number of dates to look formward', type=int)

    parser_fixed = subparsers.add_parser('fixed_dates', help='List of fixed dates to use. This mode provides fixed '
                                                             'lists of dates for split and test.')
    parser_fixed.add_argument('split_dates', help='Comma separated list of split dates. Date format: %Y-%m-%d')
    parser_fixed.add_argument('future_dates', help='Comma separated list of test end dates. Date format: %Y-%m-%d')
    parser.add_argument("output_dir", help="directory on which to store the outputs.")
    parser.add_argument("summary_file", help="name of the summary file.")

    args = parser.parse_args()

    # First, we read the parameters:
    interaction_data_file = args.interactions
    time_series_data_file = args.time_series
    date_format = args.date_format

    dates_args = []
    future_dates_args = []
    num_splits = 0
    num_future = 0
    min_date = ""
    max_date = ""

    if date_format == "range":
        min_date = dt.datetime.strptime(args.min_date, "%Y-%m-%d")
        max_date = dt.datetime.strptime(args.max_date, "%Y-%m-%d")
        num_splits = args.num_splits
        num_future = args.num_future
    elif date_format == "fixed_dates":
        dates_args = args.split_dates.split(",")
        future_dates_args = args.future_dates.split(",")
        num_splits = len(dates_args)
        num_future = len(future_dates_args)
        min_date = min(dates_args)
        max_date = max(future_dates_args)
    else:
        exit(-1)

    directory = args.output_dir
    end_file = args.summary_file

    # Now, we load the data:
    interaction_data = FinancialInteractionData(interaction_data_file, repeated=False)
    time_series_data = FinancialAssetTimeSeries(time_series_data_file)
    data = FinancialContinuousData(interaction_data, time_series_data)
    data.load()

    timeb = dt.datetime.now() - timea
    print("Dataset loaded (" + '{}'.format(timeb) + ")")

    # Generate the technical indicators:

    kpi_generator = ProfitabilityVolatilityMAKPIGenerator(data.time_series.data, 5)
    if not os.path.exists(os.path.join(directory, "kpis.csv")):
        kpi_generator.compute()
        data.add_kpis(kpi_generator.get_kpis())
        kpi_generator.get_kpis().to_csv(os.path.join(directory, "kpis.csv"))
    else:
        kpis = pd.read_csv(os.path.join(directory, "kpis.csv"))
        kpis[DEFAULT_TIMESTAMP_COL] = pd.to_datetime(kpis[DEFAULT_TIMESTAMP_COL])
        data.add_kpis(kpis)

    timeb = dt.datetime.now() - timea
    print("Technical indicators computed (" + '{}'.format(timeb) + ")")

    # Now, we select the possible dates:
    dates = []
    future_dates = []
    # Now, we select the possible dates:
    if date_format == "range":
        print("Num splits:" + str(num_splits) + " Num future: " + str(num_future))
        dates, future_dates = data.get_dates(min_date, max_date, num_splits, num_future)
    else:
        print("Num splits:" + str(num_splits))
        for date in dates_args:
            dates.append(pd.to_datetime(date))
        for date in future_dates_args:
            future_dates.append(pd.to_datetime(date))

    print("Selected dates:")
    for i in range(0, len(dates)):
        print("\t" + str(i) + "Training date: " + str(dates[i]) + "\tFuture date: " + str(future_dates[i]))

    values = []
    for i in range(0, len(dates)):
        print("Starting " + str(dates[i]))

        rec_date = dates[i]
        min_split_date = rec_date - dt.timedelta(days=36500)
        future_date = future_dates[i]
        date_values = dict()
        date_values["timestamp"] = rec_date

        splitted = data.split(min_split_date, rec_date, future_date)

        time_series = data.time_series.data
        current_val = time_series[time_series[DEFAULT_TIMESTAMP_COL] == rec_date][[DEFAULT_ITEM_COL, DEFAULT_RATING_COL]]
        future_val = time_series[time_series[DEFAULT_TIMESTAMP_COL] == future_date][[DEFAULT_ITEM_COL, DEFAULT_RATING_COL]]
        profit_df = pd.merge(current_val, future_val, on=DEFAULT_ITEM_COL, suffixes=("_current", "_future"))
        profit_df["ROI"] = (profit_df[DEFAULT_RATING_COL + "_future"] - profit_df[DEFAULT_RATING_COL + "_current"]) / profit_df[DEFAULT_RATING_COL + "_current"]
        profit_df["AnnROI"] = profit_df["ROI"].apply(lambda x: pow(1 + x, 365 / (future_date - rec_date).days) - 1)
        profit_df["MonthROI"] = profit_df["ROI"].apply(lambda x: pow(1 + x, 30 / (future_date - rec_date).days) - 1)

        tuples = []
        for asset in time_series[DEFAULT_ITEM_COL].unique().flatten():
            aux_series = time_series[time_series[DEFAULT_ITEM_COL] == asset]
            aux_series["profit"] = (aux_series[DEFAULT_RATING_COL] - aux_series[DEFAULT_RATING_COL].shift(1)) / \
                                   aux_series[DEFAULT_RATING_COL].shift(1)
            aux_series = aux_series.dropna()

            tuples.append((asset,aux_series["profit"].std() * np.sqrt(252)))

        vol_df = pd.DataFrame(tuples, columns=[DEFAULT_ITEM_COL, "Volatility"])
        profit_df = profit_df.merge(vol_df, on=DEFAULT_ITEM_COL)

        for key in ["ROI", "AnnROI", "MonthROI", "Volatility"]:
            cleanedList = profit_df[key].to_list()
            mean_val = statistics.mean(cleanedList)
            quantiles = statistics.quantiles(cleanedList)
            percentile = 1.0 - len([x for x in cleanedList if x > 0.0]) / (len(cleanedList) + 0.0)

            date_values["overall_mean_"+key] = mean_val
            date_values["overall_q1_"+key] = quantiles[0]
            date_values["overall_median_"+key] = quantiles[1]
            date_values["overall_q3_"+key] = quantiles[2]
            date_values["overall_perc_non_profitable_"+key] = percentile

        date_values["index_month_roi"] = pow(1 + date_values["overall_mean_ROI"],
                                             30/(future_date - rec_date).days) - 1.0
        date_values["index_annual_roi"] = pow(1 + date_values["overall_mean_ROI"],
                                              365 / (future_date - rec_date).days) - 1.0

        profit_df.to_csv(os.path.join(directory, "assets_stats_" + str(rec_date).split()[0] + ".csv"), index=False)
        date_values["num_assets"] = len(profit_df[DEFAULT_ITEM_COL].unique())
        date_values["num_users"] = len(splitted.users)
        date_values["num_assets_rec"] = len(splitted.assets)

        date_values_df = interaction_data.data[interaction_data.data[DEFAULT_TIMESTAMP_COL] < future_date]
        date_values_df = date_values_df[date_values_df[DEFAULT_USER_COL].isin(splitted.users)]

        date_values["train_ratings"] = splitted.train.shape[0]
        date_values["test_ratings"] = splitted.test.shape[0]

        date_values["train_transactions"] = date_values_df[(date_values_df[DEFAULT_TIMESTAMP_COL] < rec_date)].shape[0]
        date_values["training_buys"] = date_values_df[(date_values_df[DEFAULT_TIMESTAMP_COL] < rec_date) & (date_values_df[DEFAULT_RATING_COL]) > 0].shape[0]
        date_values["positive_train_ratings"] = splitted.train[splitted.train[DEFAULT_RATING_COL] > 0].shape[0]

        date_values_df = date_values_df[date_values_df[DEFAULT_ITEM_COL].isin(splitted.assets)]
        date_values["test_transactions"] = date_values_df[date_values_df[DEFAULT_TIMESTAMP_COL] >= rec_date].shape[0]
        date_values["test_buys"] = date_values_df[(date_values_df[DEFAULT_TIMESTAMP_COL] >= rec_date) & (date_values_df[DEFAULT_RATING_COL] > 0)].shape[0]
        date_values["positive_test_ratings"] = splitted.test[splitted.test[DEFAULT_RATING_COL] > 0].shape[0]

        values.append(date_values)

        print("Finished " + str(dates[i]))
    df = pd.DataFrame(values)
    df.set_index("timestamp")
    df.to_csv(os.path.join(directory, end_file))
