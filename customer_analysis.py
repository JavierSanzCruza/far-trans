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
import math
import os
import sys

import argparse
import numpy
import pandas as pd
from beta_rec.utils.constants import DEFAULT_TIMESTAMP_COL, DEFAULT_ITEM_COL, DEFAULT_RATING_COL

from data.financial_asset_time_series import FinancialAssetTimeSeries
from data.financial_data_continuous import FinancialContinuousData
from data.financial_interaction_data import FinancialInteractionData

pd.options.mode.chained_assignment = None  # default='warn'

if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        prog="dataset_analysis",
        description="Analyzes the dataset from a general asset point of view.",
        epilog="Developed by University of Glasgow"
    )

    parser.add_argument("interactions", help="Customer-asset transaction data file.")
    parser.add_argument("time_series", help="Asset pricing data file.")
    parser.add_argument("min_prices", help="File containing the minimum and maximum asset prices.")
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
    min_prices = args.min_prices
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

    timea = dt.datetime.now()

    # Now, we load the data:

    # Now, we load the data:
    interaction_data = FinancialInteractionData(interaction_data_file, repeated=False)
    time_series_data = FinancialAssetTimeSeries(time_series_data_file)
    data = FinancialContinuousData(interaction_data, time_series_data)
    data.load()

    limit_prices = pd.read_csv(min_prices)
    limit_prices["min"] = pd.to_datetime(limit_prices["min"])
    limit_prices["max"] = pd.to_datetime(limit_prices["max"])
    # Then, we get the profile data
    profile_df = pd.read_csv(interaction_data_file)
    profile_df["timestamp"] = pd.to_datetime(profile_df["timestamp"])
    profile_df = profile_df.sort_values(by=["customerID", "timestamp"], ascending=[True, True])
    timeb = dt.datetime.now() - timea
    print("Dataset loaded (" + '{}'.format(timeb) + ")")

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

    columns = ["timestamp"]
    fields = ["mean", "total_prof", "q1", "median", "q3", "min", "max", "perc_profit"]
    for x in ["profitability", "month_profit", "ann_profit"]:
        for field in fields:
            columns.append(x+field)
    user_profiles = dict()

    splits = dict()
    splits[0] = (profile_df[profile_df["timestamp"] < future_dates[0]],
                 profile_df[profile_df["timestamp"] < dates[0]],
                 profile_df[profile_df["timestamp"].between(dates[0], future_dates[0])])

    aux_df = profile_df[profile_df["timestamp"] >= dates[0]]
    for i in range(1, len(dates)):
        splits[i] = (aux_df[aux_df["timestamp"] < future_dates[i]],
                     aux_df[aux_df["timestamp"] < dates[i]],
                     aux_df[aux_df["timestamp"].between(dates[i], future_dates[i])])
        aux_df = aux_df[aux_df["timestamp"] >= dates[i]]

    timeb = dt.datetime.now() - timea
    print("Splits computed (" + '{}'.format(timeb) + ")")

    values = []
    # We run over the complete set of dates:
    for i in range(0, len(dates)):

        print(splits[i][1])

        timeb = dt.datetime.now() - timea
        print("Starting " + str(dates[i]) + " (" + '{}'.format(timeb) + ")")
        rec_date = dates[i]
        future_date = future_dates[i]
        date_values = dict()
        date_values["timestamp"] = rec_date

        customers = []
        buy_prices = []
        sell_prices = []
        profitability = []
        ann_profit = []
        monthly_profit = []

        total_buy_price = 0.0
        total_sell_price = 0.0

        aux_df = splits[i][0]

        customer_set = set(aux_df["customerID"].unique().flatten()) | user_profiles.keys()

        print("Num. customers: " + str(len(customer_set)))

        user_count = 0
        # For each customer:
        for customer in customer_set:
            stored_pers_inf = False
            cust_type = None
            cust_cat = None
            cust_risk = None

            # We first take the corresponding split
            customer_df = splits[i][1][splits[i][1]["customerID"] == customer]

            buy_price = 0.0
            sell_price = 0.0

            # We update its user profile
            if i == 0:
                user_profile = dict()
            else:
                user_profile = user_profiles[customer] if customer in user_profiles else dict()

            for index, row in customer_df.iterrows():
                if row["ISIN"] not in user_profile:
                    user_profile[row["ISIN"]] = 0.0

                if row["transactionType"] == "Buy":
                    user_profile[row["ISIN"]] += abs(row["units"])+0.0
                else:
                    user_profile[row["ISIN"]] -= abs(row["units"])+0.0

                if user_profile[row["ISIN"]] < 1e-8:
                    user_profile[row["ISIN"]] = 0.0
            prices = time_series_data.data[time_series_data.data[DEFAULT_TIMESTAMP_COL] == rec_date]
            prices = prices[prices[DEFAULT_ITEM_COL].isin(user_profile.keys())]

            # We first compute the buying price at the current date.
            for index, row in prices.iterrows():
                buy_price += abs(row[DEFAULT_RATING_COL])*user_profile[row[DEFAULT_ITEM_COL]]

            not_use = set()
            for asset in user_profile.keys() - set(prices[DEFAULT_ITEM_COL].unique().flatten()):
                if rec_date < limit_prices[limit_prices["ISIN"] == asset].iloc[0]["minDate"]:
                    price = limit_prices[limit_prices["ISIN"] == asset].iloc[0]["priceMinDate"]
                    buy_price += abs(price)*user_profile[asset]
                else:
                    price = 0
                    user_profile.pop(asset)
                    not_use.add(asset)

            # Starting here, we start working on the assets acquired during the corresponding period.
            customer_df = splits[i][2][splits[i][2]["customerID"] == customer]
            user_period_profile = user_profile.copy()
            for index, row in customer_df.iterrows():
                if row["ISIN"] in not_use:
                    continue

                if row["ISIN"] not in user_period_profile:
                    user_period_profile[row["ISIN"]] = 0.0
                if row["transactionType"] == "Buy":
                    user_period_profile[row["ISIN"]] += abs(row["units"])+0.0
                    if user_period_profile[row["ISIN"]] < 1e-8:
                        user_period_profile[row["ISIN"]] = 0.0
                    buy_price += abs(row["units"])
                else:
                    user_period_profile[row["ISIN"]] -= abs(row["units"])+0.0
                    if user_period_profile[row["ISIN"]] < 1e-8:
                        user_period_profile[row["ISIN"]] = 0.0
                    sell_price += abs(row["units"])

            prices = time_series_data.data[time_series_data.data[DEFAULT_TIMESTAMP_COL] == future_date]
            prices = prices[prices[DEFAULT_ITEM_COL].isin(user_period_profile.keys())]

            for index, row in prices.iterrows():
                sell_price += abs(row[DEFAULT_RATING_COL]) * user_period_profile[row[DEFAULT_ITEM_COL]]
            for asset in user_period_profile.keys() - set(prices[DEFAULT_ITEM_COL].unique().flatten()):
                if future_date < limit_prices[limit_prices["ISIN"] == asset].iloc[0]["minDate"]:
                    price = limit_prices[limit_prices["ISIN"] == asset].iloc[0]["minDatePrice"]
                    sell_price += abs(price) * user_period_profile[asset]
                elif future_date > limit_prices[limit_prices["ISIN"] == asset].iloc[0]["maxDate"]:
                    price = limit_prices[limit_prices["ISIN"] == asset].iloc[0]["maxDatePrice"]
                    sell_price += abs(price) * user_period_profile[asset]
                else:
                    price = 0
                    user_period_profile.pop(asset)
                    not_use.add(asset)

            if buy_price > 0:
                customers.append(customer)
                buy_prices.append(buy_price)
                sell_prices.append(sell_price)
                power = 365.0 / (future_date - rec_date).days
                profita = pow(sell_price/buy_price, power) - 1.0
                profitability.append(sell_price/buy_price - 1.0)
                power = 30.0 / (future_date - rec_date).days
                profitm = pow(sell_price/buy_price, power) - 1.0
                monthly_profit.append(profitm)
                ann_profit.append(profita)

                user_profiles[customer] = user_profile

            total_buy_price += buy_price
            total_sell_price += sell_price

            user_count += 1
            if user_count % 100 == 0:
                timeb = dt.datetime.now() - timea
                print("Processed " + str(user_count) + " customers (" + '{}'.format(timeb) + ")")

        customer_analysis_df = pd.DataFrame(
            {"customer": customers, "buy_price": buy_prices, "sell_price": sell_prices, "profitability": profitability,
             "ann_profit": ann_profit, "month_profit": monthly_profit})

        customer_analysis_df.to_csv(os.path.join(directory, "customers_stats_" +
                                                 rec_date.strftime("%Y-%m-%d") + ".csv"),
                                    index=False)

        value = [rec_date]

        for x in ["profitability", "month_profit", "ann_profit"]:

            mean = customer_analysis_df[x].mean()
            min = customer_analysis_df[x].min()
            max = customer_analysis_df[x].max()
            q1 = customer_analysis_df[x].quantile(0.25)
            median = customer_analysis_df[x].quantile(0.5)
            q3 = customer_analysis_df[x].quantile(0.75)

            quants = numpy.quantile(customer_analysis_df[x], [0, 0.25, 0.5, 0.75, 1.0])
            total_profit = (total_sell_price - total_buy_price)/total_buy_price
            if x == "month_profit":
                total_profit = math.pow(1+total_profit, 30.0/((future_date-rec_date).days + 0.0))-1
            elif x == "ann_profit":
                total_profit = math.pow(1+total_profit, 365.0 / ((future_date - rec_date).days + 0.0))-1

            perc_prof = (len([x for x in customer_analysis_df[x] if x > 0.0])+0.0)/(len(customer_analysis_df[x]) + 0.0)

            value.append(mean)
            value.append(total_profit)
            value.append(q1)
            value.append(median)
            value.append(q3)
            value.append(min)
            value.append(max)
            value.append(perc_prof)

        print(columns)
        print(value)

        values.append(value)
        timeb = dt.datetime.now() - timea
        print("Finished " + str(dates[i]) + " (" + '{}'.format(timeb) + ")")

    df = pd.DataFrame(values, columns=columns)
    df.set_index("timestamp")
    df.to_csv(os.path.join(directory, end_file), index=False)
