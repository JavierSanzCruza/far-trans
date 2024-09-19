import datetime
import math

import pandas as pd
from beta_rec.utils.constants import (
    DEFAULT_TIMESTAMP_COL, DEFAULT_USER_COL
)

from data.filter.asset.asset_with_full_test import AssetWithFullTest
from data.filter.customer.customer_in_train import CustomerInTrain
from data.filter.data_filter import DataFilter
from data.filter.rating.ratings_not_in_train import RatingsNotInTrain
from data.filter.timeseries.no_filter import NoFilter
from data.financial_asset_time_series import FinancialAssetTimeSeries
from data.financial_interaction_data import FinancialInteractionData
from data.splitted_data import SplittedData


class FinancialContinuousData:
    """
    Class for storing financial data for recommendations. This class allows splitting the
    data at multiple types.
    """

    def __init__(self, interactions: FinancialInteractionData, asset_time_series: FinancialAssetTimeSeries,
                 customers=None):
        """
        Initializes the data.
        :param interactions: the interactions data.
        :param asset_time_series: the time series data.
        :param customers: a list of customers.
        """
        self.interactions = interactions
        self.time_series = asset_time_series
        self.kpis = None
        self.customers = customers

    def load(self):
        """
        Loads the data.
        """
        self.interactions.load()
        self.customers = self.customers if self.customers is not None else set(self.interactions.data[DEFAULT_USER_COL].unique())
        self.time_series.load()

    def add_kpis(self, kpis):
        """
        Adds the computed kpis
        :param kpis: the computed kpis.
        """
        self.kpis = kpis

    def split(self, min_date: datetime.datetime, rec_date: datetime.datetime, max_date: datetime.datetime,
              data_filter: DataFilter = DataFilter(CustomerInTrain(), AssetWithFullTest(), RatingsNotInTrain(),
                                                   NoFilter(), False, True, False)):
        """
        Splits the dataset into training and test.
        :param data_filter: filters the data after the split.
        :param min_date: the minimum possible date to consider.
        :param rec_date: the recommendation date.
        :param max_date: the maximum possible date.
        :return: the splitted data.
        """
        time_series = self.time_series.data[self.time_series.data[DEFAULT_TIMESTAMP_COL].between(min_date, max_date)]
        train, test = self.interactions.split(min_date, rec_date, max_date)

        kpi_series = None
        if self.kpis is not None:
            print(min_date)
            print(max_date)
            kpi_series = self.kpis[self.kpis[DEFAULT_TIMESTAMP_COL].between(min_date, max_date)]

        customers, assets, time_series, train, valid, test = data_filter.filter(self.customers, time_series, train, None, test, rec_date)

        splitted = SplittedData(time_series, train, test, customers, assets, min_date, rec_date, max_date)
        if self.kpis is not None:
            splitted.add_kpis(kpi_series)
        return splitted

    def split_with_valid(self, min_date: datetime.datetime, valid_date: datetime.datetime, rec_date: datetime.datetime,
                         max_date: datetime.datetime,
                         data_filter: DataFilter = DataFilter(CustomerInTrain(), AssetWithFullTest(),
                                                              RatingsNotInTrain(),
                                                              NoFilter(), False, True, False)):
        """
        Splits the dataset into training, validation and test.
        :param min_date: the minimum possible date to consider.
        :param valid_date: the validation date.
        :param rec_date: the recommendation date.
        :param max_date: the maximum possible date.
        :param data_filter: filters the data after the split.
        :return: the splitted data.
        """
        time_series = self.time_series.data[self.time_series.data[DEFAULT_TIMESTAMP_COL].between(min_date, max_date)]
        train, valid = self.interactions.split(min_date, valid_date, rec_date)
        aux_valid, test = self.interactions.split(valid_date, rec_date, max_date)

        kpi_series = None
        if self.kpis is not None:
            kpi_series = self.kpis[self.kpis[DEFAULT_TIMESTAMP_COL].between(min_date, max_date)]

        customers, assets, time_series, train, valid, test = data_filter.filter(self.customers, time_series, train, valid, test, rec_date)

        splitted = SplittedData(time_series, train, valid, test, customers, assets, min_date, valid_date, rec_date,
                                max_date)
        if self.kpis is not None:
            splitted.add_kpis(kpi_series)
        return splitted

    def get_dates(self, min_date, max_date, num_split, num_future):
        """
        Gets a collection of split dates.
        :param min_date: the minimum considered recommendation date.
        :param max_date: the maximum considered recommendation date.
        :param num_split: the number of splits we want.
        :return: the list of dates.
        """

        aux = self.time_series.data[self.time_series.data[DEFAULT_TIMESTAMP_COL].between(min_date, max_date)]
        dates = aux[DEFAULT_TIMESTAMP_COL].unique().flatten()
        dates.sort()
        n_dates = len(dates)

        total_splits = num_split + math.ceil(num_future)
        division = math.floor(n_dates / total_splits)
        partial_dates = [pd.to_datetime(str(dates[0]))]

        # We first do this:
        for i in range(1, total_splits):
            partial_dates.append(pd.to_datetime(str(dates[i * division])))
        partial_dates.append(pd.to_datetime(str(dates[-1])))

        def_dates = []
        def_future_dates = []
        for i in range(0, len(partial_dates) - math.ceil(num_future)):
            def_dates.append(partial_dates[i])
            if num_future - math.floor(num_future) > 0:
                start = partial_dates[i + math.floor(num_future)]
                end = partial_dates[i + math.floor(num_future) + 1]

                dates = self.time_series.data[self.time_series.data[DEFAULT_TIMESTAMP_COL].between(start, end)]
                val = max(1, math.floor((num_future - math.floor(num_future))*len(dates)))
                def_future_dates.append(dates[val])
            else:
                def_future_dates.append(partial_dates[i + num_future])
        return def_dates, def_future_dates
