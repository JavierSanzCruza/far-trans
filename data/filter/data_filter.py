#  Copyright (c) 2022. Terrier Team at University of Glasgow, http://http://terrierteam.dcs.gla.ac.uk
#
#  This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
#  If a copy of the MPL was not distributed with this  file, you can obtain one at
#  http://mozilla.org/MPL/2.0/.
#
#  This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
#  If a copy of the MPL was not distributed with this  file, you can obtain one at
#  http://mozilla.org/MPL/2.0/.

from typing import Set, Tuple, Optional

from beta_rec.utils.constants import DEFAULT_USER_COL, DEFAULT_ITEM_COL
from pandas import DataFrame

from data.filter.asset.asset_filter import AssetFilter
from data.filter.customer.customer_filter import CustomerFilter
from data.filter.rating.rating_filter import RatingFilter
from data.filter.timeseries.time_series_filter import TimeSeriesFilter


class DataFilter:
    """
    Class for processing the financial data after a split is done.
    """

    def __init__(self, customer_filter: CustomerFilter, asset_filter: AssetFilter,
                 rating_filter: RatingFilter, ts_filter: TimeSeriesFilter,
                 apply_cf_to_ratings: bool, apply_af_to_ts: bool, apply_af_to_ratings: bool):
        """
        Initializes the filter.
        :param customer_filter: the filter for retrieving the customers.
        :param asset_filter: the filter for retrieving the assets.
        :param rating_filter: the filter for the ratings.
        :param ts_filter: the filter for time series datapoints.
        :param apply_cf_to_ratings: true if we want to apply the customer filter to all interaction partitions.
        :param apply_af_to_ts: true if we want to apply the asset filter to the time series.
        :param apply_af_to_ratings: true if we want to apply the asset filter to all interaction partitions.
        """
        self.customer_filter = customer_filter
        self.asset_filter = asset_filter
        self.rating_filter = rating_filter
        self.ts_filter = ts_filter
        self.apply_af_to_ts = apply_af_to_ts
        self.apply_cf_to_ratings = apply_cf_to_ratings
        self.apply_af_to_ratings = apply_af_to_ratings

    def filter(self, customers, time_series: DataFrame, train: DataFrame, valid: Optional[DataFrame], test: DataFrame, split_date) -> \
            Tuple[Set, Set, DataFrame, DataFrame, Optional[DataFrame], DataFrame]:
        """
        Applies the filter.
        :param time_series: the time series.
        :param train: the training data
        :param valid: the validation data
        :param test: the test data
        :return: in this order: customers, assets, time series, train split, validation split,
        test split.
        """
        customers = self.customer_filter.filter(customers, time_series, train, valid, test)
        assets = self.asset_filter.filter(time_series, train, valid, test, split_date)
        time_series_aux = self.ts_filter.filter(time_series, train, valid, test)
        ratings_train, ratings_valid, ratings_test = self.rating_filter.filter(time_series, train, valid, test)

        if self.apply_cf_to_ratings:
            ratings_train = ratings_train[ratings_train[DEFAULT_USER_COL].isin(customers)]
            if ratings_valid is not None:
                ratings_valid = ratings_valid[ratings_valid[DEFAULT_USER_COL].isin(customers)]
            ratings_test = ratings_test[ratings_test[DEFAULT_USER_COL].isin(customers)]

        if self.apply_af_to_ratings:
            ratings_train = ratings_train[ratings_train[DEFAULT_ITEM_COL].isin(assets)]
            if ratings_valid is not None:
                ratings_valid = ratings_valid[ratings_valid[DEFAULT_ITEM_COL].isin(assets)]
            ratings_test = ratings_test[ratings_test[DEFAULT_ITEM_COL].isin(assets)]

        if self.apply_af_to_ts:
            time_series_aux = time_series_aux[time_series_aux[DEFAULT_ITEM_COL].isin(assets)]

        return customers, assets, time_series_aux, ratings_train, ratings_valid, ratings_test





