#  Copyright (c) 2022. Terrier Team at University of Glasgow, http://http://terrierteam.dcs.gla.ac.uk
#
#  This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
#  If a copy of the MPL was not distributed with this  file, you can obtain one at
#  http://mozilla.org/MPL/2.0/.
#
#  This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
#  If a copy of the MPL was not distributed with this  file, you can obtain one at
#  http://mozilla.org/MPL/2.0/.

from beta_rec.data import BaseData
from beta_rec.utils.constants import (
    DEFAULT_ITEM_COL,
    DEFAULT_RATING_COL,
    DEFAULT_TIMESTAMP_COL,
    DEFAULT_USER_COL
)


class FinancialData:
    """
    Class for storing the financial data necessary to generate the recommendations.
    """

    def __init__(self, interactions, asset_time_series):
        """
        Initializes the data.
        :param interactions: the interactions data.
        :param asset_time_series: the time series data.
        """
        self.interactions = interactions
        self.time_series = asset_time_series
        self.training_date = None
        self.valid_date = None
        self.test_date = None
        self.training = None
        self.test = None
        self.valid = None
        self.users = set()
        self.assets = set()
        self.positive_assets = dict()
        self.split = None

    def valid_load(self, training_date, valid_date):
        """
        Loads the corresponding data.
        :param training_date: the maximum date from training interactions / prices.
        :param valid_date: the maximum date from validation interactions / prices.
        """

        # As a first step, load the interactions and time series from file
        self.interactions.load()
        self.time_series.load()

        # Get the training, validation and test split dates.
        self.training_date = training_date
        self.valid_date = valid_date
        self.test_date = valid_date

        # Splits the interaction dataset into training, validation and test sets, and stores them.
        inter_train, inter_valid = self.interactions.valid_divide(training_date, valid_date)
        self.training = inter_train
        self.valid = inter_valid
        self.test = inter_valid

        # Gets the users and assets to consider for the training and evaluation of the algorithms.
        # We copy the dataframes to prevent modification by BaseData:
        self.split = BaseData((self.training.copy(), [self.valid.copy()], [self.test.copy()]))
        self.users = self.split.user2id.keys()
        assets_inter = self.split.item2id.keys()

        last_time = self.time_series.data[self.time_series.data[DEFAULT_TIMESTAMP_COL] == self.test_date]
        assets_time = last_time[DEFAULT_ITEM_COL].unique()
        assets_time = set(assets_time.flatten())
        self.assets = assets_time.intersection(assets_inter)

        self.time_series.data = self.time_series.data[self.time_series.data[DEFAULT_ITEM_COL].isin(assets_inter)]

        # Gets the positive assets for each user in the test set.
        self.positive_assets.clear()
        posit = inter_valid[inter_valid[DEFAULT_ITEM_COL].isin(self.assets)]
        posit = posit[posit[DEFAULT_RATING_COL] > 0.0]
        for user in self.users:
            self.positive_assets[user] = set(
                posit[posit[DEFAULT_USER_COL] == user][DEFAULT_ITEM_COL].unique().flatten())

    def load(self, training_date, valid_date, test_date):
        """
        Loads the corresponding data.
        :param training_date: the maximum date from training interactions / prices.
        :param valid_date: the maximum date from validation interactions / prices.
        :param test_date: the maximum date for test interactions / prices.
        """

        # As a first step, load the interactions and time series from file
        self.interactions.load()
        self.time_series.load()

        # Get the training, validation and test split dates.
        self.training_date = training_date
        self.valid_date = valid_date
        self.test_date = test_date

        # Splits the interaction dataset into training, validation and test sets, and stores them.
        inter_train, inter_valid, inter_test = self.interactions.divide(training_date, valid_date, test_date)
        self.training = inter_train
        self.valid = inter_valid
        self.test = inter_test

        # Gets the users and assets to consider for the training and evaluation of the algorithms:
        self.split = BaseData((inter_train.copy(), [inter_valid.copy()], [inter_test.copy()]))
        self.users = self.split.user2id.keys()
        assets_inter = self.split.item2id.keys()

        last_time = self.time_series.data[self.time_series.data[DEFAULT_TIMESTAMP_COL] == test_date]
        assets_time = last_time[DEFAULT_ITEM_COL].unique()
        assets_time = set(assets_time.flatten())
        self.assets = assets_time.intersection(assets_inter)
        self.time_series.data = self.time_series.data[self.time_series.data[DEFAULT_ITEM_COL].isin(assets_inter)]

        # Gets the positive assets for each user in the test set.
        self.positive_assets.clear()
        posit = inter_test[inter_test[DEFAULT_ITEM_COL].isin(self.assets)]
        posit = posit[posit[DEFAULT_RATING_COL] > 0.0]
        for user in self.users:
            self.positive_assets[user] = set(
                posit[posit[DEFAULT_USER_COL] == user][DEFAULT_ITEM_COL].unique().flatten())

    def get_customers(self):
        """
        Obtains the customers to use.
        :return: the customers to use.
        """
        return self.users

    def get_assets(self):
        """
        Obtains the assets to use.
        :return: the assets to use.
        """
        return self.assets

    def get_positive_assets(self, customer):
        return self.positive_assets[customer] if customer in self.positive_assets else set()
