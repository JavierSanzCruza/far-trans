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
    DEFAULT_USER_COL
)


class SplittedData:
    """
    Class for representing the splitted data.
    """

    def __init__(self, time_series, train, test, customers, assets, min_date, rec_date, max_date, valid=None,
                 valid_date=None):
        """
        Representation of the splitted data:
        :param time_series: a DataFrame containing the time series for the split.
        :param train: the training data
        :param test: the test data.
        :param customers: the set of customers to consider.
        :param assets: the assets to consider.
        :param min_date: the minimum considered date.
        :param rec_date: the recommendation date (split date).
        :param max_date: the maximum considered date.
        """
        self.time_series = time_series
        self.train = train
        self.valid = valid
        self.test = test
        self.split = BaseData((self.train.copy(), [self.test.copy()], [self.test.copy()]))
        self.users = customers
        self.assets = assets

        self.positive_assets = dict()
        posit = self.test[self.test[DEFAULT_ITEM_COL].isin(self.assets)]
        posit = posit[posit[DEFAULT_RATING_COL] > 0.0]
        for user in self.test[DEFAULT_USER_COL].unique().flatten():
            self.positive_assets[user] = set(posit[posit[DEFAULT_USER_COL] == user][DEFAULT_ITEM_COL].unique().flatten())

        self.min_date = min_date
        self.max_date = max_date
        self.valid_date = valid_date
        self.rec_date = rec_date
        self.kpis = None

    def add_kpis(self, kpis):
        """
        Adds the set of computed technical indicators.
        :param kpis: the technical indicators to consider.
        """
        self.kpis = kpis

    def get_positive_assets(self, customer):
        """
        Gets the assets acquired by a customer in the test set.
        :param customer: the customer identifier.
        :return: the list of assets acquired by the customer in the test set.
        """
        if customer in self.positive_assets:
            return self.positive_assets[customer]
        return []
