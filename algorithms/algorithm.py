#  Copyright (c) 2022. Terrier Team at University of Glasgow, http://http://terrierteam.dcs.gla.ac.uk
#
#  This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
#  If a copy of the MPL was not distributed with this  file, you can obtain one at
#  http://mozilla.org/MPL/2.0/.
#
#  This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
#  If a copy of the MPL was not distributed with this  file, you can obtain one at
#  http://mozilla.org/MPL/2.0/.

from abc import ABC, abstractmethod


class Algorithm(ABC):
    """
    Class defining a recommendation algorithm.
    """

    def __init__(self, data, history=None):
        """
        Initializes the recommendation algorithm
        :param data: recommendation data.
        :param history: past history of the customers.
        """
        self.data = data
        self.history = history

    @abstractmethod
    def train(self, train_date):
        """
        Trains the algorithm.
        :param train_date: the training date.
        """
        pass

    @abstractmethod
    def recommend(self, rec_date, repeated, only_test_customers):
        """
        Generates the recommendations.
        :param rec_date: the recommendation date.
        :param repeated: True if we allow customers to be recommended items they invested in the past.
        :param only_test_customers: true if we only consider customers with test data.
        :return: a pandas dataframe containing the recommendations. Format: col_user \t col_item \t score
        """
        pass
