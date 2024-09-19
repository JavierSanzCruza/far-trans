#  Copyright (c) 2023. Terrier Team at University of Glasgow, http://http://terrierteam.dcs.gla.ac.uk
#
#  This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
#  If a copy of the MPL was not distributed with this  file, you can obtain one at
#  http://mozilla.org/MPL/2.0/.

from abc import abstractmethod, ABC

__author__ = "Javier Sanz-Cruzado (javier.sanz-cruzadopuig@glasgow.ac.uk)"


class GroupAlgorithm(ABC):
    """
    Finds a recommendation for a group of customers
    """

    def __init__(self, data, group_build):
        self.data = data
        self.group_build = group_build

    @abstractmethod
    def train(self, train_date):
        """
        Trains the recommendation algorithm.
        :param train_date: the training date.
        """
        pass

    @abstractmethod
    def recommend(self, rec_date):
        """
        Recommends assets to a group of customers.
        :param rec_date: the recommendation date.
        :return: the recommendation for every group of customers.
        """
        pass
