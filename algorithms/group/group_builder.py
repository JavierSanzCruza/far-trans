#  Copyright (c) 2023. Terrier Team at University of Glasgow, http://http://terrierteam.dcs.gla.ac.uk
#
#  This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
#  If a copy of the MPL was not distributed with this  file, you can obtain one at
#  http://mozilla.org/MPL/2.0/.

__author__ = "Javier Sanz-Cruzado (javier.sanz-cruzadopuig@glasgow.ac.uk)"

from abc import ABC, abstractmethod


class GroupBuilder(ABC):
    """
    Builds groups (partitions) of users
    """

    def __init__(self, data, customer_data):
        """
        Constructor.
        :param data: full financial recommendation data
        :param customer_data: information about the customers.
        """
        self.data = data
        self.customer_data = customer_data

    @abstractmethod
    def group(self, date):
        """
        Groups the customers into groups
        :param date: the date to use
        :return: two dictionaries: one indexed by group id containing lists of customers, and another one,
        indexed by customer id, containing its group
        """
        pass
