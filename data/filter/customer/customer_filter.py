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
from typing import Set, Optional

from pandas import DataFrame


class CustomerFilter(ABC):
    """
    Filters the customers in a dataset.
    """

    @abstractmethod
    def filter(self, customers: set, time_series: DataFrame, train: DataFrame, valid: Optional[DataFrame], test: DataFrame) -> Set:
        """
        Selects the customers to consider.
        :param time_series: the time series in the dataset.
        :param train: the training set
        :param valid: the validation set
        :param test: the test set.
        :return: the set of customers to consider.
        """
        pass

