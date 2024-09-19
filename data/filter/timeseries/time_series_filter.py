#  Copyright (c) 2022. Terrier Team at University of Glasgow, http://http://terrierteam.dcs.gla.ac.uk
#
#  This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
#  If a copy of the MPL was not distributed with this  file, you can obtain one at
#  http://mozilla.org/MPL/2.0/.

from abc import ABC, abstractmethod
from typing import Optional

from pandas import DataFrame


class TimeSeriesFilter(ABC):
    """
    Filters the time series in a dataset.
    """

    @abstractmethod
    def filter(self, time_series: DataFrame, train: DataFrame, valid: Optional[DataFrame], test: DataFrame) -> DataFrame:
        """
        Selects the points in the time series to consider.
        :param time_series: the time series in the dataset.
        :param train: the training set
        :param valid: the validation set
        :param test: the test set.
        :return: the filtered time series.
        """
        pass
