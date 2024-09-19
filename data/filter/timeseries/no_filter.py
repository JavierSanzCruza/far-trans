#  Copyright (c) 2022. Terrier Team at University of Glasgow, http://http://terrierteam.dcs.gla.ac.uk
#
#  This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
#  If a copy of the MPL was not distributed with this  file, you can obtain one at
#  http://mozilla.org/MPL/2.0/.

from pandas import DataFrame

from data.filter.timeseries.time_series_filter import TimeSeriesFilter


class NoFilter(TimeSeriesFilter):
    """
    Filter that leaves the time series as it is.
    """
    def filter(self, time_series: DataFrame, train: DataFrame, valid: DataFrame, test: DataFrame) -> DataFrame:
        return time_series.copy()
