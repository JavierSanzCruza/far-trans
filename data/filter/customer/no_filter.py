#  Copyright (c) 2022. Terrier Team at University of Glasgow, http://http://terrierteam.dcs.gla.ac.uk
#
#  This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
#  If a copy of the MPL was not distributed with this  file, you can obtain one at
#  http://mozilla.org/MPL/2.0/.
from typing import Set

from pandas import DataFrame

from data.filter.customer.customer_filter import CustomerFilter


class NoCustomerFilter(CustomerFilter):
    """
    Filter that keeps all assets which contain full test, i.e. those test which appear in the test set and have pricing
    information at the end of the test period.
    """
    def filter(self, customers: set, time_series: DataFrame, train: DataFrame, valid: DataFrame, test: DataFrame) -> Set:
        return customers
