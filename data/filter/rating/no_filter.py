#  Copyright (c) 2022. Terrier Team at University of Glasgow, http://http://terrierteam.dcs.gla.ac.uk
#
#  This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
#  If a copy of the MPL was not distributed with this  file, you can obtain one at
#  http://mozilla.org/MPL/2.0/.
from typing import Tuple, Optional

import pandas as pd

from data.filter.rating.rating_filter import RatingFilter


class NoFilter(RatingFilter):
    """
    Class that does not apply any filtering to the interaction splits.
    """
    def filter(self, time_series: pd.DataFrame, train: pd.DataFrame, valid: pd.DataFrame, test: pd.DataFrame) -> \
            Tuple[pd.DataFrame, Optional[pd.DataFrame], pd.DataFrame]:
        ratings_train = train.copy()
        ratings_valid = valid.copy() if valid is not None else None
        ratings_test = test.copy()

        return ratings_train, ratings_valid, ratings_test