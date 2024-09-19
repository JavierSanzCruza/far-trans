#  Copyright (c) 2022. Terrier Team at University of Glasgow, http://http://terrierteam.dcs.gla.ac.uk
#
#  This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
#  If a copy of the MPL was not distributed with this  file, you can obtain one at
#  http://mozilla.org/MPL/2.0/.

from typing import Tuple, Optional

import pandas as pd
from beta_rec.utils.constants import DEFAULT_USER_COL, DEFAULT_ITEM_COL

from data.filter.rating.rating_filter import RatingFilter


class RatingsNotInTrain(RatingFilter):
    """
    Class that removes from the test set (validation set) those interactions previously held by customers.
    Ex: if customer 1 interacted with asset 3 in both training and test sets, it removes the interaction from the
    test set.
    """
    def filter(self, time_series: pd.DataFrame, train: pd.DataFrame, valid: pd.DataFrame, test: pd.DataFrame) -> \
            Tuple[pd.DataFrame, Optional[pd.DataFrame], pd.DataFrame]:
        ratings_train = train.copy()
        ratings_valid = valid.copy() if valid is not None else None
        ratings_test = test.copy()

        if ratings_valid is not None:
            ratings_valid = self.clean(ratings_train, ratings_valid)
            ratings_test = self.clean(ratings_train, ratings_test)
            ratings_test = self.clean(ratings_valid, ratings_test)
        else:
            ratings_test = self.clean(ratings_train, ratings_test)

        return ratings_train, ratings_valid, ratings_test

    @staticmethod
    def clean(initial: pd.DataFrame, final: pd.DataFrame) -> pd.DataFrame:
        """
        Given two datasets, removes the interactions in the second dataset between customers and assets who interacted
        during the first one.
        :param initial: the initial dataset.
        :param final: the final dataset.
        :return: the cleaned final dataset.
        """
        users_df = []

        if initial.shape[0] > 0:
            customers = set(initial[DEFAULT_USER_COL].unique().flatten()) & set(final[DEFAULT_USER_COL].unique().flatten())
            if len(customers) == 0:
                return initial.copy()
            for customer in customers:
                items_per_user = initial[initial[DEFAULT_USER_COL] == customer][DEFAULT_ITEM_COL].unique().flatten()
                user_df = final[(final[DEFAULT_USER_COL] == customer) & ~final[DEFAULT_ITEM_COL].isin(items_per_user)]
                users_df.append(user_df)
            return pd.concat(users_df)
        return initial.copy()
