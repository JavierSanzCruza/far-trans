#  Copyright (c) 2022-2023. Terrier Team at University of Glasgow, http://http://terrierteam.dcs.gla.ac.uk
#
#  This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
#  If a copy of the MPL was not distributed with this  file, you can obtain one at
#  http://mozilla.org/MPL/2.0/.
#
#  This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
#  If a copy of the MPL was not distributed with this  file, you can obtain one at
#  http://mozilla.org/MPL/2.0/.
#
#  This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
#  If a copy of the MPL was not distributed with this  file, you can obtain one at
#  http://mozilla.org/MPL/2.0/.

import pandas as pd
from beta_rec.utils.constants import (
    DEFAULT_ITEM_COL,
    DEFAULT_RATING_COL,
    DEFAULT_USER_COL,
)

from .algorithm import Algorithm


class ReadFromRankingAlgorithm(Algorithm):
    """
    Algorithm that reads a ranking from a file.
    """

    def __init__(self, data, file):
        """
        Initializes the algorithm.
        :param data: the data to use.
        :param file: the file. It should be formated as col_user \t col_item \t col_rating
        """
        super().__init__(data)
        self.file = file
        self.scores = None

    def train(self, train_date):
        rec_df = pd.read_csv(self.file)
        rec_df = rec_df.dropna()
        rec_df = rec_df.drop_duplicates(subset=[DEFAULT_ITEM_COL], keep="first")
        rec_df = rec_df.sort_values(by=DEFAULT_RATING_COL, ascending=False)
        self.scores = rec_df[[DEFAULT_ITEM_COL, DEFAULT_RATING_COL]]

    def recommend(self, rec_date, repeated, only_test_customers):
        customers = (self.data.users & set(self.data.test[DEFAULT_USER_COL].unique().flatten())) if only_test_customers else self.data.users

        aux_cust = [x for x in customers]
        aux_assets = [x for x in self.data.assets]

        user_recommendations = []

        for customer in customers:
            user_recommendation = self.scores.copy()
            user_recommendation[DEFAULT_USER_COL] = customer

            if not repeated:
                items_per_user = set(self.data.train[self.data.train[DEFAULT_USER_COL] == customer][
                                         DEFAULT_ITEM_COL].unique().flatten())
                user_recommendation = user_recommendation[~user_recommendation[DEFAULT_ITEM_COL].isin(items_per_user)]
            user_recommendation = user_recommendation[user_recommendation[DEFAULT_ITEM_COL].isin(self.data.assets)]
            user_recommendation = user_recommendation.sort_values(by=DEFAULT_RATING_COL, ascending=False)
            user_recommendations.append(user_recommendation)
        def_df = pd.concat(user_recommendations)
        return def_df
