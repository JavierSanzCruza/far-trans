#  Copyright (c) 2022. Terrier Team at University of Glasgow, http://http://terrierteam.dcs.gla.ac.uk
#
#  This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
#  If a copy of the MPL was not distributed with this  file, you can obtain one at
#  http://mozilla.org/MPL/2.0/.
#
#  This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
#  If a copy of the MPL was not distributed with this  file, you can obtain one at
#  http://mozilla.org/MPL/2.0/.

import numpy as np
import pandas as pd
from beta_rec.utils.constants import (
    DEFAULT_ITEM_COL,
    DEFAULT_RATING_COL,
    DEFAULT_USER_COL,
)

from .algorithm import Algorithm


class RandomAlgorithm(Algorithm):
    """
    Algorithm that sorts assets randomly.
    """

    def __init__(self, data):
        """
        Initializes the algorithm.
        :param data: the data to use.
        """
        super().__init__(data)

    def train(self, train_date):
        return

    def recommend(self, rec_date, repeated, only_test_customers):
        customers = (self.data.users & set(self.data.test[DEFAULT_USER_COL].unique().flatten())) if only_test_customers else self.data.users

        aux_cust = [x for x in customers]
        aux_assets = [x for x in self.data.assets]
        index = pd.MultiIndex.from_product([aux_cust, aux_assets], names=[DEFAULT_USER_COL, DEFAULT_ITEM_COL])
        test = pd.DataFrame(index=index).reset_index()

        num_samples = test.shape[0]
        vals = np.random.random(num_samples).flatten()
        test[DEFAULT_RATING_COL] = vals

        user_recommendations = []

        for customer in customers:
            user_recommendation = test[test[DEFAULT_USER_COL] == customer]

            if not repeated:
                items_per_user = set(self.data.train[self.data.train[DEFAULT_USER_COL] == customer][
                                         DEFAULT_ITEM_COL].unique().flatten())
                user_recommendation = user_recommendation[~user_recommendation[DEFAULT_ITEM_COL].isin(items_per_user)]
            user_recommendation = user_recommendation[user_recommendation[DEFAULT_ITEM_COL].isin(self.data.assets)]
            user_recommendation = user_recommendation.sort_values(by=DEFAULT_RATING_COL, ascending=False)
            user_recommendations.append(user_recommendation)
        def_df = pd.concat(user_recommendations)
        return def_df
