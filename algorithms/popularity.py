#  Copyright (c) 2022. Terrier Team at University of Glasgow, http://http://terrierteam.dcs.gla.ac.uk
#
#  This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
#  If a copy of the MPL was not distributed with this  file, you can obtain one at
#  http://mozilla.org/MPL/2.0/.
#
#  This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
#  If a copy of the MPL was not distributed with this  file, you can obtain one at
#  http://mozilla.org/MPL/2.0/.

import pandas as pd
from beta_rec.utils.constants import DEFAULT_ITEM_COL, DEFAULT_USER_COL, DEFAULT_RATING_COL

from algorithms.algorithm import Algorithm


class PopularityAlgorithm(Algorithm):
    """
    Algorithm that recommends those assets on which more customers have invested.
    """
    def __init__(self, data):
        """
        Configures the profitability prediction model.
        :param data: the data for training and applying recommendations.
        """
        super().__init__(data)

        self.assets_df = None

    def train(self, train_date):

        popularity = dict()
        for asset in self.data.assets:
            asset_df = self.data.train[(self.data.train[DEFAULT_ITEM_COL] == asset) &
                                       (self.data.train[DEFAULT_RATING_COL] > 0.0)]
            popularity[asset] = asset_df.shape[0]

        self.assets_df = pd.DataFrame(popularity.items(), columns=[DEFAULT_ITEM_COL, DEFAULT_RATING_COL])
        self.assets_df = self.assets_df.sort_values(by=DEFAULT_RATING_COL, ascending=False)

    def recommend(self, rec_time, repeated, only_test_customers):
        user_recommendations = []
        customers = (self.data.users & set(self.data.test[DEFAULT_USER_COL].unique().flatten())) if only_test_customers else self.data.users

        for customer in customers:
            user_recommendation = self.assets_df.copy()
            user_recommendation[DEFAULT_USER_COL] = customer
            if not repeated:
                items_per_user = set(self.data.train[self.data.train[DEFAULT_USER_COL] == customer][DEFAULT_ITEM_COL].unique().flatten())
                user_recommendation = user_recommendation[~user_recommendation[DEFAULT_ITEM_COL].isin(items_per_user)]
            user_recommendation = user_recommendation[user_recommendation[DEFAULT_ITEM_COL].isin(self.data.assets)]
            user_recommendations.append(user_recommendation)
        return pd.concat(user_recommendations)
