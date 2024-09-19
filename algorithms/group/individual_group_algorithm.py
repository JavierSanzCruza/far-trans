#  Copyright (c) 2023. Terrier Team at University of Glasgow, http://http://terrierteam.dcs.gla.ac.uk
#
#  This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
#  If a copy of the MPL was not distributed with this  file, you can obtain one at
#  http://mozilla.org/MPL/2.0/.
import pandas as pd
from beta_rec.utils.constants import DEFAULT_USER_COL, DEFAULT_ITEM_COL, DEFAULT_RATING_COL

from algorithms.algorithm import Algorithm
from algorithms.group.group_popularity import DEFAULT_GROUP_COL

__author__ = "Javier Sanz-Cruzado (javier.sanz-cruzadopuig@glasgow.ac.uk)"


class IndividualGroupAlgorithm(Algorithm):
    def __init__(self, data, group_alg, history=None):
        super().__init__(data, history)
        self.group_alg = group_alg

    def train(self, train_date):
        self.group_alg.train(train_date)

    def recommend(self, rec_date, repeated, only_test_customers):
        user_recommendations = []
        customers = (self.data.users & set(
            self.data.test[DEFAULT_USER_COL].unique().flatten())) if only_test_customers else self.data.users

        group_recs = self.group_alg.recommend(rec_date)
        cust_recs = []
        for customer in customers:
            group = self.group_alg.groups[0][customer]
            user_recommendation = group_recs[group_recs[DEFAULT_GROUP_COL] == group]

            if self.history is not None:
                old_portfolio = self.history[self.history[DEFAULT_USER_COL] == customer]
                items_per_user = set(old_portfolio[DEFAULT_ITEM_COL].unique().flatten())
                user_recommendation = user_recommendation[~user_recommendation[DEFAULT_ITEM_COL].isin(items_per_user)]
            elif not repeated:
                items_per_user = set(self.data.train[self.data.train[DEFAULT_USER_COL] == customer][DEFAULT_ITEM_COL].unique().flatten())
                user_recommendation = user_recommendation[~user_recommendation[DEFAULT_ITEM_COL].isin(items_per_user)]

            user_recommendation = user_recommendation[[DEFAULT_ITEM_COL, DEFAULT_RATING_COL]]
            user_recommendation = user_recommendation.sort_values(by=DEFAULT_RATING_COL, ascending=False)
            user_recommendation[DEFAULT_USER_COL] = customer
            cust_recs.append(user_recommendation)

        return pd.concat(cust_recs)
