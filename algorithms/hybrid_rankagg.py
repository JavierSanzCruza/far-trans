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


class HybridRankAggregation(Algorithm):
    """
    Hybrid recommendation algorithm based on ranking aggregation.
    """
    def __init__(self, data, list_algorithms):
        """
        Configures the profitability prediction model.
        :param data: the data for training and applying recommendations.
        :param list_algorithms: the list of algorithms to consider.
        """
        super().__init__(data)

        self.assets_df = None
        self.list_algorithms = list_algorithms
        self.rec_df = None

    def train(self, train_date):

        recommendations = dict()

        # Load and normalize the recommendations:
        for algorithm in self.list_algorithms:
            alg_vals = []
            rec = pd.read_csv(algorithm)
            # Normalize each customer ranking -- we apply a min-max score:
            for customer in rec[DEFAULT_USER_COL].unique().flatten():
                customer_df = rec[rec[DEFAULT_USER_COL] == customer]
                customer_df = customer_df.sort_values(by=DEFAULT_RATING_COL, ascending=False)

                max = customer_df[DEFAULT_RATING_COL].max()
                min = customer_df[DEFAULT_RATING_COL].min()

                if max == min:
                    customer_df[DEFAULT_RATING_COL] = 0.5
                else:
                    customer_df[DEFAULT_RATING_COL] = customer_df[DEFAULT_RATING_COL].apply(lambda x: (x-min)/(max-min))
                alg_vals.append(customer_df)
            recommendations[algorithm] = pd.concat(alg_vals)
        # Aggregate the recommendations:

        initial_df = None
        for algorithm in self.list_algorithms:
            if initial_df is None:
                initial_df = recommendations[algorithm]
            else:
                initial_df = initial_df.merge(recommendations[algorithm], on=[DEFAULT_USER_COL, DEFAULT_ITEM_COL],
                                 how="outer", suffixes=("_x","_y"))
                initial_df.fillna(0.0)
                initial_df[DEFAULT_RATING_COL] = initial_df[DEFAULT_RATING_COL+"_x"] + initial_df[DEFAULT_RATING_COL+"_y"]
                initial_df = initial_df.drop(columns=[DEFAULT_RATING_COL+"_x", DEFAULT_RATING_COL+"_y"])

        self.rec_df = initial_df

    def recommend(self, rec_time, repeated, only_test_customers):
        customers = (self.data.users & set(self.data.test[DEFAULT_USER_COL].unique().flatten())) if only_test_customers else self.data.users

        user_recommendations = []
        for customer in customers:
            user_recommendation = self.rec_df[self.rec_df[DEFAULT_USER_COL] == customer]
            if not repeated:
                items_per_user = set(self.data.train[self.data.train[DEFAULT_USER_COL] == customer][DEFAULT_ITEM_COL].unique().flatten())
                user_recommendation = user_recommendation[~user_recommendation[DEFAULT_ITEM_COL].isin(items_per_user)]
            user_recommendation = user_recommendation[user_recommendation[DEFAULT_ITEM_COL].isin(self.data.assets)]
            user_recommendations.append(user_recommendation)
        return pd.concat(user_recommendations).sort_values(by=[DEFAULT_USER_COL, DEFAULT_RATING_COL], ascending=[True, False])
