#  Copyright (c) 2022. Terrier Team at University of Glasgow, http://http://terrierteam.dcs.gla.ac.uk
#
#  This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
#  If a copy of the MPL was not distributed with this  file, you can obtain one at
#  http://mozilla.org/MPL/2.0/.
#
#  This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
#  If a copy of the MPL was not distributed with this  file, you can obtain one at
#  http://mozilla.org/MPL/2.0/.

import os

import lightgbm
import pandas as pd
from beta_rec.utils.constants import DEFAULT_ITEM_COL, DEFAULT_USER_COL, DEFAULT_RATING_COL, DEFAULT_TIMESTAMP_COL

from algorithms.algorithm import Algorithm


def compute_target(x):
    """
    Relevance degree for assets based in profitability. This are
    computed if and only if the asset is purchased by the user.
    If asset is not profitable, rel(u,i) = 0
    If asset is profitable at less than 10% ROI, rel(u,i) = 1
    If asset is profitable between 10% and 25% ROI, rel(u,i) = 2
    If asset is profitable between 25% and 50% ROI, rel(u,i) = 3
    If asset is profitable between 50% and 100% ROI, rel(u,i) = 4
    If asset is profitable by more than 100%, rel(u,i) = 5
    :param x: the profitability.
    :return: the degree of relevance.
    """
    if x <= 0:
        return 0
    elif x <= 0.1:
        return 1
    elif x <= 0.25:
        return 2
    elif x <= 0.5:
        return 3
    elif x <= 1.0:
        return 4
    else:
        return 5


class HybridLambdaMART(Algorithm):
    """
    Implementation of a LambdaMART hybrid algorithm using LightGBM.
    """
    def __init__(self, data, prev_data, list_algorithms, prev_date, mode, repeated, only_test_custs):
        """
        Configures the profitability prediction model.
        :param data: the data for training and applying recommendations.
        :param list_algorithms: the list of algorithms to use as features.
        :param prev_date: the date at which we want to generate features (recommendations) from previos algorithms.
        :param mode: three modes: regression (optimize for ROI), ndcg (learning to rank, optimize for nDCG),
                                  prof-ndcg (optimize for a nDCG variant which takes profitability as relevance
                                  degree for those assets acquired by the user).
        :param data: the data for training and applying recommendations.
        :param repeated: if we want to consider assets already purchased by the user as recommendable.
        :param only_test_custs: if we want only to consider test customers for generating recommendations.
        """
        super().__init__(data)

        self.prev_data = prev_data
        self.list_algorithms = list_algorithms
        self.prev_date = prev_date
        self.model = None
        print(len(list_algorithms))
        for name in list_algorithms:
            print(name)
        self.models = None
        self.mode = mode
        self.repeated = repeated
        self.only_test_custs = only_test_custs

    def train(self, train_date):
        self.models = []
        recommendations = dict()
        for name, algorithm in self.list_algorithms.items():
            if os.path.exists(name + "_" + self.prev_date.strftime("%Y-%m-%d") + "_recs.txt"):
                print(name + "_" + self.prev_date.strftime("%Y-%m-%d") + "_recs.txt")
                rec = pd.read_csv(name + "_" + self.prev_date.strftime("%Y-%m-%d") + "_recs.txt")
            else:
                algorithm[0].train(self.prev_date)
                rec = algorithm[0].recommend(self.prev_date, self.repeated, self.only_test_custs)
                rec.to_csv(name + "_" + self.prev_date.strftime("%Y-%m-%d") + "_recs.txt")
            print(name)
            alg_vals = []
            # Normalize each customer ranking -- we apply a ranksim score:
            for customer in rec[DEFAULT_USER_COL].unique().flatten():
                customer_df = rec[rec[DEFAULT_USER_COL] == customer]
                customer_df = customer_df.sort_values(by=DEFAULT_RATING_COL, ascending=False)

                max = customer_df[DEFAULT_RATING_COL].max()
                min = customer_df[DEFAULT_RATING_COL].min()

                if max == min:
                    customer_df[DEFAULT_RATING_COL] = 0.5
                else:
                    customer_df[DEFAULT_RATING_COL] = customer_df[DEFAULT_RATING_COL].apply(
                        lambda x: (x - min) / (max - min))
                alg_vals.append(customer_df)
            aux_df = pd.concat(alg_vals)
            recommendations[name] = aux_df.rename(columns={DEFAULT_RATING_COL: name})
            self.models.append(name)

        # Now, we generate a combined model
        combined = None
        for alg in recommendations:
            if combined is None:
                combined = recommendations[alg][[DEFAULT_USER_COL, DEFAULT_ITEM_COL, alg]].copy()
            else:
                combined = pd.merge(combined, recommendations[alg], on=[DEFAULT_USER_COL, DEFAULT_ITEM_COL],
                                    how="outer")
                combined = combined.fillna(0.0)

        prices = self.data.time_series[self.data.time_series[DEFAULT_TIMESTAMP_COL] == self.prev_date][
            [DEFAULT_ITEM_COL, DEFAULT_RATING_COL]]
        prices_new = self.data.time_series[self.data.time_series[DEFAULT_TIMESTAMP_COL] == train_date][
            [DEFAULT_ITEM_COL, DEFAULT_RATING_COL]]

        prices = pd.merge(prices, prices_new, on=[DEFAULT_ITEM_COL], suffixes=("_old", "_new"), how="inner")
        prices["target"] = (prices[DEFAULT_RATING_COL + "_new"] - prices[DEFAULT_RATING_COL + "_old"]) / prices[
            DEFAULT_RATING_COL + "_old"]

        prices = prices[[DEFAULT_ITEM_COL, "target"]]

        if self.mode == "regression":
            prices["target"] = prices["target"]
        elif self.mode == "prof-ndcg":
            prices["target"] = prices["target"].apply(lambda x: compute_target(x))
        else:
            prices["target"] = prices["target"].apply(lambda x: 1.0)

        combined = combined.merge(prices, on=[DEFAULT_ITEM_COL], how="outer")
        combined = combined.dropna()

        print(combined)

        if self.mode != "regression":
            # In this case, we need to know the nDCG -- so we need to check whether the assets are or not in the test set
            positive = self.prev_data.test[self.prev_data.test[DEFAULT_RATING_COL] == 1.0]
            combined = combined.merge(positive, on=[DEFAULT_USER_COL, DEFAULT_ITEM_COL], how="left")
            combined = combined.fillna(0.0)
            combined["target"] = combined["target"] * combined[DEFAULT_RATING_COL]

            print(combined["target"].unique())
            combined = combined.drop(columns=DEFAULT_RATING_COL)

        combined = combined.sort_values(by=[DEFAULT_USER_COL, DEFAULT_ITEM_COL], ascending=[True, True])
        print(combined)
        groups = combined.groupby([DEFAULT_USER_COL])[DEFAULT_ITEM_COL].count().values
        targets = combined["target"]
        examples = combined[self.models]

        if self.mode != "regression":
            self.model = lightgbm.LGBMRanker(task="train",
                                             min_data_in_leaf=1,
                                             min_sum_hessian_in_leaf=100,
                                             max_bin=255,
                                             num_leaves=7,
                                             objective="lambdarank",
                                             metric="ndcg",
                                             ndcg_eval_at=[1, 3, 5, 10],
                                             learning_rate=.1,
                                             importance_type="gain",
                                             num_iterations=10)
            self.model.fit(examples, targets, group=groups)
        else:
            self.model = lightgbm.LGBMRegressor()
            self.model.fit(examples, targets)

    def recommend(self, rec_time, repeated, only_test_customers):
        # As a first step, we read the recommendations:
        recommendations = dict()
        for name, algorithm in self.list_algorithms.items():
            def_name = name + "_" + rec_time.strftime("%Y-%m-%d") + "_recs.txt"
            rec = pd.read_csv(def_name)
            alg_vals = []
            # Normalize each customer ranking -- we apply a min-max score:
            for customer in rec[DEFAULT_USER_COL].unique().flatten():
                customer_df = rec[rec[DEFAULT_USER_COL] == customer]
                customer_df = customer_df.sort_values(by=DEFAULT_RATING_COL, ascending=False)

                max = customer_df[DEFAULT_RATING_COL].max()
                min = customer_df[DEFAULT_RATING_COL].min()

                if max == min:
                    customer_df[DEFAULT_RATING_COL] = 0.5
                else:
                    customer_df[DEFAULT_RATING_COL] = customer_df[DEFAULT_RATING_COL].apply(
                        lambda x: (x - min) / (max - min))
                alg_vals.append(customer_df)
            recommendations[name] = pd.concat(alg_vals)
            recommendations[name] = recommendations[name].rename(columns={DEFAULT_RATING_COL: name})

        # Now, we generate a combined model
        combined = None
        for alg in recommendations:
            if combined is None:
                combined = recommendations[alg].copy()
            else:
                combined = pd.merge(combined, recommendations[alg], on=[DEFAULT_USER_COL, DEFAULT_ITEM_COL],
                                    how="outer")
                combined.fillna(0.0)

        # And we generate the recommendations
        user_recommendations = []
        for customer in combined[DEFAULT_USER_COL].unique().flatten():
            customer_df = combined[combined[DEFAULT_USER_COL] == customer]
            customer_df[DEFAULT_RATING_COL] = self.model.predict(customer_df[self.models])
            user_recommendation = customer_df[[DEFAULT_USER_COL, DEFAULT_ITEM_COL, DEFAULT_RATING_COL]]
            if not repeated:
                items_per_user = set(
                    self.data.train[self.data.train[DEFAULT_USER_COL] == customer][DEFAULT_ITEM_COL].unique().flatten())
                user_recommendation = user_recommendation[~user_recommendation[DEFAULT_ITEM_COL].isin(items_per_user)]
            user_recommendation = user_recommendation[user_recommendation[DEFAULT_ITEM_COL].isin(self.data.assets)]
            user_recommendations.append(user_recommendation)
        return pd.concat(user_recommendations).sort_values(by=[DEFAULT_USER_COL, DEFAULT_RATING_COL], ascending=[True, False])
