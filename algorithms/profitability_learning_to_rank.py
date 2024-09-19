#  Copyright (c) 2022. Terrier Team at University of Glasgow, http://http://terrierteam.dcs.gla.ac.uk
#
#  This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
#  If a copy of the MPL was not distributed with this  file, you can obtain one at
#  http://mozilla.org/MPL/2.0/.
#
#  This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
#  If a copy of the MPL was not distributed with this  file, you can obtain one at
#  http://mozilla.org/MPL/2.0/.

import datetime
import math
import random

import lightgbm as lgb
import pandas as pd
from beta_rec.utils.constants import DEFAULT_TIMESTAMP_COL, DEFAULT_ITEM_COL, DEFAULT_USER_COL, DEFAULT_RATING_COL

from algorithms.algorithm import Algorithm


class ProfitabilityLETOR(Algorithm):
    """
    Algorithm for ranking assets according to their profitability. Differently from the profitability
    estimation methods, this algorithm tries to learn the ranking of assets according to the technical
    indicator vectors -- regardless of the actual profitability value.
    """
    def __init__(self, data, months, indicators, train_examples_per_asset, num_samples, num_divisions):
        """
        Configures the profitability prediction model.
        :param data: the data for training and applying recommendations.
        :param months: how many months in the future we want to apply our model to.
        :param indicators: the technical indicators we want to use from the set of computed ones.
        :param train_examples_per_asset: the maximum number of examples per asset to use. 0 if we don't want to limit.
        :param num_samples: number of negative samples
        :param num_divisions:
        """
        super().__init__(data)

        self.kpis = data.kpis
        self.months = months
        self.indicators = indicators
        self.model = None
        self.train_examples_per_asset = train_examples_per_asset
        self.num_samples = num_samples
        self.num_divisions = num_divisions

    def train(self, train_date):
        # As a first step, we need to find the training examples. In order to do this, the first steps just finds the
        # targets.
        kpi_indicators = self.kpis[self.kpis[DEFAULT_ITEM_COL].isin(self.data.assets)]
        delta = datetime.timedelta(days=self.months*30)
        #kpi_indicators = kpi_indicators[kpi_indicators[DEFAULT_TIMESTAMP_COL] < (train_date - delta)]

        # For each asset, we get the target (profitability at k months)
        asset_dfs = []
        for asset in kpi_indicators[DEFAULT_ITEM_COL].unique():
            asset_df = kpi_indicators[kpi_indicators[DEFAULT_ITEM_COL] == asset]
            # We first find the profitability
            asset_df["final_price"] = asset_df[DEFAULT_RATING_COL].shift(-self.months * 21)
            asset_df["final_price"] = (asset_df["final_price"] - asset_df[DEFAULT_RATING_COL]) / (
                asset_df[DEFAULT_RATING_COL])
            # Then, we find the actual target:
            asset_df["target"] = asset_df["final_price"].apply(lambda x: 0 if x <= 0.0 else (int(math.floor(4*x + 1)) if x < 2.0 else 9))
            asset_df = asset_df.dropna()
            asset_df = asset_df[asset_df[DEFAULT_RATING_COL] > 0.0]
            asset_dfs.append(asset_df)
        kpi_indicators = pd.concat(asset_dfs)

        kpi_indicators = kpi_indicators[kpi_indicators[DEFAULT_TIMESTAMP_COL] < (train_date - delta)]
        if self.train_examples_per_asset > 0:
            past_delta = datetime.timedelta(days=self.train_examples_per_asset)
            kpi_indicators = kpi_indicators[kpi_indicators[DEFAULT_TIMESTAMP_COL] >= (train_date - delta - past_delta)]

        # Now, we follow the sampling procedure: first, we collect the dates in the training set:
        dates = kpi_indicators[DEFAULT_TIMESTAMP_COL].unique().flatten()

        groups = []
        query_dfs = []

        # Then, for each date:
        for date in dates:
            # We first get the indicators:
            date_df = kpi_indicators[kpi_indicators[DEFAULT_TIMESTAMP_COL] == date]

            # Divide in positive and negative examples
            positive_df = date_df[date_df["target"] > 0]
            negative_df = date_df[date_df["target"] == 0]

            # Then, we generate self.numsamples partitions of the assets in self.num_divisions categories.
            for i in range(0, self.num_samples):

                sample_pos_df = positive_df.copy()
                sample_neg_df = negative_df.copy()

                # Generate the partition, by randomly assigning categories to the positive / negative examples:
                sample_pos_df["cat"] = sample_pos_df["target"].apply(lambda x: random.randint(0, self.num_divisions))
                sample_neg_df["cat"] = sample_neg_df["target"].apply(lambda x: random.randint(0, self.num_divisions))

                for j in range(0, self.num_divisions):
                    # we find the corresponding partition and concatenate the positive and negative samples. Each
                    # partition would make the documents for the "query".
                    pos_query_df = sample_pos_df[sample_pos_df["cat"] == j]
                    neg_query_df = sample_neg_df[sample_neg_df["cat"] == j]
                    query_df = pd.concat([pos_query_df, neg_query_df])
                    query_df = query_df.drop(columns="cat")
                    query_dfs.append(query_df)
                    groups.append(query_df.shape[0])
        training_df = pd.concat(query_dfs)
        training_df.reset_index(drop=True, inplace=True)

        goals = training_df["target"]
        training_df = training_df[self.indicators]

        print("MAX: " + str(max(goals)))
        self.model = lgb.LGBMRanker(task="train",
                                    min_data_in_leaf=1,
                                    min_sum_hessian_in_leaf=100,
                                    max_bin=255,
                                    num_leaves=7,
                                    objective="lambdarank",
                                    metric="ndcg",
                                    ndcg_eval_at=[1, 5, 10],
                                    learning_rate=.1,
                                    label_gain=[i for i in range(max(goals)+1)],
                                    num_iterations=10)
        self.model.fit(training_df, goals, group=groups)

    def recommend(self, rec_time, repeated, only_test_customers):
        """
        Generates the recommendation.
        :param rec_time: the recommendation time.
        :return: the produced recommendations.
        """
        fields = [x for x in self.indicators]
        fields.append(DEFAULT_ITEM_COL)
        fields.append(DEFAULT_TIMESTAMP_COL)

        # We first obtain the KPIs
        kpi_indicators = self.kpis[fields]

        kpi_indicators = kpi_indicators[kpi_indicators[DEFAULT_TIMESTAMP_COL] == rec_time]
        kpi_indicators = kpi_indicators[kpi_indicators[DEFAULT_ITEM_COL].isin(self.data.assets)]

        # Then, we obtain the recommendation scores:
        kpi_indicators["score"] = self.model.predict(kpi_indicators[self.indicators])
        kpi_indicators = kpi_indicators[[DEFAULT_ITEM_COL, "score"]].sort_values(by="score", ascending=False)
        kpi_indicators = kpi_indicators.rename(columns={"score": DEFAULT_RATING_COL})

        user_recommendations = []
        customers = (self.data.users & set(self.data.test[DEFAULT_USER_COL].unique().flatten())) if only_test_customers else self.data.users

        for customer in customers:
            user_recommendation = kpi_indicators.copy()
            user_recommendation[DEFAULT_USER_COL] = customer

            if not repeated:
                items_per_user = set(
                    self.data.train[self.data.train[DEFAULT_USER_COL] == customer][DEFAULT_ITEM_COL].unique().flatten())
                user_recommendation = user_recommendation[~user_recommendation[DEFAULT_ITEM_COL].isin(items_per_user)]
            user_recommendations.append(user_recommendation)
        return pd.concat(user_recommendations)
