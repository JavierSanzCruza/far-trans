#  Copyright (c) 2022. Terrier Team at University of Glasgow, http://http://terrierteam.dcs.gla.ac.uk
#
#  This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
#  If a copy of the MPL was not distributed with this  file, you can obtain one at
#  http://mozilla.org/MPL/2.0/.
#  Copyright (c) 2022. Terrier Team at University of Glasgow, http://http://terrierteam.dcs.gla.ac.uk
#
#  This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
#  If a copy of the MPL was not distributed with this  file, you can obtain one at
#  http://mozilla.org/MPL/2.0/.
#
#  This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
#  If a copy of the MPL was not distributed with this  file, you can obtain one at
#  http://mozilla.org/MPL/2.0/.

import math

import pandas as pd
from beta_rec.utils.constants import DEFAULT_TIMESTAMP_COL, DEFAULT_ITEM_COL, DEFAULT_USER_COL, DEFAULT_RATING_COL

from algorithms.algorithm import Algorithm


class RankingBasedAlgorithm(Algorithm):
    """
    Algorithm that ranks assets according to the value of an individual technical indicator.
    """
    def __init__(self, data, kpi):
        """
        Configures the profitability prediction model.
        :param data: the data for training and applying recommendations.
        :param kpi: the name of the technical indicator.
        """
        super().__init__(data)

        self.assets_df = None
        self.kpi = kpi

    def train(self, train_date):

        kpis = self.data.kpis
        kpis = kpis[[DEFAULT_ITEM_COL, DEFAULT_TIMESTAMP_COL, self.kpi]]
        kpis = kpis[kpis[DEFAULT_TIMESTAMP_COL] < train_date]
        kpis = kpis[kpis[DEFAULT_ITEM_COL].isin(self.data.assets)]
        kpis = kpis.dropna()

        asset_list = list(self.data.assets)
        initial_vals = [0.0 for x in asset_list]

        val_pd = pd.DataFrame({DEFAULT_ITEM_COL: asset_list, DEFAULT_RATING_COL: initial_vals, "count": initial_vals})

        for date in kpis[DEFAULT_TIMESTAMP_COL].unique():
            aux_kpis = kpis[kpis[DEFAULT_TIMESTAMP_COL] == date].sort_values(by=self.kpi, ascending = False)
            aux_kpis = aux_kpis.reset_index(drop=True, inplace=False)
            aux_kpis = aux_kpis.reset_index(drop=False, inplace=False)
            aux_kpis = aux_kpis[[DEFAULT_ITEM_COL, self.kpi, "index"]]
            print(val_pd)
            print(aux_kpis)
            val_pd = pd.merge(val_pd, aux_kpis, how="left", on=DEFAULT_ITEM_COL)
            print(val_pd)
            val_pd["count"] += val_pd["index"].apply(lambda x: 1.0 if not math.isnan(x) else 0.0)
            aux_val = val_pd["index"].apply(lambda x: 0.0 if math.isnan(x) else (x+1))
            val_pd[DEFAULT_RATING_COL] = aux_val + (val_pd[DEFAULT_RATING_COL] - aux_val)/(val_pd["count"].replace(0.0, 1.0))
            val_pd[DEFAULT_RATING_COL] += aux_val*val_pd["count"].apply(lambda x: 1.0 if x == 1.0 else 0.0)
            val_pd = val_pd[[DEFAULT_ITEM_COL, DEFAULT_RATING_COL, "count"]]

        val_pd[DEFAULT_RATING_COL] = val_pd[DEFAULT_RATING_COL].apply(lambda x: len(asset_list) if x == 0.0 else x)
        self.assets_df = val_pd.sort_values(by=DEFAULT_RATING_COL, ascending=True)
        self.assets_df = self.assets_df[[DEFAULT_ITEM_COL, DEFAULT_RATING_COL]]

    def recommend(self, rec_time, repeated, only_test_customers):
        """
        Generates the recommendation.
        :param rec_time: the recommendation time.
        :return: the produced recommendations.
        """

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
