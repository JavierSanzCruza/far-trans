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
from beta_rec.utils.constants import (
    DEFAULT_ITEM_COL,
    DEFAULT_RATING_COL,
    DEFAULT_TIMESTAMP_COL,
    DEFAULT_USER_COL,
)

from .algorithm import Algorithm


class CPSAlgorithm(Algorithm):
    """
    Implementation of a demographic user-based kNN algorithm (customer profile similarity-based)
    """
    def __init__(self, data, user_profile_data, k, norm):
        """
        Initializes the algorithm.
        :param data: the
        :param user_profile_data:
        :param k:
        :param norm:
        """
        super().__init__(data)
        self.modules = None
        self.k = k
        self.norm = norm

        names = [DEFAULT_USER_COL, DEFAULT_TIMESTAMP_COL]
        for i in range(0, 25):
            names.append("q" + str(i))
        names.append("prevDate")

        self.user_profile_data = pd.read_csv(
            user_profile_data,
            skiprows=[0],
            engine="python",
            names=names
        )
        self.user_profile_data[DEFAULT_TIMESTAMP_COL] = pd.to_datetime(self.user_profile_data[DEFAULT_TIMESTAMP_COL],
                                                                       format='%Y-%m-%d')
        self.user_profile_data = self.user_profile_data[self.user_profile_data[DEFAULT_USER_COL].isin(self.data.users)]

    def train(self, train_date):
        names = []
        for i in range(0, 25):
            names.append("q" + str(i))
        self.user_profile_data["module"] = self.user_profile_data.drop(columns=[DEFAULT_USER_COL, DEFAULT_TIMESTAMP_COL, "prevDate"]).pow(2.0).sum(axis=1)
        self.user_profile_data = self.user_profile_data.sort_values(by=[DEFAULT_USER_COL, DEFAULT_TIMESTAMP_COL], ascending=[True, True])
        return

    def recommend(self, rec_date, repeated, only_test_customers):

        customers = (self.data.users & set(self.data.test[DEFAULT_USER_COL].unique().flatten())) if only_test_customers else self.data.users

        aux_cust = [x for x in customers]
        aux_assets = [x for x in self.data.assets]
        index = pd.MultiIndex.from_product([aux_cust, aux_assets], names=[DEFAULT_USER_COL, DEFAULT_ITEM_COL])
        test = pd.DataFrame(index=index).reset_index()

        names = []
        for i in range(0, 25):
            names.append("q" + str(i))

        user_profiles = self.user_profile_data[self.user_profile_data[DEFAULT_TIMESTAMP_COL] < rec_date]\
                            .drop_duplicates(subset=[DEFAULT_USER_COL], keep="last")
        mods = dict()
        vectors = dict()
        for index, row in user_profiles.iterrows():
            mods[row[DEFAULT_USER_COL]] = math.sqrt(row["module"])
            vector = []
            for name in names:
                vector.append(row[name])
            vectors[row[DEFAULT_USER_COL]] = vector


        user_recs = []
        for u in aux_cust:
            sim_dict = dict()

            if u not in vectors:
                continue
            for v in vectors:
                if u == v:
                    continue
                dot_prod = 0.0
                for i in range(0, 25):
                    dot_prod += vectors[u][i]*vectors[v][i]
                sim_dict[v] = dot_prod/(mods[u]*mods[v])

            sim_df = pd.DataFrame(sim_dict.items(), columns=["v", "sim"])
            sim_df = sim_df.sort_values(by=["sim"], ascending=False).head(self.k)

            if self.norm:
                item_dict = dict()
                item_norm = dict()
                for index, v in sim_df.iterrows():
                    for index2, j in self.data.train[self.data.train[DEFAULT_USER_COL] == v["v"]].iterrows():
                        j_item = j[DEFAULT_ITEM_COL]
                        if j_item in item_dict:
                            item_dict[j_item] = item_dict[j_item] + v["sim"]*j[DEFAULT_RATING_COL]
                        else:
                            item_dict[j_item] = v["sim"]*j[DEFAULT_RATING_COL]

                        if j_item not in item_norm:
                            item_norm[j_item] = abs(v["sim"])
                        else:
                            item_norm[j_item] = item_norm[j_item] + abs(v["sim"])
                for j_item in item_dict:
                    if item_norm[j_item] > 0.0:
                        item_dict[j_item] = item_dict[j_item] / item_norm[j_item]
                    else:
                        item_dict[j_item] = 0.0
                item_df = pd.DataFrame(item_dict.items(), columns=[DEFAULT_ITEM_COL, DEFAULT_RATING_COL])
                item_df[DEFAULT_USER_COL] = u
            else:
                item_dict = dict()
                for index, v in sim_df.iterrows():
                    for index2, j in self.data.train[self.data.train[DEFAULT_USER_COL] == v["v"]].iterrows():
                        j_item = j[DEFAULT_ITEM_COL]
                        if j_item in item_dict:
                            item_dict[j_item] = item_dict[j_item] + v["sim"] * j[DEFAULT_RATING_COL]
                        else:
                            item_dict[j_item] = v["sim"] * j[DEFAULT_RATING_COL]
                item_df = pd.DataFrame(item_dict.items(), columns=[DEFAULT_ITEM_COL, DEFAULT_RATING_COL])
                item_df[DEFAULT_USER_COL] = u

            if not repeated:
                items_per_user = set(self.data.train[self.data.train[DEFAULT_USER_COL] == u][
                                     DEFAULT_ITEM_COL].unique().flatten())
                item_df = item_df[~item_df[DEFAULT_ITEM_COL].isin(items_per_user)]
            item_df = item_df.sort_values(by=DEFAULT_RATING_COL, ascending=False)
            item_df = item_df[item_df[DEFAULT_ITEM_COL].isin(self.data.assets)]
            user_recs.append(item_df)

        def_recs = pd.concat(user_recs)
        def_recs = pd.merge(test, def_recs, on=[DEFAULT_USER_COL, DEFAULT_ITEM_COL], how="left")
        def_recs = def_recs.fillna(0.0)
        def_recs = def_recs.sort_values(by=[DEFAULT_USER_COL, DEFAULT_RATING_COL], ascending=[True, False])
        return def_recs
