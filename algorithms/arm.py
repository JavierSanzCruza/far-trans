#  Copyright (c) 2022. Terrier Team at University of Glasgow, http://http://terrierteam.dcs.gla.ac.uk
#
#  This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
#  If a copy of the MPL was not distributed with this  file, you can obtain one at
#  http://mozilla.org/MPL/2.0/.
#
#  This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
#  If a copy of the MPL was not distributed with this  file, you can obtain one at
#  http://mozilla.org/MPL/2.0/.

from operator import itemgetter

import pandas as pd
from beta_rec.utils.constants import (
    DEFAULT_ITEM_COL,
    DEFAULT_RATING_COL,
    DEFAULT_USER_COL,
)
from mlxtend.frequent_patterns import association_rules, apriori

from .algorithm import Algorithm


class ARMAlgorithm(Algorithm):
    """
    Implementation of the a-priori association-rule mining algorithm.
    This algorithm extracts associations between past and new investments to recommend new assets
    on which to invest.

    See:


    """

    def __init__(self, data):
        """
        Initializes the algorithm.
        :param data: the data to use.
        """
        super().__init__(data)
        self.ants = None
        self.cons = None
        self.lifts = None

    def train(self, train_date):
        basket = (self.data.train.groupby([DEFAULT_USER_COL, DEFAULT_ITEM_COL])[DEFAULT_RATING_COL]
                  .sum().unstack().reset_index().fillna(0).set_index(DEFAULT_USER_COL))

        basket_train = basket.applymap(lambda x: 1 if x > 0 else 0)
        frequent_itemsets_train = apriori(basket_train, min_support=0.007, use_colnames=True, max_len=2)
        rules = association_rules(frequent_itemsets_train, metric='lift').sort_values(by='lift',
                                                                                      ascending=False).reset_index()

        idx = rules['index'].values.tolist()

        self.ants = []
        self.cons = dict()
        self.lifts = dict()

        for i in idx:
            self.ants.append((i, list(rules['antecedents'][rules['index'] == i].values.tolist()[0])))
            self.cons[i] = list(rules['consequents'][rules['index'] == i].values.tolist()[0])
            self.lifts[i] = list(rules['lift'][rules['index'] == i])[0]

    def recommend(self, rec_date, repeated, only_test_customers):
        user_recommendations = []

        customers = (self.data.users & set(self.data.test[DEFAULT_USER_COL].unique().flatten())) if only_test_customers else self.data.users
        aux_cust = [x for x in customers]
        aux_assets = [x for x in self.data.assets]

        index = pd.MultiIndex.from_product([aux_cust, aux_assets], names=[DEFAULT_USER_COL, DEFAULT_ITEM_COL])
        test = pd.DataFrame(index=index).reset_index()

        for customer in customers:
            recs = []
            cats = self.data.train[DEFAULT_ITEM_COL][self.data.train[DEFAULT_USER_COL] == customer].values.tolist()
            for a in self.ants:
                if set(a[1]).issubset(set(cats)):
                    recs.append(a[0])

            if len(recs) > 0:
                recs = list(set(recs))
                newlifts = []

                for item_id in recs:
                    newlifts.append(self.lifts[item_id])
                crl = zip(recs, newlifts)
                cust_recs_with_lifts = sorted(list(crl), key=itemgetter(1), reverse=True)
                records = []
                for cl in cust_recs_with_lifts:
                    con_names = self.cons[cl[0]]
                    for con_name in con_names:
                        asset_name = con_name
                        r = [customer, asset_name, cl[1]]
                        records.append(r)

                user_recommendation = pd.DataFrame(records, columns=[DEFAULT_USER_COL, DEFAULT_ITEM_COL, DEFAULT_RATING_COL])
                user_recommendation = user_recommendation.groupby([DEFAULT_USER_COL, DEFAULT_ITEM_COL], as_index=False, sort=False)[
                    DEFAULT_RATING_COL].sum()

                if not repeated:
                    items_per_user = set(self.data.train[self.data.train[DEFAULT_USER_COL] == customer][
                                             DEFAULT_ITEM_COL].unique().flatten())
                    user_recommendation = user_recommendation[~user_recommendation[DEFAULT_ITEM_COL].isin(items_per_user)]
                user_recommendation = user_recommendation.sort_values(by=DEFAULT_RATING_COL, ascending=False)
                user_recommendation = user_recommendation[user_recommendation[DEFAULT_ITEM_COL].isin(self.data.assets)]
                user_recommendations.append(user_recommendation)
        def_recs = pd.concat(user_recommendations)
        def_recs = pd.merge(test, def_recs, on=[DEFAULT_USER_COL, DEFAULT_ITEM_COL], how="left")
        def_recs = def_recs.fillna(0.0)
        def_recs = def_recs.sort_values(by=[DEFAULT_USER_COL, DEFAULT_RATING_COL], ascending=[True, False])
        return def_recs
