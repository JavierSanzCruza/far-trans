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
    DEFAULT_USER_COL, )

from metrics.kpi_evaluation_metric import KPIEvaluationMetric


class KPINDCG(KPIEvaluationMetric):
    """
    Class for computing the profitability of an asset in the future.
    """
    def __init__(self, data, values, metric_threshold=0.0, history=None):
        """
        Initializes the value of the metric.
        :param data: the complete data.
        """
        super().__init__(data, values)
        self.metric_threshold = metric_threshold
        self.asset_ranking = None
        self.history = history

    def evaluate_cutoffs(self, recs, cutoffs, target_custs, only_test_customers):
        # Step 1: We generate a ranking of assets to assist.
        list_vals = [(k, v) for k, v in self.values.items()]
        vals_df = pd.DataFrame(list_vals, columns=[DEFAULT_ITEM_COL, "metric"])
        vals_df = vals_df[vals_df["metric"] > self.metric_threshold]
        vals_df.sort_values(by=["metric"], ascending=False, inplace=True)
        vals_df.reset_index(drop=True, inplace=True)
        self.asset_ranking = vals_df

        # Step 2:
        customers = self.data.users & set(
            self.data.test[DEFAULT_USER_COL].unique().flatten()) if only_test_customers else self.data.users
        customers = customers & target_custs

        num_customers = len(customers)

        aux_cutoffs = [x for x in cutoffs]
        aux_cutoffs.sort(reverse=True)

        aux_recs = recs[recs[DEFAULT_USER_COL].isin(customers)]
        aux_recs = aux_recs.groupby(DEFAULT_USER_COL).head(aux_cutoffs[0])
        aux_recs["metric"] = aux_recs[DEFAULT_ITEM_COL].apply(lambda x: self.values[x])

        # Step 3: Compute the individual metrics
        cust_evals = dict()
        gen_evals = dict()

        for cutoff in cutoffs:
            cust_evals[cutoff] = []
            gen_evals[cutoff] = 0.0

        for customer in customers:
            customer_df = aux_recs[aux_recs[DEFAULT_USER_COL] == customer]
            if customer_df.shape[0] == 0:
                for cutoff in aux_cutoffs:
                    cust_evals[cutoff].append((customer, 0.0))
            else:
                aux_dict = self.evaluate_indiv_cutoffs(customer_df, aux_cutoffs)
                for cutoff in cutoffs:
                    cust_evals[cutoff].append((customer, aux_dict[cutoff]))
                    gen_evals[cutoff] += (aux_dict[cutoff]) / (num_customers + 0.0)

        def_dict = dict()
        for cutoff in cutoffs:
            def_dict[cutoff] = (pd.DataFrame(cust_evals[cutoff], columns=[DEFAULT_USER_COL, "metric"]), gen_evals[cutoff])
        return def_dict

    def evaluate_indiv_cutoffs(self, customer_df, cutoffs):
        customer = customer_df[DEFAULT_USER_COL].unique()[0]


        current_assets = set()
        if self.history is not None:
            current_pf = self.history[self.history[DEFAULT_USER_COL]==customer]
            if current_pf.shape[0] > 0:
                current_assets = set(current_pf[DEFAULT_ITEM_COL].unique())

        aux_cutoffs = [x for x in cutoffs]
        aux_cutoffs.sort()

        res_dict = dict()

        positive_assets = self.data.get_positive_assets(customer) - current_assets
        if len(positive_assets) == 0:
            for cutoff in cutoffs:
                res_dict[cutoff] = 0
            return res_dict
        else:
            idcg_df = self.asset_ranking[self.asset_ranking[DEFAULT_ITEM_COL].isin(positive_assets)]

            items_that_add = positive_assets & {k for k, v in self.values.items() if v > self.metric_threshold}

            if len(items_that_add) == 0:
                for cutoff in cutoffs:
                    res_dict[cutoff] = 0
                return res_dict

            # We first compute the idcg values:
            idcgs = dict()

            i = 0
            k = 0
            current_cutoff = aux_cutoffs[i]
            idcg = 0.0
            for index, row in idcg_df.iterrows():
                idcg += row["metric"]/math.log(k + 2.0)
                k += 1
                if k == current_cutoff:
                    idcgs[k] = idcg
                    if current_cutoff == aux_cutoffs[-1]:
                        current_cutoff = -1
                    else:
                        i += 1
                        current_cutoff = aux_cutoffs[i]
            if current_cutoff != -1:
                for j in range(i, len(cutoffs)):
                    idcgs[aux_cutoffs[j]] = idcg

            dcg = 0.0
            i = 0
            k = 0
            max_cutoff = aux_cutoffs[-1]
            current_cutoff = aux_cutoffs[0]

            aux_cust_df = customer_df.head(max_cutoff)
            for index, row in aux_cust_df.iterrows():
                val = row["metric"] if row[DEFAULT_ITEM_COL] in items_that_add else 0.0
                dcg += val / math.log(k + 2.0)
                k += 1
                if k == current_cutoff:
                    res_dict[current_cutoff] = dcg / idcgs[current_cutoff]

                    if current_cutoff == max_cutoff:
                        current_cutoff = -1
                    else:
                        i += 1
                        current_cutoff = aux_cutoffs[i]

            # In case there is not enough recommended assets, we consider only
            # those in the ranking.
            if current_cutoff != -1:
                for j in range(i, len(cutoffs)):
                    res_dict[aux_cutoffs[j]] = dcg / idcgs[aux_cutoffs[j]]

        return res_dict

    def evaluate_indiv(self, customer_df, cutoff):
        customer = customer_df[DEFAULT_USER_COL].unique()[0]

        # user profitability:
        user_profit = dict()
        user_profit_list = []

        current_assets = set()
        if self.history is not None:
            current_pf = self.history[self.history[DEFAULT_USER_COL] == customer]
            if current_pf.shape[0] > 0:
                current_assets = set(current_pf[DEFAULT_ITEM_COL].unique())

        positive_assets = self.data.get_positive_assets(customer) - current_assets
        for asset in positive_assets:
            if asset not in self.values:
                continue
            else:
                val = self.values[asset] if self.values[asset] > self.metric_threshold else 0.0
            user_profit[asset] = val
            user_profit_list.append(val)
        user_profit_list = sorted(user_profit_list, reverse=True)

        idcg = 0.0
        for k in range(0, min(cutoff, len(user_profit_list))):
            idcg += user_profit_list[k]/math.log(k+2.0)

        if idcg == 0.0:
            return 0.0

        k = 0
        dcg = 0.0
        for index, row in customer_df.iterrows():
            asset = row[DEFAULT_ITEM_COL]
            if asset in positive_assets and asset in self.values:
                val = self.values[asset] if self.values[asset] > self.metric_threshold else 0.0
                dcg += val / math.log(k+2.0)
            k += 1

        return dcg/idcg

