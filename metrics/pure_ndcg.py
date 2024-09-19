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
    DEFAULT_USER_COL,
)

from metrics.metric import Metric


class PureNDCG(Metric):
    """
    Class for computing the profitability of an asset in the future.
    """

    def __init__(self, data, history=None):
        """
        Initializes the value of the metric.
        :param data: the complete data.
        :param rec_date: the date the recommendation is produced.
        :param max_date: the maximum date to consider (here, the future date for computing the profitability).
        """
        super().__init__(data)
        self.idcgs = dict()
        self.idcgs[0] = 1.0
        self.history = history

    def evaluate_cutoffs(self, recs, cutoffs, target_custs, only_test_customers):
        aux_cutoffs = [x for x in cutoffs]
        aux_cutoffs.sort(reverse=True)

        idcg = 0.0
        for i in range(0, aux_cutoffs[0]):
            idcg += 1.0 / math.log(i + 2.0)
            self.idcgs[i+1] = idcg

        customers = self.data.users & set(
            self.data.test[DEFAULT_USER_COL].unique().flatten()) if only_test_customers else self.data.users
        customers = customers & target_custs

        num_customers = len(customers)

        aux_recs = recs[recs[DEFAULT_USER_COL].isin(customers)]
        aux_recs = aux_recs.groupby(DEFAULT_USER_COL).head(aux_cutoffs[0])

        cust_evals = dict()
        gen_evals = dict()

        for cutoff in cutoffs:
            cust_evals[cutoff] = []
            gen_evals[cutoff] = 0.0

        for customer in customers:
            customer_df = aux_recs[aux_recs[DEFAULT_USER_COL] == customer]
            if customer_df.shape[0] == 0:
                for cutoff in cutoffs:
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

        aux_cutoffs = [x for x in cutoffs]
        aux_cutoffs.sort()

        res_dict = dict()

        current_assets = set()
        if self.history is not None:
            current_pf = self.history[self.history[DEFAULT_USER_COL] == customer]
            if current_pf.shape[0] > 0:
                current_assets = set(current_pf[DEFAULT_ITEM_COL].unique())

        positive_assets = self.data.get_positive_assets(customer) - current_assets
        if len(positive_assets) == 0:
            for cutoff in cutoffs:
                res_dict[cutoff] = 0.0
            return res_dict
        else:
            cutoff_idcgs = dict()
            ## We compute the particular idcgs:
            for cutoff in aux_cutoffs:
                idcg = 0.0
                key = min(cutoff, len(positive_assets))
                if key not in self.idcgs:
                    for j in range(0, key):
                        idcg += 1.0 / math.log(j + 2.0)
                    self.idcgs[key] = idcg
                cutoff_idcgs[cutoff] = self.idcgs[key]

            # And, then, the DCGs
            dcg = 0.0
            i = 0
            k = 0
            max_cutoff = aux_cutoffs[-1]
            current_cutoff = aux_cutoffs[0]

            aux_cust_df = customer_df.head(max_cutoff)
            for index, row in aux_cust_df.iterrows():
                val = 1.0 if row[DEFAULT_ITEM_COL] in positive_assets else 0.0
                dcg += val / math.log(k + 2.0)
                k += 1
                if k == current_cutoff:
                    res_dict[current_cutoff] = dcg/cutoff_idcgs[current_cutoff]

                    if current_cutoff == max_cutoff:
                        current_cutoff = -1
                    else:
                        i += 1
                        current_cutoff = aux_cutoffs[i]

        # In case there is not enough recommended assets, we consider only
        # those in the ranking.
        if current_cutoff != -1:
            for j in range(i, len(cutoffs)):
                res_dict[aux_cutoffs[j]] = dcg / cutoff_idcgs[aux_cutoffs[j]]

        return res_dict

    def evaluate_indiv(self, customer_df, cutoff):

        customer = customer_df[DEFAULT_USER_COL].unique()[0]

        current_assets = set()
        if self.history is not None:
            current_pf = self.history[self.history[DEFAULT_USER_COL] == customer]
            if current_pf.shape[0] > 0:
                current_assets = set(current_pf[DEFAULT_ITEM_COL].unique())

        positive_assets = self.data.get_positive_assets(customer) - current_assets
        
        k = min(cutoff, len(positive_assets))
        idcg = 0.0
        if k not in self.idcgs:
            for i in range(0, min(cutoff, len(positive_assets))):
                idcg += 1.0 / math.log(i + 2.0)
            self.idcgs[k] = idcg
        else:
            idcg = self.idcgs[k]

        if idcg == 0.0:
            return 0.0

        k = 0
        dcg = 0.0
        for index, row in customer_df.iterrows():
            asset = row[DEFAULT_ITEM_COL]
            val = 1.0 if asset in positive_assets else 0.0
            dcg += val / math.log(k + 2.0)
            k += 1

        return dcg / idcg

