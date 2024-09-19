#  Copyright (c) 2022. Terrier Team at University of Glasgow, http://http://terrierteam.dcs.gla.ac.uk
#
#  This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
#  If a copy of the MPL was not distributed with this  file, you can obtain one at
#  http://mozilla.org/MPL/2.0/.
#
#  This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
#  If a copy of the MPL was not distributed with this  file, you can obtain one at
#  http://mozilla.org/MPL/2.0/.

from beta_rec.utils.constants import (
    DEFAULT_ITEM_COL, DEFAULT_USER_COL, )

from metrics.metric import Metric


class KPIEvaluationMetric(Metric):
    """
    Class for evaluating an asset considering a technical indicator (profitability, volatility, etc.)
    """

    def __init__(self, data, values):
        """
        Initializes the value of the metric.
        :param data: the complete data.
        :param values: the values of the technical indicator.
        """
        super().__init__(data)
        self.values = values
        for asset in self.data.assets:
            if asset not in self.values:
                self.values[asset] = 0.0

    def evaluate(self, recs, cutoff, target_custs, only_test_customers):
        customers = self.data.users & set(
            self.data.test[DEFAULT_USER_COL].unique().flatten()) if only_test_customers else self.data.users
        customers = customers & target_custs

        aux_recs = recs[recs[DEFAULT_USER_COL].isin(customers)]
        aux_recs.groupby(DEFAULT_USER_COL).head(cutoff)
        aux_recs["metric"] = aux_recs[DEFAULT_ITEM_COL].apply(lambda x: self.values[x])
        aux_recs = aux_recs.groupby(DEFAULT_USER_COL).mean()
        aggregated = aux_recs["metric"].sum() / (0.0 + len(customers))

        return aux_recs.reset_index()[[DEFAULT_USER_COL, "metric"]], aggregated

    def evaluate_cutoffs(self, recs, cutoffs, target_custs, only_test_customers):
        customers = self.data.users & set(self.data.test[DEFAULT_USER_COL].unique().flatten()) if only_test_customers else self.data.users
        customers = customers & target_custs

        cutoffs.sort(reverse=True)

        max_cutoff = cutoffs[0]
        aux_recs = recs[recs[DEFAULT_USER_COL].isin(customers)]
        aux_recs.groupby(DEFAULT_USER_COL).head(max_cutoff)
        aux_recs["metric"] = aux_recs[DEFAULT_ITEM_COL].apply(lambda x: self.values[x])

        cutoff_dict = dict()
        for cutoff in cutoffs:
            aux_recs = aux_recs.groupby(DEFAULT_USER_COL).head(cutoff)
            recs_res = aux_recs.groupby(DEFAULT_USER_COL).mean()
            aggregated = recs_res["metric"].sum()/(0.0 + len(customers))

            cutoff_dict[cutoff] = (recs_res.reset_index()[[DEFAULT_USER_COL, "metric"]], aggregated)

        return cutoff_dict

    def evaluate_indiv_cutoffs(self, customer_df, cutoffs):
        """
        Evaluate an individual ranking on different cutoffs
        :param customer_df: the individual recommendation ranking
        :param cutoffs: the cutoffs we want to consider.
        :return: the value of the metric
        """

        cutoffs.sort()
        cutoff_dict = dict()
        value = 0.0

        max_cutoff = cutoffs[-1]
        current_cutoff =cutoffs[0]

        aux_cust_df = customer_df.head(max_cutoff)
        i = 0
        k = 0
        for index, row in aux_cust_df.iterrows():
            value += (self.values[row[DEFAULT_ITEM_COL]] + k*value)/(k+1.0)
            k += 1
            if k == current_cutoff:
                cutoff_dict[current_cutoff] = cutoff_dict
                if current_cutoff == max_cutoff:
                    current_cutoff = -1
                else:
                    i += 1
                    current_cutoff = cutoffs[i]

        # In case there is not enough recommended assets, we consider only
        # those in the ranking.
        if current_cutoff != -1:
            for j in range(i, len(cutoffs)):
                cutoff_dict[cutoffs[j]] = value

        return cutoff_dict

    def evaluate_indiv(self, customer_df, cutoff):
        """
        Evaluation of an individual ranking.
        :param customer_df: the individual recommendation ranking.
        :param cutoff: the cutoff we want to consider. If negative, we consider the full ranking.
        :return: the value of the metric.
        """
        value = 0.0
        j = 0
        divide = cutoff if cutoff >= 0 else customer_df.shape[0]
        for index, row in customer_df.iterrows():
            if j < divide:
                value += self.values[row[DEFAULT_ITEM_COL]]
                j += 1
            else:
                break
        return value / (divide + 0.0)



