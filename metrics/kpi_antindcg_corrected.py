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

from beta_rec.utils.constants import (
    DEFAULT_ITEM_COL,
    DEFAULT_RATING_COL,
    DEFAULT_TIMESTAMP_COL,
    DEFAULT_USER_COL,
)

from metrics.kpi_evaluation_metric import KPIEvaluationMetric


class KPIAntiNDCGCorrected(KPIEvaluationMetric):
    """
    Class for computing the profitability of an asset in the future.
    """
    def __init__(self, data, kpi_gen, kpi_name, metric_threshold=0.0):
        """
        Initializes the value of the metric.
        :param data: the complete data.
        """
        super().__init__(data, kpi_gen, kpi_name)
        self.metric_threshold = metric_threshold


    def evaluate_indiv(self, customer_df, cutoff):
        value = 0.0
        customer = customer_df[DEFAULT_USER_COL].unique()[0]

        # user profitability:
        user_profit = dict()
        user_profit_list = []
        positive_assets = self.data.get_positive_assets(customer)

        if len(positive_assets) == 0.0:
            # In this case it is impossible to recommend good things:
            return 0.0

        for asset in positive_assets:
            if asset not in self.values:
                continue
            else:
                val = self.values[asset] if self.values[asset] < self.metric_threshold else self.metric_threshold
            user_profit[asset] = val
            user_profit_list.append(val)
        user_profit_list = sorted(user_profit_list, reverse=False)

        idcg = 0.0
        for k in range(0, min(cutoff, len(user_profit_list))):
            idcg += user_profit_list[k]/math.log(k+2.0)

        count_positive = 0
        k = 0
        dcg = 0.0
        for index, row in customer_df.iterrows():
            asset = row[DEFAULT_ITEM_COL]
            if asset in positive_assets:
                count_positive += 1
            if asset in positive_assets and asset in self.values:
                val = self.values[asset] if self.values[asset] < self.metric_threshold else self.metric_threshold
                dcg += val / math.log(k+2.0)

            k += 1

        if count_positive > 0:
            if idcg == 0.0:
                return 1.0
            return 1.0 - dcg/idcg
        return 0.0
