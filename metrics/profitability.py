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
from beta_rec.utils.constants import (
    DEFAULT_ITEM_COL,
    DEFAULT_RATING_COL,
    DEFAULT_TIMESTAMP_COL,
)

from metrics.metric import Metric


class Profitability(Metric):
    """
    Class for computing the profitability of an asset in the future.
    """
    def __init__(self, data, num_years):
        """
        Initializes the value of the metric.
        :param data: the complete data.
        :param rec_date: the date the recommendation is produced.
        :param max_date: the maximum date to consider (here, the future date for computing the profitability).
        """
        super().__init__(data)
        self.profitabilities = dict()

        rec_df = data.time_series[data.time_series[DEFAULT_TIMESTAMP_COL] == self.data.valid_date]
        rec_df = rec_df.rename(columns={DEFAULT_RATING_COL: "current_val"})
        fut_df = data.time_series[data.time_series[DEFAULT_TIMESTAMP_COL] == self.data.test_date]
        fut_df = fut_df.rename(columns={DEFAULT_RATING_COL: "future_val"})
        combined = pd.merge(rec_df, fut_df, how="inner", on=[DEFAULT_ITEM_COL])
        combined = combined.dropna()
        combined["profit"] = pow(combined["future_val"], 1.0/(num_years + 0.0))/(combined["current_val"]) - 1.0

        for index, row in combined.iterrows():
            self.profitabilities[row[DEFAULT_ITEM_COL]] = row["profit"]


    def eval_indiv(self, customer_df, cutoff):
        value = 0.0
        for row in customer_df.iterrows():
            value += self.profitabilities[row[DEFAULT_ITEM_COL]]
        return value / (cutoff + 0.0)



