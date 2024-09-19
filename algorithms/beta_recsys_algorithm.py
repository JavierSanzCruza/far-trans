#  Copyright (c) 2022. Terrier Team at University of Glasgow, http://http://terrierteam.dcs.gla.ac.uk
#
#  This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
#  If a copy of the MPL was not distributed with this  file, you can obtain one at
#  http://mozilla.org/MPL/2.0/.
#
#  This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
#  If a copy of the MPL was not distributed with this  file, you can obtain one at
#  http://mozilla.org/MPL/2.0/.

import numpy as np
import pandas as pd
from beta_rec.utils.constants import (
    DEFAULT_ITEM_COL,
    DEFAULT_RATING_COL,
    DEFAULT_USER_COL,
)

from .algorithm import Algorithm


class BetaRecSysAlgorithm(Algorithm):
    """
    Recommendation algorithm extracted from the Beta-RecSys recommendation open-source framework.

    Zaiqiao Meng, Richard McCreadie, Craig Macdonald et al. Beta-Rec: Build, Evaluate and Tune Automated Recommender
    Systems. 14th ACM Conference on Recommender Systems (RecSys 2020), pp. 588-590.
    """

    def __init__(self, data, model):
        """
        Initializes the algorithm.
        :param data: the data to use.
        :param model: the recommendation algorithm to use.
        """
        super().__init__(data)
        self.model = model

    def train(self, train_date):
        self.model.init()
        self.model.train()

    def recommend(self, rec_date, repeated, only_test_customers):
        recs = self.model.test()
        user_recommendations = []

        customers = (self.data.users & set(self.data.test[DEFAULT_USER_COL].unique().flatten())) if only_test_customers else self.data.users

        for customer in customers:
            user_recommendation = recs[recs[DEFAULT_USER_COL] == customer]

            if user_recommendation.shape[0] == 0: #if there are no recommendations for the customer, random rec:
                aux_assets = [x for x in self.data.assets]
                vals = np.random.random(len(aux_assets)).flatten()

                rec_list = []
                for i in range(len(aux_assets)):
                    rec_list.append({ DEFAULT_USER_COL: customer, DEFAULT_ITEM_COL: aux_assets[i], DEFAULT_RATING_COL: vals[i]})
                user_recommendation = pd.DataFrame(rec_list)
                
            if not repeated:
                items_per_user = set(self.data.train[self.data.train[DEFAULT_USER_COL] == customer][
                                         DEFAULT_ITEM_COL].unique().flatten())
                user_recommendation = user_recommendation[~user_recommendation[DEFAULT_ITEM_COL].isin(items_per_user)]
            user_recommendation = user_recommendation[user_recommendation[DEFAULT_ITEM_COL].isin(self.data.assets)]
            user_recommendation = user_recommendation.sort_values(by=DEFAULT_RATING_COL, ascending=False)
            user_recommendations.append(user_recommendation)
        def_df = pd.concat(user_recommendations)
        return def_df