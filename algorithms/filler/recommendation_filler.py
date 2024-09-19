#  Copyright (c) 2023. Terrier Team at University of Glasgow, http://http://terrierteam.dcs.gla.ac.uk
#
#  This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
#  If a copy of the MPL was not distributed with this  file, you can obtain one at
#  http://mozilla.org/MPL/2.0/.
import pandas as pd
from beta_rec.utils.constants import DEFAULT_USER_COL, DEFAULT_ITEM_COL, DEFAULT_RATING_COL

__author__ = "Javier Sanz-Cruzado (javier.sanz-cruzadopuig@glasgow.ac.uk)"


class RecommendationFiller:
    """
    Class for filling recommendations with the remaining assets.
    """

    def __init__(self, data, repeated, in_portfolio, portfolios):
        """
        Constructor.
        :param data: the full data.
        :param repeated: True if we allow repeated assets, False otherwise
        :param in_portfolio: True if we allow assets in the customer portfolio, False otherwise.
        :param portfolios: the current portfolios.
        """
        self.data = data
        self.repeated = repeated
        self.in_portfolio = in_portfolio
        self.portfolios = portfolios

    def fill(self, recommendations):
        """
        Fill a recommendation. New assets are added with minimum score equal to minimum value - 1.0.
        :param recommendations: the recommendations to complete.
        :return: the filled recommendations.
        """
        user_recommendations = []
        customers = [x for x in recommendations[DEFAULT_USER_COL].unique()]
        assets = {x for x in self.data.assets}

        for customer in customers:
            user_recommendation = recommendations[recommendations[DEFAULT_USER_COL] == customer]
            user_recommendation = user_recommendation.dropna()
            to_add = assets - set(user_recommendation[DEFAULT_ITEM_COL].unique())
            min_value = min(user_recommendation[DEFAULT_RATING_COL]) - 1.0
            if not self.repeated:
                items_per_user = set(
                    self.data.train[self.data.train[DEFAULT_USER_COL] == customer][DEFAULT_ITEM_COL].unique().flatten())
                # We remove repeated assets:
                user_recommendation = user_recommendation[~user_recommendation[DEFAULT_ITEM_COL].isin(items_per_user)]
                to_add = to_add - items_per_user
            elif not self.in_portfolio:
                items_per_user = set(
                    self.portfolios[self.portfolios[DEFAULT_USER_COL] == customer][DEFAULT_ITEM_COL].unique().flatten())
                user_recommendation = user_recommendation[~user_recommendation[DEFAULT_ITEM_COL].isin(items_per_user)]
                to_add = to_add - items_per_user

            list_assets = []
            for asset in to_add:
                list_assets.append({DEFAULT_USER_COL: customer, DEFAULT_ITEM_COL: asset, DEFAULT_RATING_COL: min_value})
            extra_df = pd.DataFrame(list_assets)
            user_recommendation = pd.concat([user_recommendation, extra_df])
            user_recommendations.append(user_recommendation)

        def_df = pd.concat(user_recommendations)
        def_df.reset_index(inplace=True, drop=True)

        return def_df
