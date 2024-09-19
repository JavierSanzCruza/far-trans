#  Copyright (c) 2023. Terrier Team at University of Glasgow, http://http://terrierteam.dcs.gla.ac.uk
#
#  This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
#  If a copy of the MPL was not distributed with this  file, you can obtain one at
#  http://mozilla.org/MPL/2.0/.
#

import pandas as pd
from beta_rec.utils.constants import DEFAULT_ITEM_COL, DEFAULT_USER_COL, DEFAULT_RATING_COL

from algorithms.group.group_algorithm import GroupAlgorithm

DEFAULT_GROUP_COL = "col_group"

__author__ = "Javier Sanz-Cruzado (javier.sanz-cruzadopuig@glasgow.ac.uk)"


class GroupPopularityAlgorithm(GroupAlgorithm):
    """
    Group recommendation algorithm based on popularity
    """

    def __init__(self, data, group_build):
        """
        Configures the profitability prediction model.
        :param data: the data for training and applying recommendations.
        :param group_build: the group builder.
        """
        super().__init__(data, group_build)

        self.assets_df = None
        self.groups = None
        self.rankings = None

    def train(self, train_date):

        self.groups = self.group_build.group(train_date)

        rankings = []

        i = 0
        for group in self.groups[1]:
            pops = self.data.train[(self.data.train[DEFAULT_USER_COL].isin(group))].groupby(DEFAULT_ITEM_COL).sum()
            pops = pops.reset_index()
            pops = pops[[DEFAULT_ITEM_COL, DEFAULT_RATING_COL]]

            aux_pops = []
            for asset in self.data.assets - set(pops[DEFAULT_ITEM_COL].unique()):
                aux_pops.append({DEFAULT_ITEM_COL: asset, DEFAULT_RATING_COL: 0.0})

            pops = pd.concat([pops, pd.DataFrame(aux_pops)])
            pops[DEFAULT_GROUP_COL] = i
            rankings.append(pops)
            i += 1
        self.rankings = pd.concat(rankings)

    def recommend(self, rec_time):
        """
        Generates the recommendation.
        :param rec_time: the recommendation time.
        :return: the produced recommendations.
        """

        return self.rankings
