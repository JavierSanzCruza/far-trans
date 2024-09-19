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

import numpy as np
import pandas as pd
from beta_rec.utils.constants import (
    DEFAULT_ITEM_COL,
    DEFAULT_RATING_COL,
    DEFAULT_USER_COL,
)
from scipy.sparse import csr_matrix, lil_matrix

from .algorithm import Algorithm


class UBkNNAlgorithm(Algorithm):
    """
    Implementation of the user-based kNN recommendation algorithm.
    """

    def __init__(self, data, nn_method, k, norm):
        """
        Initializes the algorithm.
        :param data: the data.
        :param nn_method: the name of the nearest neighbor similarity. Allowed: "cosine" and "jaccard".
        :param k: the number of neighbors to consider.
        :param norm: True if we want to use normalized kNN, false otherwise.
        """
        super().__init__(data)
        self.matrix = None
        self.modmatrix = None
        self.nn_method = nn_method
        self.modules = None
        self.users_to_idx = dict()
        self.idx_to_users = dict()
        self.items_to_idx = dict()
        self.idx_to_items = dict()
        self.k = k
        self.norm = norm

    def train(self, train_date):
        self.modules = dict()
        if self.nn_method == "cosine":
            for u in self.data.train[DEFAULT_USER_COL].unique().flatten():
                u_df = self.data.train[self.data.train[DEFAULT_USER_COL] == u]
                module = 0.0
                for index, row in u_df.iterrows():
                    module += row[DEFAULT_RATING_COL]*row[DEFAULT_RATING_COL]
                self.modules[u] = math.sqrt(module)
        else: # if it is jaccard:
            for u in self.data.train[DEFAULT_USER_COL].unique().flatten():
                u_df = self.data.train[self.data.train[DEFAULT_USER_COL] == u]
                self.modules[u] = u_df.shape[0] + 0.0

        j = 0
        self.users_to_idx = dict()
        self.idx_to_users = dict()
        for u in self.data.train[DEFAULT_USER_COL].unique():
            self.users_to_idx[u] = j
            self.idx_to_users[j] = u
            j += 1

        self.items_to_idx = dict()
        self.idx_to_items = dict()
        j = 0
        for i in (set(self.data.train[DEFAULT_ITEM_COL].unique())):
            self.items_to_idx[i] = j
            self.idx_to_items[j] = i
            j += 1

        self.matrix = lil_matrix((len(self.users_to_idx), len(self.items_to_idx)))
        self.modmatrix = lil_matrix((len(self.users_to_idx), len(self.users_to_idx)))

        def row_function(row):
            uidx = self.users_to_idx[row[DEFAULT_USER_COL]]
            iidx = self.items_to_idx[row[DEFAULT_ITEM_COL]]
            self.matrix[uidx, iidx] = row[DEFAULT_RATING_COL]

        self.data.train.apply(lambda x: row_function(x), axis=1)
        for uidx, u in self.idx_to_users.items():
            mod = self.modules[u]
            self.modmatrix[uidx, uidx] = 1.0/(mod+0.0) if mod > 0 else 1.0

        self.matrix = self.matrix.tocsr()
        self.modmatrix = self.modmatrix.tocsr()
        return

    def recommend(self, rec_date, repeated, only_test_customers):

        customers = (self.data.users & set(self.data.test[DEFAULT_USER_COL].unique().flatten())) if only_test_customers else self.data.users

        aux_cust = [x for x in customers]
        aux_assets = [x for x in self.data.assets]
        index = pd.MultiIndex.from_product([aux_cust, aux_assets], names=[DEFAULT_USER_COL, DEFAULT_ITEM_COL])
        test = pd.DataFrame(index=index).reset_index()

        # Find the similarities
        sims = self.matrix * self.matrix.transpose()
        if self.nn_method == "cosine":
            sims = self.modmatrix * sims * self.modmatrix
            for i in range(0, len(self.users_to_idx)):
                sims[i, i] = 0.0
        else: # Jaccard
            for uidx in range(0, len(self.users_to_idx)):
                for vidx in range(0, len(self.users_to_idx)):
                    sims[uidx, vidx] = sims[uidx, vidx] / (1.0/self.modmatrix[uidx, uidx] +
                                                           1.0/self.modmatrix[vidx, vidx] - sims[uidx, vidx])

        # Find the neighbors:
        mask = lil_matrix((len(self.users_to_idx), len(self.users_to_idx)))
        mask[:, np.argpartition(sims.toarray(), -self.k)[:, -self.k:]] = True
        sims = csr_matrix(np.where(mask.toarray(), sims.toarray(),
                                   csr_matrix((len(self.users_to_idx), len(self.users_to_idx))).toarray()))

        ratings = sims * self.matrix

        recs = []
        for u in aux_cust:
            uidx = self.users_to_idx[u]

            if not repeated:
                to_explore = set(np.nonzero(self.matrix[uidx, :] == 0)[1].flatten()) & \
                             set(np.nonzero(ratings[uidx, :] > 0)[1].flatten())
            else:
                to_explore = set(np.nonzero(self.matrix[uidx, :] >= 0)[1].flatten()) & \
                             set(np.nonzero(ratings[uidx, :] > 0)[1].flatten())

            for i in to_explore:
                item = self.idx_to_items[i]

                if self.norm:
                    recs.append({
                        DEFAULT_USER_COL: u,
                        DEFAULT_ITEM_COL: item,
                        DEFAULT_RATING_COL: ratings[uidx, i] if ratings[uidx, i] > 0 else 0
                    })
                else:
                    recs.append({
                        DEFAULT_USER_COL: u,
                        DEFAULT_ITEM_COL: item,
                        DEFAULT_RATING_COL: ratings[uidx, i]
                    })

        recomms = pd.DataFrame(recs)
        recomms = recomms[recomms[DEFAULT_ITEM_COL].isin(self.data.assets)]
        recomms = pd.merge(test, recomms, on=[DEFAULT_USER_COL, DEFAULT_ITEM_COL], how="left")
        recomms = recomms.fillna(0.0)
        recomms = recomms.sort_values(by=[DEFAULT_USER_COL, DEFAULT_RATING_COL], ascending=[True, False])
        return recomms

