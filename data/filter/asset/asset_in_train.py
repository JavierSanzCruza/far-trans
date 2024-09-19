#  Copyright (c) 2022. Terrier Team at University of Glasgow, http://http://terrierteam.dcs.gla.ac.uk
#
#  This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
#  If a copy of the MPL was not distributed with this  file, you can obtain one at
#  http://mozilla.org/MPL/2.0/.
from typing import Set

from beta_rec.utils.constants import DEFAULT_ITEM_COL
from pandas import DataFrame

from data.filter.asset.asset_filter import AssetFilter


class AssetInTrain(AssetFilter):
    """
    Only keeps those customers in the training se.
    """

    def filter(self, time_series: DataFrame, train: DataFrame, valid: DataFrame, test: DataFrame, split_date) -> Set:
        return set(train[DEFAULT_ITEM_COL].unique().flatten())
