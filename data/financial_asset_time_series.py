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

class FinancialAssetTimeSeries:
    """
    Class for loading and storing the time series pricing information for financial assets.
    We assume that the time series prices is stored in a csv file, containing three fields:
    asset_id, timestamp (date in YYYY-mm-dd format) and closing price for the corresponding date.
    """

    def __init__(self, file_name):
        """
        Initializes the financial time series data.
        :param file_name: location of the file containing the time series information.
        """
        # Initialize the financial interaction dataset.
        self.file_name = file_name
        self.data = None

    def load(self):
        """
        Loads the data in the raw file into memory. It turns the contents of the file into a dataframe.
        :return: a dataframe containing the timeseries information: asset_id, date, close_price.
        """
        data = pd.read_csv(
            self.file_name,
            skiprows=[0],
            engine="python",
            names=[
                DEFAULT_ITEM_COL,
                DEFAULT_TIMESTAMP_COL,
                DEFAULT_RATING_COL,
            ],
        )
        self.data = data[[DEFAULT_ITEM_COL, DEFAULT_RATING_COL, DEFAULT_TIMESTAMP_COL]]
        self.data[DEFAULT_TIMESTAMP_COL] = pd.to_datetime(self.data[DEFAULT_TIMESTAMP_COL])
        # As they represent cases where the data has not been properly collected, we remove those items with
        # zero values.
        assets = self.data[self.data[DEFAULT_RATING_COL] == 0.0][DEFAULT_ITEM_COL].unique().flatten()
        self.data = self.data[~self.data[DEFAULT_ITEM_COL].isin(assets)]
        # Remove duplicates
        self.data.drop_duplicates(subset=[DEFAULT_ITEM_COL, DEFAULT_TIMESTAMP_COL], keep='last', inplace=True)
