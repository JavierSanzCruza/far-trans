import datetime
import pandas as pd
from beta_rec.utils.constants import (
    DEFAULT_ITEM_COL,
    DEFAULT_RATING_COL,
    DEFAULT_TIMESTAMP_COL,
)

from .indicators import *
from .kpi_generator import KPIGenerator


class ProfitVolatilityKPIGenerator(KPIGenerator):
    """
    Class for computing the basic technical indicators for the recommendation.
    """

    def __init__(self, data):
        super().__init__()
        self.data = data

    def compute(self):
        """
        Computes the desired KPIs.
        :return: a dataframe containing the KPIs.
        """

        timea = datetime.datetime.now()
        assets = self.data[DEFAULT_ITEM_COL].unique()
        asset_dfs = []
        # Step 2: For each asset:
        j = 0
        for asset in assets:
            # b) Now, we add it to a pandas DataFrame
            asset_time_series_df = self.data[self.data[DEFAULT_ITEM_COL] == asset]
            asset_time_series_df = asset_time_series_df.sort_values(by=DEFAULT_TIMESTAMP_COL, ascending=True)

            # b) Compute the technical indicators:
            asset_time_series_df = roi(asset_time_series_df)
            asset_time_series_df = volatility(asset_time_series_df)
            #asset_time_series_df = asset_time_series_df.dropna()

            asset_dfs.append(asset_time_series_df)

            j += 1
            if j % 100 == 0:
                string = "Generated the indicators for " + str(j) + " assets ("
                time_elapsed = datetime.datetime.now() - timea
                print(string + '{}'.format(time_elapsed) + ")")

        full_df = pd.concat(asset_dfs)
        self.kpis = full_df
