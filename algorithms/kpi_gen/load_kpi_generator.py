import pandas as pd
from beta_rec.utils.constants import DEFAULT_TIMESTAMP_COL

from algorithms.kpi_gen.kpi_generator import KPIGenerator


class LoadKPIGenerator(KPIGenerator):

    def __init__(self, file):
        super().__init__()
        self.file = file

    def compute(self):
        self.kpis = pd.read_csv(self.file)
        self.kpis[DEFAULT_TIMESTAMP_COL] = pd.to_datetime(self.kpis[DEFAULT_TIMESTAMP_COL])