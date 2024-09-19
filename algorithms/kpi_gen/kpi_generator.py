from abc import abstractmethod, ABC


class KPIGenerator(ABC):

    def __init__(self):
        self.kpis = None

    @abstractmethod
    def compute(self):
        pass

    def get_kpis(self):
        if self.kpis is None:
            self.compute()
        else:
            return self.kpis

    def print_kpis(self, file):
        self.kpis.to_csv(file, index=False)
