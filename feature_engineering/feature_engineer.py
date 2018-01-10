from abc import abstractmethod


class FeatureEngineer:

    @abstractmethod
    # train DataFrame
    # tests DataFrame[]
    def execute(self, train, test):
        pass
