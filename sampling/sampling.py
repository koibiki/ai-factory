from abc import abstractmethod


class Sampling:

    @abstractmethod
    def train_test_split(self, train, test, test_size, random_state):
        pass
