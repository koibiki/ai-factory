from abc import abstractmethod


class PredictModel:

    @abstractmethod
    def create_predict_model(self):
        pass

    @abstractmethod
    def run(self, X_train, y_train, X_valid, y_valid):
        pass
