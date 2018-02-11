import pandas as pd
from sklearn.linear_model import Lasso
from sklearn.preprocessing import MaxAbsScaler

from model_selection.predict_model import PredictModel


class LassoR(PredictModel):

    ls = None
    mas = None
    columns = None

    def create_predict_model(self):
        self.ls = Lasso(max_iter=10000, random_state=520)
        self.mas = MaxAbsScaler()

    def fit(self, X_train, X_valid, y_train, y_valid):
        self.create_predict_model()
        self.columns = X_train.columns
        # X_train = self.mas.fit_transform(X_train)
        self.ls.fit(X_train, y_train)

    def predict(self, X_test):
        # X_test = self.mas.transform(X_test)
        return self.ls.predict(X_test)

    def can_get_feature_importance(self):
        return True

    def feature_importance(self, level=2):
        coef = pd.Series(self.ls.coef_, index=self.columns)
        return coef[coef != 0]


class LassoC(PredictModel):

    def create_predict_model(self):
        pass

    def fit(self, X_train, X_valid, y_train, y_valid):
        pass

    def predict(self, X_test):
        pass
