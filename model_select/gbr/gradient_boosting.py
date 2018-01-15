from sklearn.ensemble import GradientBoostingRegressor
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import mean_squared_error

from model_select.predict_model import PredictModel

# gbr 需要删除 nan 列


class GBR(PredictModel):
    gbr = None

    def create_predict_model(self):
        self.gbr = GradientBoostingRegressor(loss='ls', alpha=0.9,
                                             n_estimators=500,
                                             learning_rate=0.05,
                                             max_depth=8,
                                             subsample=0.8,
                                             max_features=0.6,
                                             min_samples_split=9,
                                             max_leaf_nodes=10)

    def run(self, X_train, y_train, X_valid, y_valid):
        self.create_predict_model()
        self.gbr.fit(X_train, y_train)
        y_pred = self.gbr.predict(X_valid)
        print("gbr mean_squared_error:", mean_squared_error(y_pred, y_valid))

        # x = range(len(y_pred))
        # plt.plot(x, y_pred, 'r-*', label='y_pred')
        # plt.plot(x, y_valid, 'b-o', label='y_valid')
        # plt.legend()
        # plt.show()
        #
        # pd_valid = pd.concat([X_valid, y_valid], axis=1)
        # pd_valid = pd_valid.sort_values(by='Y')
        #
        # X_valid = pd_valid.drop(['Y'], axis=1).values
        # y_pred = self.gbr.predict(X_valid)
        # y_valid = pd_valid.Y.values
        #
        # x = range(len(y_pred))
        # plt.plot(x, y_pred, 'r-*', label='y_pred')
        # plt.plot(x, y_valid, 'b-o', label='y_valid')
        # plt.legend()
        # plt.show()

    def predict(self, test_X):
        a_pred = self.gbr.predict(test_X)
        return a_pred
