from sklearn.neighbors import KNeighborsRegressor
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import mean_squared_error

from model_select.predict_model import PredictModel

# knr 需要删除 nan 列


class KNR(PredictModel):
    knr = None

    def create_predict_model(self):
        self.knr = KNeighborsRegressor(weights='uniform')

    def run(self, X_train, y_train, X_valid, y_valid):
        self.create_predict_model()
        self.knr.fit(X_train, y_train)
        y_pred = self.knr.predict(X_valid)
        print("knr mean_squared_error:", mean_squared_error(y_pred, y_valid))

        x = range(len(y_pred))
        plt.plot(x, y_pred, 'r-*', label='y_pred')
        plt.plot(x, y_valid, 'b-o', label='y_valid')
        plt.legend()
        plt.show()

        pd_valid = pd.concat([X_valid, y_valid], axis=1)
        pd_valid = pd_valid.sort_values(by='Y')

        sort_X_valid = pd_valid.drop(['Y'], axis=1).values
        sort_y_pred = self.knr.predict(sort_X_valid)
        sort_y_valid = pd_valid.Y.values

        x = range(len(sort_y_pred))
        plt.plot(x, sort_y_pred, 'r-*', label='y_pred')
        plt.plot(x, sort_y_valid, 'b-o', label='y_valid')
        plt.legend()
        plt.show()

    def predict(self, test_X):
        a_pred = self.knr.predict(test_X)
        return a_pred
