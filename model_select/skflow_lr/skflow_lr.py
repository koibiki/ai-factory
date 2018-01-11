import matplotlib.pyplot as plt
from tensorflow.contrib.learn import LinearRegressor
import pandas as pd
from sklearn.metrics import mean_squared_error

from model_select.predict_model import PredictModel


# knr 需要删除 nan 列
class TensorLr(PredictModel):
    tlr = None

    def create_predict_model(self):
        self.tlr = LinearRegressor()

    def run(self, X_train, y_train, X_valid, y_valid):
        self.create_predict_model()
        self.tlr.fit(X_train, y_train)
        y_pred = self.tlr.predict(X_valid)
        print("knr mean_squared_error:", mean_squared_error(y_pred, y_valid))

        x = range(len(y_pred))
        plt.plot(x, y_pred, 'r-*', label='y_pred')
        plt.plot(x, y_valid, 'b-o', label='y_valid')
        plt.legend()
        plt.show()

        pd_X = pd.DataFrame(X_valid, columns=None)
        pd_Y = pd.DataFrame(y_valid, columns=['Y'])
        pd_valid = pd.concat([pd_X, pd_Y], axis=1)
        pd_valid = pd_valid.sort_values(by='Y')

        X_valid = pd_valid.drop(['Y'], axis=1).values
        y_pred = self.tlr.predict(X_valid)
        y_valid = pd_valid.Y.values

        x = range(len(y_pred))
        plt.plot(x, y_pred, 'r-*', label='y_pred')
        plt.plot(x, y_valid, 'b-o', label='y_valid')
        plt.show()

    def predict(self, test_X):
        a_pred = self.tlr.predict(test_X)
        gbm_a_pattern = pd.read_csv('input/a_pattern.csv', names=['id'])
        gbm_a_pattern['Y'] = a_pred
        gbm_a_pattern.to_csv('output/knr_a_pattern.csv', index=None, header=None)
        print(gbm_a_pattern.head())
