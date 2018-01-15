from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error

from model_select.predict_model import PredictModel

# gbr 需要删除 nan 列


class LinearR(PredictModel):
    lr = None
    ss = None

    def create_predict_model(self):
        self.lr = LinearRegression()

    def run(self, X_train, y_train, X_valid, y_valid):
        self.create_predict_model()
        self.ss = StandardScaler()
        ss_X_train = self.ss.fit_transform(X_train)
        ss_X_valid = self.ss.transform(X_valid)
        self.lr.fit(ss_X_train, y_train)
        y_pred = self.lr.predict(ss_X_valid)
        print("lr mean_squared_error:", mean_squared_error(y_pred, y_valid))

        x = range(len(y_pred))
        plt.plot(x, y_pred, 'r-*', label='y_pred')
        plt.plot(x, y_valid, 'b-o', label='y_valid')
        plt.legend()
        plt.show()

        pd_valid = pd.concat([X_valid, y_valid], axis=1)
        pd_valid = pd_valid.sort_values(by='Y')

        sort_X_valid = pd_valid.drop(['Y'], axis=1).values
        sort_X_valid = self.ss.transform(sort_X_valid)
        sort_y_pred = self.lr.predict(sort_X_valid)
        sort_y_valid = pd_valid.Y.values

        x = range(len(sort_y_pred))
        plt.plot(x, sort_y_pred, 'r-*', label='y_pred')
        plt.plot(x, sort_y_valid, 'b-o', label='y_valid')
        plt.legend()
        plt.show()

    def predict(self, test_X):
        ss_test_X = self.ss.transform(test_X)
        a_pred = self.lr.predict(ss_test_X)
        return a_pred
