from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error

from model_select.predict_model import PredictModel

# gbr 需要删除 nan 列


class LinearR(PredictModel):
    lr = None

    def create_predict_model(self):
        self.lr = LinearRegression()

    def run(self, X_train, y_train, X_valid, y_valid):
        self.create_predict_model()
        ss = StandardScaler()
        X_train = ss.fit_transform(X_train)
        X_valid = ss.transform(X_valid)
        self.lr.fit(X_train, y_train)
        y_pred = self.lr.predict(X_valid)
        print("lr mean_squared_error:", mean_squared_error(y_pred, y_valid))

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
        y_pred = self.lr.predict(X_valid)
        y_valid = pd_valid.Y.values

        x = range(len(y_pred))
        plt.plot(x, y_pred, 'r-*', label='y_pred')
        plt.plot(x, y_valid, 'b-o', label='y_valid')
        plt.show()

    def predict(self, test_X):
        a_pred = self.lr.predict(test_X)
        gbr_a_pattern = pd.read_csv('input/a_pattern.csv', names=['id'])
        gbr_a_pattern['Y'] = a_pred
        gbr_a_pattern.to_csv('output/gbr_a_pattern.csv', index=None, header=None)
        print(gbr_a_pattern.head())
        return a_pred
