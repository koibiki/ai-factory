from sklearn.svm import SVR
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler

from model_select.predict_model import PredictModel

# gbr 需要删除 nan 列


class Svr(PredictModel):
    svr = None
    mms = None

    def create_predict_model(self):
        self.svr = SVR(kernel='rbf')

    def run(self, X_train, y_train, X_valid, y_valid):
        self.create_predict_model()
        self.mms = MinMaxScaler()
        X_train = self.mms.fit_transform(X_train)
        X_valid = self.mms.transform(X_valid)
        self.svr.fit(X_train, y_train)
        svr_y_pred = self.svr.predict(X_valid)
        print("svr mean_squared_error:", mean_squared_error(svr_y_pred, y_valid))
        return svr_y_pred, mean_squared_error(svr_y_pred, y_valid)

        # x = range(len(y_pred))
        # plt.plot(x, y_pred, 'r-*', label='y_pred')
        # plt.plot(x, y_valid, 'b-o', label='y_valid')
        # plt.legend()
        # plt.show()
        #
        # pd_X = pd.DataFrame(X_valid, columns=None)
        # pd_Y = pd.DataFrame(y_valid, columns=['Y'])
        # pd_valid = pd.concat([pd_X, pd_Y], axis=1)
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
        ss_test_X = self.mms.transform(test_X)
        a_pred = self.svr.predict(ss_test_X)
        return a_pred