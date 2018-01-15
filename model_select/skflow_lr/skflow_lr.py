import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.contrib.learn import LinearRegressor
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error

from model_select.predict_model import PredictModel
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import MaxAbsScaler

# knr 需要删除 nan 列
class TensorLr(PredictModel):

    ss = None
    tdnn = None
    feature_columns = None

    def input_fn(self, X_train, y_train):
        feature_cols = {k: tf.constant(X_train[k].values)for k in self.feature_columns}
        labels = tf.constant(y_train.values)
        return feature_cols, labels

    def create_predict_model(self):
        self.ss = MaxAbsScaler()
        print()

    def run(self, X_train, y_train, X_valid, y_valid):
        self.create_predict_model()

        self.feature_columns = X_train.columns
        print(self.feature_columns)
        tf_feature_cols = [tf.contrib.layers.real_valued_column(k) for k in self.feature_columns]

        ss_X_train = self.ss.fit_transform(X_train)
        ss_X_valid = self.ss.transform(X_valid)

        ss_X_train = pd.DataFrame(ss_X_train, columns=self.feature_columns)
        ss_X_valid = pd.DataFrame(ss_X_valid, columns=self.feature_columns)

        print(ss_X_train.head())
        print(ss_X_valid.head())

        self.tdnn = LinearRegressor(feature_columns=tf_feature_cols)
        self.tdnn.fit(input_fn=lambda: self.input_fn(ss_X_train, y_train), steps=5000)
        y_pred = self.tdnn.predict(input_fn=lambda: self.input_fn(ss_X_valid, y_valid), as_iterable=False)
        print("knr mean_squared_error:", mean_squared_error(y_pred, y_valid))

        x = range(len(y_pred))
        plt.plot(x, y_pred, 'r-*', label='y_pred')
        plt.plot(x, y_valid, 'b-o', label='y_valid')
        plt.legend()
        plt.show()
        #
        # pd_valid = pd.concat([ss_X_train, y_valid], axis=1)
        # pd_valid = pd_valid.sort_values(by='Y')
        #
        # sort_X_valid = pd_valid.drop(['Y'], axis=1).values
        # ss_sort_X_valid = pd.DataFrame(sort_X_valid, columns=self.feature_columns)
        # sort_y_valid = pd_valid.iloc[:, -1]
        # sort_y_pred = self.tdnn.predict(input_fn=lambda: self.input_fn(ss_sort_X_valid, sort_y_valid), as_iterable=False)
        #
        # x = range(len(y_pred))
        # plt.plot(x, sort_y_pred, 'r-*', label='y_pred')
        # plt.plot(x, sort_y_valid, 'b-o', label='y_valid')
        # plt.legend()
        # plt.show()

    def predict(self, test_X):
        ss_test_X = self.ss.transform(test_X)
        ss_X_test = pd.DataFrame(ss_test_X, columns=self.feature_columns)
        a_pred = self.tdnn.predict(input_fn=lambda: self.input_fn(ss_X_test, pd.DataFrame(np.zeros(len(test_X)))), as_iterable=False)
        return a_pred
