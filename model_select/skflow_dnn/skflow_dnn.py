from tensorflow.contrib.learn import DNNRegressor
import tensorflow as tf
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import mean_squared_error

from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import MaxAbsScaler

from model_select.predict_model import PredictModel


# knr 需要删除 nan 列
class TensorDNN(PredictModel):
    tdnn = None
    X_train = None
    y_train =None
    X_valid = None
    y_valid = None
    feature_columns = None
    ss = None

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

        self.tdnn = DNNRegressor(feature_columns=tf_feature_cols, hidden_units=[64, 128])
        self.tdnn.fit(input_fn=lambda: self.input_fn(ss_X_train, y_train))
        y_pred = self.tdnn.predict(input_fn=lambda: self.input_fn(ss_X_valid, y_valid), as_iterable=False)
        print("knr mean_squared_error:", mean_squared_error(y_pred, y_valid))

        x = range(len(y_pred))
        plt.plot(x, y_pred, 'r-*', label='y_pred')
        plt.plot(x, y_valid, 'b-o', label='y_valid')
        plt.legend()
        plt.show()

        pd_valid = pd.concat([X_train, y_train], axis=1)
        pd_valid = pd_valid.sort_values(by='Y')

        sort_X_valid = pd_valid.drop(['Y'], axis=1).values
        ss_sort_X_valid = self.ss.transform(sort_X_valid)
        ss_sort_X_valid = pd.DataFrame(ss_sort_X_valid, columns=self.feature_columns)
        sort_y_valid = pd_valid.Y.values
        sort_y_pred = self.tdnn.predict(input_fn=lambda: self.input_fn(ss_sort_X_valid, sort_y_valid), as_iterable=False)

        x = range(len(y_pred))
        plt.plot(x, sort_y_pred, 'r-*', label='y_pred')
        plt.plot(x, sort_y_valid, 'b-o', label='y_valid')
        plt.legend()
        plt.show()

    def predict(self, test_X):
        ss_test_X = self.ss.transform(test_X)
        a_pred = self.tdnn.predict(ss_test_X)
        return a_pred
