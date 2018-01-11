from tensorflow.contrib.learn import DNNRegressor
import tensorflow as tf
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import mean_squared_error

from model_select.predict_model import PredictModel


# knr 需要删除 nan 列
class TensorDNN(PredictModel):
    tdnn = None
    X_train = None
    y_train =None
    X_valid = None
    y_valid = None
    FEATURE_COLUMNS = None

    def __init__(self, feature_cols):
        self.FEATURE_COLUMNS = feature_cols

    def input_fn(self,X_train, y_train):
        feature_cols = {k: tf.constant(X_train[k].values)for k in range(len(self.FEATURE_COLUMNS))}
        labels = tf.constant(y_train.values)
        return feature_cols, labels

    def create_predict_model(self):
        print()

    def run(self, X_train, y_train, X_valid, y_valid):
        self.create_predict_model()
        feature_cols = [tf.contrib.layers.real_valued_column(k) for k in self.FEATURE_COLUMNS]
        self.tdnn = DNNRegressor(hidden_units=[100, 40], feature_columns=feature_cols)
        self.tdnn.fit(input_fn=lambda: self.input_fn(X_train, y_train))
        y_pred = self.tdnn.predict(X_valid)
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
        y_pred = self.tdnn.predict(X_valid)
        y_valid = pd_valid.Y.values

        x = range(len(y_pred))
        plt.plot(x, y_pred, 'r-*', label='y_pred')
        plt.plot(x, y_valid, 'b-o', label='y_valid')
        plt.show()

    def predict(self, test_X):
        a_pred = self.tdnn.predict(test_X)
        gbm_a_pattern = pd.read_csv('input/a_pattern.csv', names=['id'])
        gbm_a_pattern['Y'] = a_pred
        gbm_a_pattern.to_csv('output/knr_a_pattern.csv', index=None, header=None)
        print(gbm_a_pattern.head())
