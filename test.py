import tensorflow as tf
from sklearn import datasets
import pandas as pd
import numpy as np
from sklearn.cross_validation import train_test_split
from sklearn.metrics import mean_squared_error
from model_select.skflow_dnn.skflow_dnn import TensorDNN

boston = datasets.load_boston()
boston_df = pd.DataFrame(np.c_[boston.data, boston.target], columns=np.append(boston.feature_names, 'MEDV'))
LABEL_COLUMN = ['MEDV']

FEATURE_COLUMNS = list(boston_df.columns[0:-1])
X_train, X_test, y_train, y_test = train_test_split(boston_df[FEATURE_COLUMNS], boston_df[LABEL_COLUMN], test_size=0.3)

# print(type(X_train))
#
# tf_feature_cols = [tf.contrib.layers.real_valued_column(k) for k in FEATURE_COLUMNS]
#
#
# def input_fn(X_train, y_train):
#     feature_cols = {k: tf.constant(X_train[k].values) for k in FEATURE_COLUMNS}
#     labels = tf.constant(y_train.values)
#     return feature_cols, labels
#
# def train_input_fn(X_train, y_train):
#     return input_fn(X_train, y_train)
# 
# def test_input_fn(X_test, y_test):
#     return input_fn(X_test, y_test)
#
# regressor = tf.contrib.learn.DNNRegressor(feature_columns=tf_feature_cols,hidden_units=[64, 128])
#
# regressor.fit(input_fn=lambda: train_input_fn(X_train, y_train), steps=5000)
#
# predict = regressor.predict(input_fn=lambda: input_fn(X_test, y_test), as_iterable=False)
# ev = regressor.evaluate(input_fn=lambda: test_input_fn(X_test, y_test), steps=1)
# print('ev: {}'.format(ev))
# print('mse:', mean_squared_error(predict, y_test.values))
dnn = TensorDNN()
dnn.run(X_train, y_train, X_test, y_test)