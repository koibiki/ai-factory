import pandas as pd
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cross_validation import train_test_split
from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist
from sklearn.metrics import mean_squared_error

from feature_engineering.delete_date import delete_date
from feature_engineering.separate_str_num import separate_str_num
from feature_engineering.feature_vector import feature_vector
from feature_engineering.compute_angle import compute_angle
from feature_engineering.delete_constant import delete_constant
from feature_engineering.rank_by_max import rank_feature_by_max
from feature_engineering.rank_feature import rank_feature
from feature_engineering.rank_feature import rank_feature_count
from feature_engineering.nan_stastics import nan_statics
from feature_engineering.delete_nan import delete_all_nan
from sampling.stratified_sampling import stratified_sampling
from sklearn.decomposition import PCA
from model_select.gbr.gradient_boosting import GBR
from model_select.xgboost.xgb import Xgb
from model_select.knn.knn import KNR
from model_select.skflow_dnn.skflow_dnn import TensorDNN
from model_select.light_gbm.light_gbm import LightGBM
from model_select.linear.linear import LinearR

from sklearn.preprocessing import StandardScaler

train = pd.read_csv('input/train.csv')
test = pd.read_csv('input/testa.csv')

train_X = train.iloc[:, 1:-1]
train_Y = train.Y
test_X = test.iloc[:, 1:]

train_X, test_X = delete_date(train_X, test_X)

train_X_str, train_X_num = separate_str_num(train_X)

test_X_str, test_X_num = separate_str_num(test_X)

columns = []
for index in range(len(train_X_str.columns)):
    columns.append('T' + str(index))

train_X_str.columns = columns
test_X_str.columns = columns

train_X_str, test_X_str = feature_vector(train_X_str, test_X_str)

train_X_num, test_X_num = delete_constant(train_X_num, test_X_num)

# train_X_num, test_X_num = rank_feature_by_max(train_X_num, test_X_num)

train_X_num, test_X_num = rank_feature(train_X_num, test_X_num)

train_X_num, test_X_num = rank_feature_count(train_X_num, test_X_num)
#
# train_X_num, test_X_num = nan_statics(train_X_num, test_X_num)
#
train_X_num, test_X_num = delete_all_nan(train_X_num, test_X_num)

train_X = pd.concat([train_X_num, train_X_str], axis=1)
test_X = pd.concat([test_X_num, test_X_str], axis=1)

# print(train_X_str)
# K = range(1, 20)
# meandistortions = []
#
# for k in K:
#     kmeans = KMeans(n_clusters=k)
#     kmeans.fit(train_X_str)
#     meandistortions.append(sum(np.min(cdist(train_X_str, kmeans.cluster_centers_, 'euclidean'), axis=1))/train_X_str.shape[0])
#
# plt.plot(K, meandistortions, 'bx-')
# plt.xlabel('k')
# plt.ylabel('Average Dispersion')
# plt.show()


# train_X = train_X_num

X_train, X_valid, y_train, y_valid = train_test_split(train_X, train_Y, test_size=0.25, random_state=33)
X_test = test_X

lgm = LightGBM()
lgm.run(X_train, y_train, X_valid, y_valid)
important_features = lgm.get_important_features()
train_X_important = train_X.loc[:, important_features]
X_train, X_valid, y_train, y_valid = train_test_split(train_X_important, train_Y, test_size=0.25, random_state=33)
print(len(X_train))
# X_train, X_valid, y_train, y_valid = stratified_sampling(train_X, train_Y)
# X_test = test_X.drop(['nan_standard'], axis=1).values


# lr = LinearR()
# lr.run(X_train, X_valid, y_train, y_valid)

# estimator = PCA(100)
# X_train = estimator.fit_transform(X_train)
# X_valid = estimator.transform(X_valid)
# X_test = estimator.transform(test_X)

# FEATURE_COLUMNS = train_X.columns
# tf_feature_cols = [tf.contrib.layers.real_valued_column(k) for k in FEATURE_COLUMNS]
#
#
# def input_fn(X_train, y_train):
#     feature_cols = {k: tf.constant(X_train[k].values) for k in FEATURE_COLUMNS}
#     labels = tf.constant(y_train.values)
#     return feature_cols, labels
#
# regressor = tf.contrib.learn.DNNRegressor(feature_columns=tf_feature_cols,hidden_units=[64, 128])
#
# regressor.fit(input_fn=lambda: input_fn(X_train, y_train), steps=5000)

# ev = regressor.evaluate(input_fn=lambda: input_fn(X_valid, y_valid), steps=1)
# predict = regressor.predict(input_fn=lambda: input_fn(X_valid, y_valid), as_iterable=False)
# print('ev: {}'.format(ev))
# print('mse:', mean_squared_error(predict, y_valid))


lgm2 = LightGBM()
# lgm.cv(train_X.values, train_Y.values, cv=5)
lgm2.run(X_train, y_train, X_valid, y_valid)

lr = LinearR()
lr.run(X_train, X_valid, y_train, y_valid)

# tdnn =TensorDNN(list(train_X.drop(['nan_standard'], axis=1).columns))
# tdnn.run(X_train, y_train, X_valid, y_valid)

# knr = KNR()
# knr.run(X_train, y_train, X_valid, y_valid)
# knr.predict(X_test)

# gbr = GBR()
# gbr.run(X_train, y_train, X_valid, y_valid)
#
#
# pd_X = pd.DataFrame(X_valid, columns=None)
# pd_Y = pd.DataFrame(y_valid, columns=['Y'])
# pd_valid = pd.concat([pd_X, pd_Y], axis=1)
# pd_valid = pd_valid.sort_values(by='Y')
#
# X_valid = pd_valid.drop(['Y'], axis=1).values
# lgm_pred = lgm.gbm.predict(X_valid)
# knr_pred = knr.knr.predict(X_valid)
# gbr_pred = gbr.gbr.predict(X_valid)
# y_pred = 0.6 * lgm_pred + 0.3*gbr_pred + 0.1*knr_pred
# y_valid_v = pd_valid.Y.values
#
# x = range(len(y_pred))
# plt.plot(x, y_pred, 'r-*', label='y_pred')
# plt.plot(x, y_valid_v, 'b-o', label='y_valid')
# plt.show()
# print("mix mean_squared_error:", mean_squared_error(y_pred, y_valid_v))
#
#
# y_pred = 0.5 * lgm_pred + 0.3*gbr_pred + 0.1*knr_pred + 0.1*predict
# plt.plot(x, y_pred, 'r-*', label='y_pred')
# plt.plot(x, y_valid_v, 'b-o', label='y_valid')
# plt.show()
# print("mix tensorflow mean_squared_error:", mean_squared_error(y_pred, y_valid_v))

