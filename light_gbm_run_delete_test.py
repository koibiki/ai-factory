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
from model_select.svm.svm import Svr
from model_select.linear.linear import LinearR

import time
from sklearn.cross_validation import KFold

from sklearn.preprocessing import StandardScaler

train = pd.read_csv('input/train.csv')
test = pd.read_csv('input/testb.csv')
#
train_X = train.iloc[:, 1:-1]
train_Y = train.Y
test_X = test.iloc[:, 1:]

# train_X = train_X.drop(['TOOL (#1)', 'TOOL (#2)'], axis=1)
# test_X = test_X.drop(['TOOL (#1)', 'TOOL (#2)'], axis=1)
#
# train_X_Y =pd.concat([train_X, train_Y], axis=1)
# train_X_Y = train_X_Y[train_X_Y['TOOL_ID'] != 'K']
#
# train_X_Y = train_X_Y[train_X_Y['tool'] != 2409]
#
# train_X_Y = train_X_Y[train_X_Y['tool (#1)'] != 'Q']
# train_X_Y = train_X_Y[train_X_Y['tool (#1)'] != 'R']
# train_X_Y = train_X_Y[train_X_Y['tool (#1)'] != 'P']
#
# train_X = train_X_Y.drop(['Y'], axis=1)
# train_Y = train_X_Y.Y
#
# train_X, test_X = delete_date(train_X, test_X)
#
# train_X_str, train_X_num = separate_str_num(train_X)
#
# test_X_str, test_X_num = separate_str_num(test_X)
#
# columns = []
# for index in range(len(train_X_str.columns)):
#     columns.append('T' + str(index))
#
# train_X_str.columns = columns
# test_X_str.columns = columns
#
# train_X_str, test_X_str = feature_vector(train_X_str, test_X_str)
#
# train_X_num, test_X_num = delete_constant(train_X_num, test_X_num)
#
# train_X_num, test_X_num = rank_feature(train_X_num, test_X_num)
#
# train_X_num, test_X_num = rank_feature_count(train_X_num, test_X_num)
# #
# train_X_num, test_X_num = nan_statics(train_X_num, test_X_num)
# #
# train_X_num, test_X_num = delete_all_nan(train_X_num, test_X_num)
#
# train_X = pd.concat([train_X_num, train_X_str], axis=1)
# test_X = pd.concat([test_X_num, test_X_str], axis=1)
#
# train_X.to_csv('output/train_X_xx.csv', index=None)
# test_X.to_csv('output/test_X_xx.csv', index=None)
train_X = pd.read_csv('output/train_X_xx.csv')
test_X = pd.read_csv('output/test_X_xx.csv')
print(train_X.shape)
print(test_X.shape)


def k_fold_validation(X, Y, model):
    print('开始CV 5折训练...')
    kf = KFold(len(X), n_folds=5, shuffle=True, random_state=33)
    mses = []
    tests_pred = []
    for i, (train_index, test_index) in enumerate(kf):
        print('第{}次训练...'.format(i))
        kf_X_train = X.iloc[train_index]
        kf_y_train = Y.iloc[train_index]
        kf_X_valid = X.iloc[test_index]
        kf_y_valid = Y.iloc[test_index]
        print(kf_X_train.shape)
        print(kf_X_valid.shape)
        model.run(kf_X_train, kf_y_train, kf_X_valid, kf_y_valid)
        kf_y_pred = model.gbm.predict(kf_X_valid)
        kf_y_test = model.gbm.predict(test_X)
        tests_pred.append(kf_y_test)
        mses.append(mean_squared_error(kf_y_pred, kf_y_valid))
        time.sleep(2)
    print('k fold validation:', sum(mses)/len(mses))
    return tests_pred


lgm = LightGBM()
svr = Svr()
tests_pred = k_fold_validation(train_X, train_Y, lgm)

sum_pred = None
for item in tests_pred:
    if sum_pred is None:
        sum_pred = item
    else:
        sum_pred += sum_pred
predict = sum_pred/len(tests_pred)

gbm_a_pattern = pd.read_csv('input/b_pattern.csv', names=['id'])
gbm_a_pattern['Y'] = predict
gbm_a_pattern.to_csv('output/gbm_k_b_pattern.csv', index=None, header=None)


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


# lgm.cv(train_X.values, train_Y.values, cv=5)
# lgm.run(X_train, y_train, X_valid, y_valid)

# lr = LinearR()
# lr.run(X_train, X_valid, y_train, y_valid)

# tdnn =TensorDNN(list(train_X.drop(['nan_standard'], axis=1).columns))
# tdnn.run(X_train, y_train, X_valid, y_valid)

# knr = KNR()
# knr.run(X_train, y_train, X_valid, y_valid)
# knr.predict(X_test)

# gbr = GBR()
# gbr.run(X_train, y_train, X_valid, y_valid)

