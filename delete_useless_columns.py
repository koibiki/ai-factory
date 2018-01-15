import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

from feature_engineering.delete_date import delete_date
from feature_engineering.separate_str_num import separate_str_num
from feature_engineering.feature_vector import feature_vector
from feature_engineering.compute_angle import compute_angle
from feature_engineering.delete_constant import delete_constant
from feature_engineering.rank_feature import rank_feature
from feature_engineering.rank_feature import rank_feature_count
from feature_engineering.nan_stastics import nan_statics
from feature_engineering.delete_nan import delete_all_nan
from model_select.svm.svm import Svr
from sampling.stratified_sampling import stratified_sampling
from sklearn.decomposition import PCA
from model_select.gbr.gradient_boosting import GBR
from model_select.xgboost.xgb import Xgb
from model_select.knn.knn import KNR
from model_select.skflow_dnn.skflow_dnn import TensorDNN
from model_select.light_gbm.light_gbm import LightGBM
from model_select.linear.linear import LinearR
from sklearn.model_selection import KFold

from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import MaxAbsScaler

train_x_file = 'output/train_X_delete_useless.csv'
test_x_file = 'output/test_X_delete_useless.csv'

train = pd.read_csv('input/train.csv')
test = pd.read_csv('input/testb.csv')

train_X = train.iloc[:, 1:-1]
train_Y = train.Y
test_X = test.iloc[:, 1:]

train_X, test_X = delete_date(train_X, test_X)

train_X_Y = pd.concat([train_X, train_Y], axis=1)

train_X_Y = train_X_Y[train_X_Y['tool (#1)'] != 'T']

train_X_Y = train_X_Y[train_X_Y['TOOL_ID (#2)'] != 'E']

train_X_Y = train_X_Y[train_X_Y['TOOL'] != 'C']

train_X_Y = train_X_Y.reset_index()

train_X = train_X_Y.drop(['index', 'Y'], axis=1)
train_Y = train_X_Y.Y

train_X_str, train_X_num = separate_str_num(train_X)
test_X_str, test_X_num = separate_str_num(test_X)

train_X_str = train_X_str.drop(['TOOL (#1)', 'TOOL (#2)'], axis=1)
test_X_str = test_X_str.drop(['TOOL (#1)', 'TOOL (#2)'], axis=1)

columns = []
for index in range(len(train_X_str.columns)):
    columns.append('T' + str(index))

train_X_str.columns = columns
test_X_str.columns = columns

train_X_str, test_X_str = feature_vector(train_X_str, test_X_str)
train_X_num, test_X_num = delete_constant(train_X_num, test_X_num)
train_X_num, test_X_num = rank_feature(train_X_num, test_X_num)
# train_X_num, test_X_num = rank_feature_count(train_X_num, test_X_num)
# train_X_num, test_X_num = nan_statics(train_X_num, test_X_num)
train_X_num, test_X_num = delete_all_nan(train_X_num, test_X_num)
train_X = pd.concat([train_X_num, train_X_str], axis=1)
test_X = pd.concat([test_X_num, test_X_str], axis=1)
# train_X = train_X_num
# test_X = test_X_num


# 0.0294508463994
def k_fold_validation(X, Y):
    print('开始CV 10折训练...')
    kf = KFold(n_splits=10, shuffle=True, random_state=33)
    mses = []

    for i, (train_index, test_index) in enumerate(kf.split(X)):
        print('第{}次训练...'.format(i))
        lgm = LightGBM()
        kf_X_train = X.iloc[train_index]
        kf_y_train = Y.iloc[train_index]
        kf_X_valid = X.iloc[test_index]
        kf_y_valid = Y.iloc[test_index]
        lgm.run(kf_X_train, kf_y_train, kf_X_valid, kf_y_valid)
        kf_y_pred = lgm.predict(kf_X_valid)
        mses.append(mean_squared_error(kf_y_pred, kf_y_valid))
    print('lightgbm k fold validation:', sum(mses)/len(mses))


# k_fold_validation(train_X, train_Y)


# 0.0325479360484
def k_fold_validation_xgb(X, Y):
    print('开始CV 10折训练...')
    kf = KFold(n_splits=10, shuffle=True, random_state=33)
    mses = []

    for i, (train_index, test_index) in enumerate(kf.split(X)):
        print('第{}次训练...'.format(i))
        xgb = Xgb()
        kf_X_train = X.iloc[train_index]
        kf_y_train = Y.iloc[train_index]
        kf_X_valid = X.iloc[test_index]
        kf_y_valid = Y.iloc[test_index]
        xgb.run(kf_X_train, kf_y_train, kf_X_valid, kf_y_valid)
        kf_y_pred = xgb.predict(kf_X_valid)
        mses.append(mean_squared_error(kf_y_pred, kf_y_valid))
    print('xgb k fold validation:', sum(mses)/len(mses))


# k_fold_validation_xgb(train_X, train_Y)


# 0.037264351556
def k_fold_validation_svm(X, Y):
    print('开始CV 10折训练...')
    kf = KFold(n_splits=10, shuffle=True, random_state=33)
    mses = []
    for i, (train_index, test_index) in enumerate(kf.split(X)):
        print('第{}次训练...'.format(i))
        svr = Svr()
        kf_X_train = X.iloc[train_index]
        kf_y_train = Y.iloc[train_index]
        kf_X_valid = X.iloc[test_index]
        kf_y_valid = Y.iloc[test_index]
        svr.run(kf_X_train, kf_y_train, kf_X_valid, kf_y_valid)
        kf_y_pred = svr.predict(kf_X_valid)
        mses.append(mean_squared_error(kf_y_pred, kf_y_valid))
    print('svr k fold validation:', sum(mses)/len(mses))


# k_fold_validation_svm(train_X, train_Y)


# 0.0342181739443
def k_fold_validation_gbr(X, Y):
    print('开始CV 10折训练...')
    kf = KFold(n_splits=10, shuffle=True, random_state=33)
    mses = []
    for i, (train_index, test_index) in enumerate(kf.split(X)):
        print('第{}次训练...'.format(i))
        gbr = GBR()
        kf_X_train = X.iloc[train_index]
        kf_y_train = Y.iloc[train_index]
        kf_X_valid = X.iloc[test_index]
        kf_y_valid = Y.iloc[test_index]
        gbr.run(kf_X_train, kf_y_train, kf_X_valid, kf_y_valid)
        kf_y_pred = gbr.predict(kf_X_valid)
        mses.append(mean_squared_error(kf_y_pred, kf_y_valid))
    print('gbr k fold validation:', sum(mses)/len(mses))


# k_fold_validation_gbr(train_X, train_Y)

# 0.040864561424
def k_fold_validation_knr(X, Y):
    print('开始CV 10折训练...')
    kf = KFold(n_splits=10, shuffle=True, random_state=33)
    mses = []
    for i, (train_index, test_index) in enumerate(kf.split(X)):
        print('第{}次训练...'.format(i))
        knr = KNR()
        kf_X_train = X.iloc[train_index]
        kf_y_train = Y.iloc[train_index]
        kf_X_valid = X.iloc[test_index]
        kf_y_valid = Y.iloc[test_index]
        knr.run(kf_X_train, kf_y_train, kf_X_valid, kf_y_valid)
        kf_y_pred = knr.predict(kf_X_valid)
        mses.append(mean_squared_error(kf_y_pred, kf_y_valid))
    print('knr k fold validation:', sum(mses)/len(mses))


# k_fold_validation_knr(train_X, train_Y)


# 0.0546840900518
def k_fold_validation_lr(X, Y):
    print('开始CV 10折训练...')
    kf = KFold(n_splits=10, shuffle=True, random_state=33)
    mses = []
    for i, (train_index, test_index) in enumerate(kf.split(X)):
        print('第{}次训练...'.format(i))
        lr = LinearR()
        kf_X_train = X.iloc[train_index]
        kf_y_train = Y.iloc[train_index]
        kf_X_valid = X.iloc[test_index]
        kf_y_valid = Y.iloc[test_index]
        lr.run(kf_X_train, kf_y_train, kf_X_valid, kf_y_valid)
        kf_y_pred = lr.predict(kf_X_valid)
        mses.append(mean_squared_error(kf_y_pred, kf_y_valid))
    print('lr k fold validation:', sum(mses)/len(mses))


# k_fold_validation_lr(train_X, train_Y)

X_train, X_valid, y_train, y_valid = train_test_split(train_X, train_Y, test_size=0.25, random_state=33)

tdnn = TensorDNN()
tdnn.run(X_train, y_train, X_valid, y_valid)

# FEATURE_COLUMNS = train_X.columns
# tf_feature_cols = [tf.contrib.layers.real_valued_column(k) for k in FEATURE_COLUMNS]
#
#
# mms = MaxAbsScaler()
# X_train = mms.fit_transform(X_train)
# X_valid = mms.transform(X_valid)
#
# X_train = pd.DataFrame(X_train, columns=train_X.columns)
# X_valid = pd.DataFrame(X_valid, columns=train_X.columns)
#
# def input_fn(X_train, y_train):
#     feature_cols = {k: tf.constant(X_train[k].values) for k in FEATURE_COLUMNS}
#     labels = tf.constant(y_train.values)
#     return feature_cols, labels
#
# regressor = tf.contrib.learn.DNNRegressor(feature_columns=tf_feature_cols,hidden_units=[64, 128])
#
# regressor.fit(input_fn=lambda: input_fn(X_train, y_train), steps=5000)
#
# ev = regressor.evaluate(input_fn=lambda: input_fn(X_valid, y_valid), steps=1)
# predict = regressor.predict(input_fn=lambda: input_fn(X_valid, y_valid), as_iterable=False)
# print('ev: {}'.format(ev))
# print('mse:', mean_squared_error(predict, y_valid))