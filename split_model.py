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
from feature_engineering.nan_stastics import nan_count_statics
from feature_engineering.delete_nan import delete_all_nan
from feature_engineering.pipeline_split import split_by_tool
from sampling.stratified_sampling import stratified_sampling
from sklearn.decomposition import PCA
from model_select.gbr.gradient_boosting import GBR
from model_select.xgboost.xgb import Xgb
from model_select.knn.knn import KNR
from model_select.skflow_dnn.skflow_dnn import TensorDNN
from model_select.light_gbm.light_gbm import LightGBM
from model_select.linear.linear import LinearR
from model_select.svm.svm import Svr
import time
from sklearn.cross_validation import KFold

from sklearn.preprocessing import StandardScaler

train = pd.read_csv('input/train.csv')
test = pd.read_csv('input/testb.csv')

train_X = train.iloc[:, 1:-1]
train_Y = train.Y
test_X = test.iloc[:, 1:]

trains = split_by_tool(train_X)
tests = split_by_tool(test_X)

# 删除常数
for index in range(len(trains)):
    trains[index], tests[index] = delete_constant(trains[index], tests[index])

# 删除时间
for index in range(len(trains)):
    trains[index], tests[index] = delete_date(trains[index], tests[index])

# 分离 字符 和 数字
trains_str = []
tests_str = []
trains_num = []
tests_num = []
for index in range(len(trains)):
    train_str, train_num = separate_str_num(trains[index])
    test_str, test_num = separate_str_num(tests[index])
    train_str.columns = ['T'+ str(index)]
    test_str.columns = ['T'+ str(index)]
    train_str, test_str = feature_vector(train_str, test_str)
    trains_str.append(train_str)
    trains_num.append(train_num)
    tests_str.append(test_str)
    tests_num.append(test_num)

# rank feature
for index in range(len(trains)):
    trains_num[index], tests_num[index] = rank_feature(trains_num[index], tests_num[index])

# rank feature count
# for index in range(len(trains)):
#     trains_num[index], tests_num[index] = rank_feature_count(trains_num[index], tests_num[index])

# statics nan
# for index in range(len(trains)):
#     trains_num[index] = nan_count_statics(trains_num[index], str(index))
#     tests_num[index] = nan_count_statics(tests_num[index], str(index))

# delete nan
for index in range(len(trains)):
    trains_num[index], tests_num[index] = delete_all_nan(trains_num[index], tests_num[index])

trains_data = []
for index in range(len(trains)):
    train_data = pd.concat([trains_str[index], trains_num[index]], axis=1)
    trains_data.append(train_data)


def k_fold_validation(X, Y):
    print('开始CV 5折训练...')
    lgm = Svr()
    k = 5
    kf = KFold(len(X), n_folds=k, shuffle=True, random_state=33)
    mses = []
    ky_y_pred_sum = None
    for i, (train_index, test_index) in enumerate(kf):
        print('第{}次训练...'.format(i))
        kf_X_train = X.iloc[train_index]
        kf_y_train = Y.iloc[train_index]
        kf_X_valid = X.iloc[test_index]
        kf_y_valid = Y.iloc[test_index]
        kf_y_pred, mse = lgm.run(kf_X_train, kf_y_train, kf_X_valid, kf_y_valid)
        if ky_y_pred_sum is None:
            ky_y_pred_sum = kf_y_pred
        else:
            ky_y_pred_sum += kf_y_pred
        mses.append(mse)
    print('k fold validation:', sum(mses)/len(mses))
    return ky_y_pred_sum/k


kf_y_preds = {}
for index in range(len(trains)):
    print('train:' + str(index))
    y_k = k_fold_validation(trains_data[index], train_Y)
    kf_y_preds[str(index)] = y_k

