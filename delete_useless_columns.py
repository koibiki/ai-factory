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
from sklearn.cross_validation import KFold

train_x_file = 'output/train_X_delete_useless.csv'
test_x_file = 'output/test_X_delete_useless.csv'

train = pd.read_csv('input/train.csv')
test = pd.read_csv('input/testa.csv')

train_X = train.iloc[:, 1:-1]
train_Y = train.Y
test_X = test.iloc[:, 1:]
#
# train_X, test_X = delete_date(train_X, test_X)
#
# train_X_Y = pd.concat([train_X, train_Y], axis=1)
#
# train_X_Y = train_X_Y[train_X_Y['tool (#1)']!='T'] # -17
#
# train_X_Y = train_X_Y[train_X_Y['TOOL_ID (#2)']!='E'] # -166
#
# train_X_Y = train_X_Y[train_X_Y['TOOL']!='C'] # -42
#
# train_X = train_X_Y.drop(['Y'], axis=1)
# train_Y = train_X_Y.Y
#
# train_X_str, train_X_num = separate_str_num(train_X)
# test_X_str, test_X_num = separate_str_num(test_X)
#
# train_X_str = train_X_str.drop(['TOOL (#1)','TOOL (#2)'], axis=1)
# test_X_str = test_X_str.drop(['TOOL (#1)','TOOL (#2)'], axis=1)
#
# columns = []
# for index in range(len(train_X_str.columns)):
#     columns.append('T' + str(index))
#
# train_X_str.columns = columns
# test_X_str.columns = columns
#
# train_X_str, test_X_str = feature_vector(train_X_str, test_X_str)
# train_X_num, test_X_num = delete_constant(train_X_num, test_X_num)
# train_X_num, test_X_num = rank_feature(train_X_num, test_X_num)
# train_X_num, test_X_num = rank_feature_count(train_X_num, test_X_num)
# train_X_num, test_X_num = nan_statics(train_X_num, test_X_num)
# train_X_num, test_X_num = delete_all_nan(train_X_num, test_X_num)
# train_X = pd.concat([train_X_num, train_X_str], axis=1)
# test_X = pd.concat([test_X_num, test_X_str], axis=1)
#
# train_X.to_csv('output/train_X_delete_useless.csv', index=None)
# test_X.to_csv('output/test_X_delete_useless.csv', index=None)

train_X = pd.read_csv(train_x_file)
test_X = pd.read_csv(test_x_file)

def k_fold_validation(X, Y, model):
    print('开始CV 5折训练...')
    kf = KFold(len(train_X), n_folds = 10, shuffle=True, random_state=33)
    mses = []
    for i, (train_index, test_index) in enumerate(kf):
        print('第{}次训练...'.format(i))
        kf_X_train = train_X.iloc[train_index]
        kf_y_train = train_Y.iloc[train_index]
        kf_X_valid = train_X.iloc[test_index]
        kf_y_valid = train_Y.iloc[test_index]
        model.run(kf_X_train, kf_y_train, kf_X_valid, kf_y_valid)
        kf_y_pred = lgm.gbm.predict(kf_X_valid)
        mses.append(mean_squared_error(kf_y_pred, kf_y_valid))
    print('k fold validation:',sum(mses)/len(mses))

lgm = LightGBM()
k_fold_validation(train_X, train_Y,lgm)