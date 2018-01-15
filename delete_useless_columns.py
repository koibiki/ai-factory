import pandas as pd
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
from sampling.stratified_sampling import stratified_sampling
from sklearn.decomposition import PCA
from model_select.model_factory import ModelFactory
from model_select.cv import k_fold_validation


import matplotlib.pyplot as plt


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

# X_train, X_valid, y_train, y_valid = train_test_split(train_X, train_Y, test_size=0.1, random_state=33)
# pd_valid = pd.concat([X_valid, y_valid], axis=1)
# pd_valid = pd_valid.sort_values(by='Y')
# X_valid = pd_valid.drop(['Y'], axis=1)
# y_valid = pd_valid.Y

print(test_X.shape)

cv_pred, cv_mse = k_fold_validation(train_X, train_Y, test_X, ModelFactory.MODEL_TENSOR_LR, cv=5)
print(cv_mse)
# x = range(len(cv_pred))
# plt.plot(x, cv_pred, 'r-*', label='y_pred')
# plt.plot(x, y_valid, 'b-o', label='y_valid')
# plt.legend()
# plt.show()
#
# print(mean_squared_error(cv_pred, y_valid))

# light gbm 0.032688
# xgboost   0.037545
# gbr       0.035062
# svr       0.037691
# knr       0.041189
# lr        0.053019
# dnn       0.0462087