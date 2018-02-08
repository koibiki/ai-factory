from feature_engineering.feature_selector import *
from feature_engineering.create_sample import *
from feature_engineering.separate_str_num import *
from feature_engineering.fill_nan import *
from model_selection.regressor_model_factory import RegressorModelFactory
from model_selection.cv import k_fold_regressor
from model_selection.smote_cv import *
from sklearn.model_selection import train_test_split

train = pd.read_csv('./input/train.csv',)
train_X = train.iloc[:, 1:-1]
train_Y = train.Y

train_X = delete_constant(train_X)
train_X = delete_nan(train_X)

data_num, data_str = separate_num_str(train_X)

data_str = pd.get_dummies(data_str)

train_data = pd.concat([ data_num], axis=1)

print(train_data.shape)

predict, cv_indexs, importances = \
    k_fold_regressor(train_data, train_Y, train_data, model_num=RegressorModelFactory.MODEL_LIGHET_GBM, cv=10, important_level=0)

importances_train = train_data[importances]

print(importances_train.shape)

pd.DataFrame(importances, columns=['importance']).to_csv('./output/important_feature.csv', index=None)

k_fold_regressor(importances_train, train_Y, importances_train, model_num=RegressorModelFactory.MODEL_LIGHET_GBM, cv=5)

