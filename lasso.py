from feature_engineering.feature_selector import *
from feature_engineering.separate_str_num import *
from model_selection.cv import k_fold_regressor
from model_selection.smote_cv import *

train = pd.read_csv('./input/train.csv')
test = pd.read_csv('./input/testa.csv')
train_X = train.iloc[:, 1:-1]
train_Y = train.Y
test_X = test.iloc[:, 1:]

train_X = delete_constant(train_X)
train_X = delete_nan(train_X)
train_X = delete_duplicates(train_X)

data_num, data_str = separate_num_str(train_X)


train_data = pd.concat([data_num], axis=1)
test_data = test.loc[:, train_data.columns]

train_data_origin = train_data.copy()
test_data_origin = test_data.copy()

train_data.fillna(train_data.median(axis=0), inplace=True)
test_data.fillna(train_data.median(axis=0), inplace=True)

print(train_data.isnull().sum(axis=0).sort_values(ascending=False))

predict, cv_indexs, importances = \
    k_fold_regressor(train_data, train_Y, test_data, model_num=RegressorModelFactory.MODEL_LASSO, cv=10, important_level=0)

test_Y = pd.read_csv('./input/testa_anwser.csv', header=None, names=['id', 'Y'])

print(mean_squared_error(test_Y.iloc[:, -1], predict))

importances_train = train_data[importances]

print(importances_train.shape)

importance_test = test_data.loc[:, importances_train.columns]

predict, cv_indexs, importances2 =\
    k_fold_regressor(importances_train, train_Y, importance_test, model_num=RegressorModelFactory.MODEL_LASSO, cv=5)

print(mean_squared_error(test_Y.iloc[:, -1], predict))

rmf = RegressorModelFactory()

ls = rmf.create_model(RegressorModelFactory.MODEL_LASSO)

ls.fit(importances_train, importance_test, train_Y, test_Y.iloc[:, -1])

pred = ls.predict(importance_test)

print('lasso mse:', mean_squared_error(test_Y.iloc[:, -1], pred))

ls = rmf.create_model(RegressorModelFactory.MODEL_LIGHET_GBM)

ls.fit(importances_train, importance_test, train_Y, test_Y.iloc[:, -1])

pred = ls.predict(importance_test)

print('light_gbm mse:', mean_squared_error(test_Y.iloc[:, -1], pred))

importances_train = train_data_origin[importances]
importance_test = test_data_origin[importances]

ls = rmf.create_model(RegressorModelFactory.MODEL_LIGHET_GBM)

ls.fit(importances_train, importance_test, train_Y, test_Y.iloc[:, -1])

pred = ls.predict(importance_test)

print('light_gbm mse:', mean_squared_error(test_Y.iloc[:, -1], pred))