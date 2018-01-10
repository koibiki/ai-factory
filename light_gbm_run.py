import pandas as pd
from sklearn.cross_validation import train_test_split

from feature_engineering.delete_date import delete_date
from feature_engineering.separate_str_num import separate_str_num
from feature_engineering.feature_vector import feature_vector
from feature_engineering.rank_feature import rank_feature
from feature_engineering.nan_stastics import nan_statics
from feature_engineering.delete_nan import delete_nan
from sampling.stratified_sampling import stratified_sampling
from sklearn.decomposition import PCA
from model_select.gbr.gradient_boosting import GBR
from model_select.knn.knn import KNR
from model_select.light_gbm.light_gbm import LightGBM

train = pd.read_csv('input/train.csv')
test = pd.read_csv('input/testa.csv')

train_X = train.iloc[:, 1:-1]
train_Y = train.Y
test_X = test.iloc[:, 1:]

train_X, test_X = delete_date(train_X, test_X)

train_X_str, train_X_num, test_X_str, test_X_num = separate_str_num(train_X, test_X)

train_X_str, test_X_str = feature_vector(train_X_str, test_X_str)

train_X_num, test_X_num = rank_feature(train_X_num, test_X_num)
#
train_X_num, test_X_num = nan_statics(train_X_num, test_X_num)
#
train_X_num, test_X_num = delete_nan(train_X_num, test_X_num)

train_X = pd.concat([train_X_num, train_X_str], axis=1)
print(train_X.shape)
test_X = pd.concat([test_X_num, test_X_str], axis=1)
print(test_X.shape)

# train_X = train_X_num

# X_train, X_valid, y_train, y_valid = train_test_split(train_X.values, train_Y.values, test_size=0.25, random_state=33)

X_train, X_valid, y_train, y_valid = stratified_sampling(train_X, train_Y)
X_test = test_X.drop(['nan_standard'], axis=1).values

print()

estimator = PCA(150)
X_train = estimator.fit_transform(X_train)
X_valid = estimator.transform(X_valid)
X_test = estimator.transform(X_test)

lgm = LightGBM()
lgm.run(X_train, y_train, X_valid, y_valid)
lgm.predict(X_test)

knr = KNR()
knr.run(X_train, y_train, X_valid, y_valid)
knr.predict(X_test)
#
gbr = GBR()
gbr.run(X_train, y_train, X_valid, y_valid)
