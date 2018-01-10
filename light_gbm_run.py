import pandas as pd
import numpy as np
from sklearn.cross_validation import train_test_split

from feature_engineering.delete_date import DeleteDate
from feature_engineering.separate_str_num import SeparateStrNum
from feature_engineering.feature_vector import FeatureVector
from feature_engineering.rank_feature import RankFeature
from feature_engineering.nan_stastics import NanStatics
from feature_engineering.delete_nan import DeleteNan
from model_select.light_gbm.light_gbm import LightGBM

train = pd.read_csv('input/train.csv')
test = pd.read_csv('input/testa.csv')

train_X = train.iloc[:, 1:-1]
train_Y = train.Y
test_X = test.iloc[:, 1:]

dd = DeleteDate()
train_X, test_X = dd.execute(train_X, test_X)

ss = SeparateStrNum()
train_X_str, train_X_num, test_X_str, test_X_num = ss.execute(train_X, test_X)

fv = FeatureVector()
train_X_str, test_X_str = fv.execute(train_X_str, test_X_str)

X_train, X_valid, y_train, y_valid = \
    train_test_split(train_X_num.values, train_Y.values, test_size=0.3, random_state=33)

lgm = LightGBM()
lgm.run(X_train, y_train, X_valid, y_valid)