from feature_engineering.feature_selector import *
from feature_engineering.create_sample import *
from feature_engineering.separate_str_num import *
from feature_engineering.fill_nan import *
from model_selection.regressor_model_factory import RegressorModelFactory
from model_selection.cv import k_fold_regressor
from model_selection.smote_cv import k_fold_smote_regressor
from sklearn.model_selection import train_test_split

train = pd.read_csv('./input/train.csv')
train_X = train.iloc[:, 1:-1]
train_Y = train.Y

train_X = delete_constant(train_X)
train_X = delete_nan(train_X)

data_str, data_num = separate_num_str(train_X)