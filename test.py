from feature_engineering.feature_selector import *
from feature_engineering.fill_nan import *

train = pd.read_csv('./input/train.csv')
train_X = train.iloc[:, 1:-1]
train_Y = train.Y

train_X = delete_constant(train_X)
train_X = delete_nan(train_X)

tool_dict = separate_tool_process(train_X)

tool_dfs = get_tool_dfs(train_X, tool_dict)


