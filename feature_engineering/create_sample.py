import random
import pandas as pd
import numpy as np


def sample_rule(x):
    if np.isnan(x):
        return x
    else:
        random.seed(4)
        return x * (0.97 + 0.06 * random.random())


def create_sample(data):
    randoms = []
    data_target = data.iloc[:, -1].reset_index(drop=True)
    data_train = data.iloc[:, :-1]
    str_columns = [column for column in data_train.columns if column.startswith("T") or column.startswith("t")]
    num_columns = [column for column in data_train.columns if column not in str_columns]
    data_str = data.loc[:, str_columns]
    data_num = data.loc[:, num_columns]
    for index in range(len(data_num)):
        random_value = [sample_rule(item) for item in data_num.iloc[index].values]
        random_series = pd.Series(random_value, index=data_num.columns)
        randoms.append(random_series)
    random_data = pd.DataFrame(randoms).reset_index(drop=True)
    return pd.concat([data_str, random_data, data_target], axis=1)
