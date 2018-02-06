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
    data_target = data.iloc[:, -1]
    data_train = data.iloc[:, :-1]
    for index in range(len(data_train)):
        random_value = [sample_rule(item) for item in data_train.iloc[index].values]
        random_series = pd.Series(random_value, index=data_train.columns)
        randoms.append(random_series)
    random_data = pd.DataFrame(randoms).reset_index(drop=True)
    return pd.concat([random_data, data_target], axis=1)
