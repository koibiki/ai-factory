import numpy as np


# 只能处理数值特征 , 需先分离字符和数值 项
def nan_count_statics(data):
    data['nan_count'] = (np.isnan(data)).sum(axis=1)
    data['nan_standard'] = data.nan_count.apply(lambda x: standard_nan(x))
    return data


def nan_statics(train, test):
    print("执行 Nan Statics")
    train_count = train
    test_count = test
    train_count = nan_count_statics(train_count)
    test_count = nan_count_statics(test_count)
    return train_count, test_count


def standard_nan(x):
    if x > 210:
        return 3
    elif x <= 210 and x > 150:
        return 2
    else:
        return 1



