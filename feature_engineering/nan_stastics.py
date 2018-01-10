import numpy as np


# 只能处理数值特征 , 需先分离字符和数值 项
def nan_statics(train, test):
    print("执行 Nan Statics")
    train_count = train
    test_count = test
    train_count['nan_count'] = (np.isnan(train)).sum(axis=1)
    test_count['nan_count'] = (np.isnan(test)).sum(axis=1)
    train_count['nan_standard'] = train.nan_count.apply(lambda x: standard_nan(x))
    test_count['nan_standard'] = test.nan_count.apply(lambda x: standard_nan(x))
    return train_count, test_count


def standard_nan(x):
    if x > 210:
        return 3
    elif x <= 210 and x > 150:
        return 2
    else:
        return 1



