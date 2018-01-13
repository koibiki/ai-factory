import numpy as np


# 只能处理数值特征 , 需先分离字符和数值 项
def nan_count_statics(data, prefix=''):
    data['nan_count_' + prefix] = (np.isnan(data)).sum(axis=1)
    if data['nan_count_' + prefix].max() == data['nan_count_' + prefix].min() or (data['nan_count_' + prefix].describe()['max'] == 151):
        data = data.drop(['nan_count_' + prefix], axis=1)
        print('nan count 数量相同')
    return data


def nan_statics(train, test):
    print("执行 Nan Statics")
    train_count = train
    test_count = test
    train_count = nan_count_statics(train_count, 'all')
    test_count = nan_count_statics(test_count, 'all')
    train_count['nan_standard'] = train_count['nan_count_all'].apply(lambda x: standard_nan(x))
    test_count['nan_standard'] = test_count['nan_count_all'].apply(lambda x: standard_nan(x))
    return train_count, test_count


def standard_nan(x):
    if x > 160:
        return 3
    elif x <= 160 and x > 120:
        return 2
    else:
        return 1



