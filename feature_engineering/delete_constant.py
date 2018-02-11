import pandas as pd


# 删除 常量 特征
def delete_constant(train, test):
    print("执行 Delete Constant")
    train_test = pd.concat([train, test], axis=0)
    constant_series = train_test.max() == train_test.min()
    train_test_index = constant_series.index
    train_test_values = constant_series.values
    constant_columns = []
    for index in range(len(train_test_values)):
        if train_test_values[index]:
            constant_columns.append(train_test_index[index])
    train_test = train_test.drop(constant_columns, axis=1)
    train_delete_constant = train_test.iloc[0:train.shape[0], :]
    test_delete_constant = train_test.iloc[train.shape[0]:, :]
    return train_delete_constant, test_delete_constant
