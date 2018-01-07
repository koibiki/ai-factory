import pandas as pd
import numpy as np

# 读取数据
train = pd.read_csv('../input/train.csv')
testa = pd.read_csv('../input/testa.csv')
testb = pd.read_csv('../input/testb.csv')

# 去除时间特征


def is_prefix2017(num):
    str_num = str(num)
    return str_num.startswith('2017')


def get_date_columns(data):
    columns = data.iloc[0, :].index
    date_column = []
    for row_index in range(len(data)):
        item = data.iloc[row_index]
        for index in range(len(item)):
            if type(item[index]) == np.int64 and is_prefix2017(item[index]):
                date_column.append(columns[index])
    return date_column


train_date_columns = get_date_columns(train)
testa_date_columns = get_date_columns(testa)
testb_date_columns = get_date_columns(testb)

# 合并获得所有时间特征项


def mix_date_columns(train_date_columns, testa_date_columns, testb_date_columns):
    date_columns = []
    for item in train_date_columns:
        if item not in date_columns:
            date_columns.append(item)
    for item in testa_date_columns:
        if item not in date_columns:
            date_columns.append(item)
    for item in testb_date_columns:
        if item not in date_columns:
            date_columns.append(item)
    return date_columns


date_columns = mix_date_columns(train_date_columns, testa_date_columns, testb_date_columns)

train = train.drop(date_columns, axis=1)
testa = testa.drop(date_columns, axis=1)
testb = testb.drop(date_columns, axis=1)

# 输出数据
train.to_csv('../handled_data/train_drop_date.csv', index=None)
testa.to_csv('../handled_data/testa_drop_date.csv', index=None)
testb.to_csv('../handled_data/testb.drop_date.csv', index=None)
