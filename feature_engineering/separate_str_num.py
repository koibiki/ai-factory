import pandas as pd
import numpy as np


def get_str_columns(date):
    str_columns = []
    for item in date.columns:
        if item.startswith("T") or item.startswith("t"):
            str_columns.append(item)
    return str_columns


train = pd.read_csv('../handled_data/drop_date/train_drop_date.csv')
testa = pd.read_csv('../handled_data/drop_date/testa_drop_date.csv')
testb = pd.read_csv('../handled_data/drop_date/testb_drop_date.csv')

str_columns = get_str_columns(train)

train_str = train.loc[:, str_columns]
train_num = train.drop(str_columns, axis=1)


def num_to_str(data):
    columns = data.columns
    for item in columns:
        data[item] = data[item].apply(lambda x: str(x) + "T")
    return data


train_str = num_to_str(train_str)
train_str.to_csv('../handled_data/str_num/train_str.csv', index=None)
train_num.to_csv('../handled_data/str_num/train_num.csv', index=None)

