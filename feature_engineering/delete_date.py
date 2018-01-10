import numpy as np
import pandas as pd
import os


train_x_file = 'output/train_X_delete_date.csv'
test_x_file = 'output/test_X_delete_date.csv'


def delete_date(train, test):
    print("执行 Delete Date")
    if os.path.exists(train_x_file) and os.path.exists(test_x_file):
        train_x = pd.read_csv(train_x_file)
        test_x = pd.read_csv(test_x_file)
    else:
        train_date_columns = get_date_columns(train)
        test_date_columns = get_date_columns(test)
        date_columns = mix_date_columns([train_date_columns, test_date_columns])
        train_x = train.drop(date_columns, axis=1)
        test_x = test.drop(date_columns, axis=1)
        train_x.to_csv('output/train_X_delete_date.csv', index=None)
        test_x.to_csv('output/test_X_delete_date.csv', index=None)
    return train_x, test_x


def is_prefix2017(num):
    str_num = str(num)
    return str_num.startswith('2017')


def get_date_columns(self, data):
    columns = data.iloc[0, :].index
    date_column = []
    for row_index in range(len(data)):
        item = data.iloc[row_index]
        for index in range(len(item)):
            if type(item[index]) == np.int64 and self.is_prefix2017(item[index]):
                date_column.append(columns[index])
    return date_column


def mix_date_columns(data_frames):
    date_columns = []
    for data_frame in data_frames:
        for item in data_frame:
            if item not in date_columns:
                date_columns.append(item)
    return date_columns
