import pandas as pd
import numpy as np


def delete_constant(df):
    columns = df.columns
    non_constant_columns = [column for column in columns if df[column].max() != df[column].min()]
    return df[non_constant_columns]


def delete_nan(df):
    null_num = df.isnull().sum(axis=0)/len(df)
    columns = null_num[null_num < 0.2].index
    return df[columns]


def separate_date_feature(df):
    date_columns = get_date_columns(df)
    return df.drop([date_columns], axis=1), df[date_columns]


def separate_tool_process(df):
    tool_dict = {}
    current_column = None
    for column in df.columns:
        if column.startswith("T") or column.startswith("t"):
            tool_dict[column] = []
            current_column = column
        else:
            tool_dict[current_column].append(column)
    return tool_dict


def is_prefix2017(num):
    return str(num).startswith('2017')


def get_date_columns(df):
    columns = df.iloc[0, :].index
    date_column = []
    for row_index in range(len(df)):
        item = df.iloc[row_index]
        for index in range(len(item)):
            if type(item[index]) == np.int64 and is_prefix2017(item[index]):
                date_column.append(columns[index])
    return date_column
