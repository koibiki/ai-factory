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

