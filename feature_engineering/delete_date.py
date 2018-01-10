import numpy as np
import pandas as pd

from feature_engineering.feature_engineer import FeatureEngineer


class DeleteDate(FeatureEngineer):

    train_x_file = 'output/train_X_delete_date.csv'
    test_x_file = 'output/test_X_delete_date.csv'

    def __init__(self):
        print("执行 Delete Date")

    def execute(self, train, test):
        train_file = open(self.train_x_file)
        test_file = open(self.test_x_file)
        if train_file == None or test_file == None:

        pd.read_csv
        train_date_columns = self.get_date_columns(train)
        test_date_columns = self.get_date_columns(test)
        date_columns = self.mix_date_columns([train_date_columns, test_date_columns])
        train_X = train.drop(date_columns, axis=1)
        test_X = test.drop(date_columns, axis=1)
        train_X.to_csv('output/train_X_delete_date.csv')
        test_X.to_csv('output/test_X_delete_date.csv')
        return train_X, test_X

    @staticmethod
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

    @staticmethod
    def mix_date_columns(data_frames):
        date_columns = []
        for data_frame in data_frames:
            for item in data_frame:
                if item not in date_columns:
                    date_columns.append(item)
        return date_columns
