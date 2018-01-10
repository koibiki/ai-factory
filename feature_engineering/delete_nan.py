import numpy as np

from feature_engineering.feature_engineer import FeatureEngineer


class DeleteNan(FeatureEngineer):

    def __init__(self):
        print("执行 Delete nan")

    # 只能处理数值特征 , 需先分离字符和数值 项
    def execute(self, train, test):
        train_nan_columns = np.where(np.isnan(train))[1]
        test_nan_columns = np.where(np.isnan(test))[1]
        all_nan_columns = self.mix(train_nan_columns, test_nan_columns)
        train_delete_nan = train.drop(train.columns[all_nan_columns], axis=1)
        test_delete_nan = test.drop(test.columns[all_nan_columns], axis=1)
        return train_delete_nan, test_delete_nan

    @staticmethod
    def get_nan_indexes(data):
        indexes = []
        for index in data:
            if index not in indexes:
                indexes.append(index)
        return indexes

    def mix(self, train, test):
        train_indexes = self.get_nan_indexes(train)
        test_indexes = self.get_nan_indexes(test)
        nan_indexes = list(train_indexes) + list(test_indexes)
        return nan_indexes
