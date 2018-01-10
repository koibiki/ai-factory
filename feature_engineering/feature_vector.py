import pandas as pd
import numpy as np

from feature_engineering.feature_engineer import FeatureEngineer


class FeatureVector(FeatureEngineer):

    def __init__(self):
        print("执行 Feature Vector")

    def execute(self, train, test):
        train_dummies= pd.get_dummies(train)
        test_dummies = pd.get_dummies(test)
        test_dummies = self.delete_feature(train_dummies, test_dummies)
        test_dummies = self.feature_align(train_dummies, test_dummies)
        return train_dummies, test_dummies

    @staticmethod
    def delete_feature(train, test):
        train_columns = train.columns
        test_columns = test.columns
        for item in test_columns:
            if item not in train_columns:
                test = test.drop(item, axis=1)
        return test

    @staticmethod
    def feature_align(train, test):
        train_columns = train.columns.values
        for index in range(len(train_columns)):
            column = train_columns[index]
            if column != test.iloc[:, index].name:
                test.insert(index, column, np.zeros(len(test)))
        return test
