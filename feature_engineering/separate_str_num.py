from feature_engineering.feature_engineer import FeatureEngineer


class SeparateStrNum(FeatureEngineer):

    def __init__(self):
        print("执行 Separate Str Num")

    def execute(self, train, test):
        str_columns = self.get_str_columns(train)
        train_str = train.loc[:, str_columns]
        train_str = self.num_to_str(train_str)
        train_num = train.drop(str_columns, axis=1)

        test_str = train.loc[:, str_columns]
        test_str = self.num_to_str(test_str)
        test_num = train.drop(str_columns, axis=1)
        return train_str, train_num, test_str, test_num

    @staticmethod
    def get_str_columns(date):
        str_columns = []
        for item in date.columns:
            if item.startswith("T") or item.startswith("t"):
                str_columns.append(item)
        return str_columns

    @staticmethod
    def num_to_str(data):
        columns = data.columns
        for item in columns:
            data[item] = data[item].apply(lambda x: str(x) + "T")
        return data


