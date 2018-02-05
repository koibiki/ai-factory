import random
import pandas as pd
import numpy as np
import math


def sample_rule(x):
    if np.isnan(x):
        return x
    else:
        random.seed(4)
        return x * (0.97 + 0.06 * random.random())


def create_sample(data):
    randoms = []
    data_sex_age_date = data.iloc[:, :3].reset_index(drop=True)
    data_other = data.iloc[:, 3:].reset_index(drop=True)
    for index in range(len(data_other)):
        random_value = [sample_rule(item) for item in data_other.iloc[index].values]
        random_series = pd.Series(random_value, index=data_other.columns)
        randoms.append(random_series)
    random_data = pd.DataFrame(randoms).reset_index(drop=True)
    return pd.concat([data_sex_age_date, random_data], axis=1)


def fix_min(data):
    for index in range(len(data)):
        column_values = data.iloc[:, index]
        min_value = 0
        if column_values.min() == 0:
            sort = column_values.sort_values()
            for item in sort:
                if item > 0:
                    min_value = item
                    break
        for i in column_values.index:
            if data.ix[i, index] == 0:
                data.ix[i, index] = min_value
    return data


def fix_min_all(data):
    for index in range(data.shape[1]):
        print(index)
        column_values = data.ix[:, index]
        min_value = 0
        if column_values.min() == 0:
            sort = column_values.sort_values()
            for item in sort:
                if item > 0:
                    min_value = item
                    break
        for i in column_values.index:
            if data.ix[i, index] == 0:
                data.ix[i, index] = min_value
    return data


def get_euclidean_metric(vec1, vec2):
    return np.sqrt(np.sum(np.square(vec1 - vec2)))


def get_cosine(vec1, vec2):
    np_vec1, np_vec2 = np.array(vec1), np.array(vec2)
    return np_vec1.dot(np_vec2) / (math.sqrt((np_vec1 ** 2).sum()) * math.sqrt((np_vec2 ** 2).sum()))


def get_cosine_angle(vec1, vec2):
    return math.acos(get_cosine(vec1, vec2)) / math.pi * 180


def combine_all(arr, start, result, index, group_array):
    for ct in range(start, len(arr) - index + 1):
        result[index - 1] = ct
        if index - 1 == 0:
            copy = result.copy()
            copy.reverse()
            group_array.append(copy)
        else:
            combine_all(arr, ct + 1, result, index - 1, group_array)


def logloss_to_class(data, class_level=0.5):
    return [np.math.ceil(x - class_level) for x in data]


def softmax_to_class(data, level=0.5):
    classes = []
    for index in range(len(data)):
        class_type = 0 if np.max(data[index]) < level else np.where(data[index] == np.max(data[index]))[0][0]
        classes.append(class_type)
    return classes
