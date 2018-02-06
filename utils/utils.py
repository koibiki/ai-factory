import random
import pandas as pd
import numpy as np
import math


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


def calculate_mean(preds):
    sum_pred = np.zeros(len(preds[0]))
    for item in preds:
        sum_pred += np.array(item)
    return sum_pred/len(preds)


def calculate_multi_mean(preds):
    new_preds = []
    for index in range(len(preds[0])):
        sum_pred = np.zeros(len(preds[0][0]))
        for i in range(len(preds)):
            sum_pred += np.array(preds[i][index])
        new_preds.append(sum_pred/len(preds))
    return new_preds


def calculate_importance_feature(importances):
    all_feature = []
    for features in importances:
        for feature in features.index:
            if feature not in all_feature:
                all_feature.append(feature)
    return all_feature


def cv_important_feature(importances):
    all_feature = calculate_importance_feature(importances)
    non_important_feature = []
    for feature in all_feature:
        count = 0
        for important_feature in importances:
            if feature not in important_feature.index:
                count += 1
        if count >= 3:
            non_important_feature.append(feature)

    return [feature for feature in all_feature if feature not in non_important_feature]
