import pandas as pd
import numpy as np
import math


def compute_angle(train, test):
    print("执行 Compute Angle")
    train_test = pd.concat([train, test], axis=0)
    angles = angle_classify(train_test)
    angles_standard = angle_standard(angles)
    angle_rank = rank_angle(angles_standard)
    train_test['angle_rank'] = angle_rank
    train_test.angle_rank = train_test.angle_rank.apply(lambda x: rank_angle(x))
    train_angle = train_test.iloc[0:train.shape[0], :]
    test_angle = train_test.iloc[train.shape[0]:, :]
    return train_angle, test_angle


def rank_angle(x):
    if x <= 0.1:
        x = 1
    elif x > 0.1 and x <= 0.2:
        x = 2
    elif x > 0.2 and x <= 0.3:
        x = 3
    elif x > 0.3 and x <= 0.4:
        x = 4
    elif x > 0.4 and x <= 0.5:
        x = 5
    elif x > 0.5 and x <= 0.6:
        x = 6
    elif x > 0.6 and x <= 0.7:
        x = 7
    elif x > 0.7 and x <= 0.8:
        x = 8
    elif x > 0.8 and x <= 0.9:
        x = 9
    else:
        x = 10
    return x


def angle_classify(data):
    angles = []
    for index in range(len(data)):
        cos_value = cos(list(data.iloc[0, :].values), list(data.iloc[index, :].values))
        angles.append(cos_value)
    return angles


def angle_classify_trainsfer(vector, data):
    angles = []
    for index in range(len(data)):
        cos_value = cos(list(vector.values), list(data.iloc[index, :].values))
        angles.append(math.acos(cos_value) * 180 / math.pi)
    return angles


def angle_standard(angles):
    max_value = np.max(angles)
    min_value = np.min(angles)
    angle_rank = []
    for item in angles:
        value = (item - min_value)/(max_value - min_value)
        angle_rank.append(value)
    return angle_rank


def cos(vector1, vector2):
    dot_product = 0.0
    normA = 0.0
    normB = 0.0
    for a, b in zip(vector1, vector2):
        dot_product += a*b
        normA += a**2
        normB += b**2
    if normA == 0.0 or normB == 0.0:
        return None
    else:
        return dot_product / ((normA*normB)**0.5)
