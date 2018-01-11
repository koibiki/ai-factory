import pandas as pd

from sklearn.cross_validation import train_test_split


def stratified_sampling(train_x, train_y):
    train_x_y = pd.concat([train_x, train_y], axis=1)
    train_x_y_1 = train_x_y[train_x_y.nan_standard == 1]
    x_data_1 = train_x_y_1.drop(['nan_standard', 'Y'], axis=1)
    y_data_1 = train_x_y_1.Y
    x_train_1, x_valid_1, y_train_1, y_valid_1 = \
        train_test_split(x_data_1.values, y_data_1.values, test_size=0.2, random_state=33)
    print(x_data_1.shape, " " + str(y_data_1.shape))

    train_x_y_2 = train_x_y[train_x_y.nan_standard == 2]
    x_data_2 = train_x_y_2.drop(['nan_standard', 'Y'], axis=1)
    y_data_2 = train_x_y_2.Y
    x_train_2, x_valid_2, y_train_2, y_valid_2 = \
        train_test_split(x_data_2.values, y_data_2.values, test_size=0.2, random_state=33)
    print(x_data_2.shape, " " + str(y_data_2.shape))

    train_x_y_3 = train_x_y[train_x_y.nan_standard == 3]
    x_data_3 = train_x_y_3.drop(['nan_standard', 'Y'], axis=1)
    y_data_3 = train_x_y_3.Y
    x_train_3, x_valid_3, y_train_3, y_valid_3 = \
        train_test_split(x_data_3.values, y_data_3.values, test_size=0.2, random_state=33)

    x_train = list(x_train_1) + list(x_train_2) + list(x_train_3)
    y_train = list(y_train_1) + list(y_train_2) + list(y_train_3)
    print("x_train1:" + str(len(x_train_1)) + "  x_train2:" + str(len(x_train_2))+"  x_train3:" + str(len(x_train_3)))
    x_valid = list(x_valid_1) + list(x_valid_2) + list(x_valid_3)
    y_valid = list(y_valid_1) + list(y_valid_2) + list(y_valid_3)
    print("x_valid_1:" + str(len(x_valid_1)) + "  x_valid_2:" + str(len(x_valid_2))+"  x_valid_3:" + str(len(x_valid_3)))

    return x_train, x_valid, y_train, y_valid
