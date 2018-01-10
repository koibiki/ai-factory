def separate_str_num(train, test):
    print("执行 Separate Str Num")
    str_columns = get_str_columns(train)
    train_str = train.loc[:, str_columns]
    train_str = num_to_str(train_str)
    train_num = train.drop(str_columns, axis=1)

    test_str = test.loc[:, str_columns]
    test_str = num_to_str(test_str)
    test_num = test.drop(str_columns, axis=1)
    return train_str, train_num, test_str, test_num


def get_str_columns(date):
    str_columns = []
    for item in date.columns:
        if item.startswith("T") or item.startswith("t"):
            str_columns.append(item)
    return str_columns


def num_to_str(data):
    columns = data.columns
    for item in columns:
        data[item] = data[item].apply(lambda x: str(x) + "T")
    return data


