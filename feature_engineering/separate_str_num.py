def separate_str_num(data):
    str_columns = get_str_columns(data)
    data_str = data.loc[:, str_columns]
    data_str = num_to_str(data_str)
    data_num = data.drop(str_columns, axis=1)
    return data_str, data_num


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


