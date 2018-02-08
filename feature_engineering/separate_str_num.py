def separate_num_str(df):
    str_columns = get_str_columns(df)
    num_columns = [column for column in df.columns if column not in str_columns]
    return df.loc[:, num_columns], num_to_str(df.loc[:, str_columns])


def get_str_columns(df):
    return [column for column in df.columns if column.startswith("T") or column.startswith("t") or column.startswith("O")]


def num_to_str(data):
    columns = data.columns
    for item in columns:
        data.loc[:, item] = data.loc[:, item].apply(lambda x: str(x) + "T")
    return data


