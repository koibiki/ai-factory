def fillna_by_median(nan_index, df):
    for index in nan_index:
        df.iloc[index] = df.iloc[index].fillna(df[df.iloc[:, 0] == df.iloc[index].iloc[0]].median())
    return df


def get_tool_dfs(df, tool_dict):
    tool_dfs = {}
    for tool in tool_dict.keys():
        print('execute ' + tool)
        tool_columns = [tool] + tool_dict[tool]
        tool_df = df[tool_columns]
        tool_df[tool + '_nan'] = tool_df.isnull().sum(axis=1)
        if tool_df[tool + '_nan'].max() > tool_df[tool + '_nan'].min():
            nan_index = tool_df[tool_df[tool + '_nan'] > 0].index
            #tool_df = fillna_by_median(nan_index, tool_df)
        else:
            tool_df = tool_df.drop([tool + '_nan'], axis=1)
        tool_dfs[tool] = tool_df
    return tool_dfs
