import pandas as pd


# 与 rank by max 择一执行
def rank_feature(train, test):
    print("执行 Rank Feature")
    train_test = pd.concat([train, test], axis=0)
    train_test_rank = train_test.rank(method='max')
    rank_count = train_test_rank.shape[0]
    train_test_rank[train_test_rank <= rank_count / 10] = 1
    train_test_rank[(rank_count / 10 < train_test_rank) & (train_test_rank <= rank_count / 5)] = 2
    train_test_rank[(rank_count / 5 < train_test_rank) & (train_test_rank <= rank_count * 3 / 10)] = 3
    train_test_rank[(rank_count * 3 / 10 < train_test_rank) & (train_test_rank <= rank_count * 2 / 5)] = 4
    train_test_rank[(rank_count * 2 / 5 < train_test_rank) & (train_test_rank <= rank_count * 1 / 2)] = 5
    train_test_rank[(rank_count * 1 / 2 < train_test_rank) & (train_test_rank <= rank_count * 3 / 5)] = 6
    train_test_rank[(rank_count * 3 / 5 < train_test_rank) & (train_test_rank <= rank_count * 7 / 10)] = 7
    train_test_rank[(rank_count * 7 / 10 < train_test_rank) & (train_test_rank <= rank_count * 4 / 5)] = 8
    train_test_rank[(rank_count * 4 / 5 < train_test_rank) & (train_test_rank <= rank_count * 9 / 10)] = 9
    train_test_rank[rank_count * 9 / 10 < train_test_rank] = 10

    train_num_rank = train_test_rank.iloc[0:train.shape[0], :]
    test_num_rank = train_test_rank.iloc[train.shape[0]:, :]
    return train_num_rank, test_num_rank


def rank_feature_count(train, test):
    train_test = pd.concat([train, test], axis=0)
    for i in range(1, 11, 1):
        train_test['n' + str(i)] = (train_test == i).sum(axis=1)
    train_rank_count = train_test.iloc[0:train.shape[0], :]
    test_rank_count = train_test.iloc[train.shape[0]:, :]
    return train_rank_count, test_rank_count
