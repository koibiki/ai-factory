import pandas as pd


# 需要在 rank 之前运行
def rank_feature_by_max(train, test):
    print("执行 Rank Feature by max")
    train_test = pd.concat([train, test], axis=0)
    train_test_rank_standard = (train_test - train_test.min())/(train_test.max() - train_test.min())
    train_test_rank_standard[train_test_rank_standard <= 0.1] = 1
    train_test_rank_standard[(0.1 < train_test_rank_standard) & (train_test_rank_standard <= 0.2)] = 2
    train_test_rank_standard[(0.2 < train_test_rank_standard) & (train_test_rank_standard <= 0.3)] = 3
    train_test_rank_standard[(0.3 < train_test_rank_standard) & (train_test_rank_standard <= 0.4)] = 4
    train_test_rank_standard[(0.4 < train_test_rank_standard) & (train_test_rank_standard <= 0.5)] = 5
    train_test_rank_standard[(0.5 < train_test_rank_standard) & (train_test_rank_standard <= 0.6)] = 6
    train_test_rank_standard[(0.6 < train_test_rank_standard) & (train_test_rank_standard <= 0.7)] = 7
    train_test_rank_standard[(0.7 < train_test_rank_standard) & (train_test_rank_standard <= 0.8)] = 8
    train_test_rank_standard[(0.8 < train_test_rank_standard) & (train_test_rank_standard <= 0.9)] = 9
    train_test_rank_standard[(0.9 < train_test_rank_standard) & (train_test_rank_standard <= 1.0)] = 10

    for i in range(1, 11, 1):
        train_test_rank_standard['n' + str(i)] = (train_test_rank_standard == i).sum(axis=1)

    train_num_rank = train_test_rank_standard.iloc[0:train.shape[0], :]
    test_num_rank = train_test_rank_standard.iloc[train.shape[0]:, :]
    return train_num_rank, test_num_rank
