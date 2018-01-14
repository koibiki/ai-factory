import lightgbm as lgm
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error

from model_select.predict_model import PredictModel


class LightGBM(PredictModel):
    gbm = None
    sort_valid = None
    important_feature = []

    def create_predict_model(self):
        self.gbm = lgm.LGBMRegressor(objective='regression',
                                     boosting_type='gbdt',
                                     metric='l2',
                                     alpha=0.9,
                                     n_estimators=500,
                                     learning_rate=0.05,
                                     max_depth=6,
                                     # bagging_fraction=0.7,
                                     # bagging_freq=2,
                                     num_leaves=64,
                                     subsample=0.8)

    def run(self, X_train, y_train, X_valid, y_valid):
        self.create_predict_model()

        self.gbm.fit(X_train, y_train,
                     eval_set=[(X_valid, y_valid)],
                     eval_metric='l2',
                     early_stopping_rounds=200)
        y_pred = self.gbm.predict(X_valid)
        print("gbm mean_squared_error:", mean_squared_error(y_pred, y_valid))

        # x = range(len(y_pred))
        # plt.plot(x, y_pred, 'r-*', label='y_pred')
        # plt.plot(x, y_valid, 'b-o', label='y_valid')
        # plt.show()
        #
        # pd_X = pd.DataFrame(X_valid, columns=None)
        # pd_Y = pd.DataFrame(y_valid, columns=['Y'])
        # pd_valid = pd.concat([pd_X, pd_Y], axis=1)
        # pd_valid = pd_valid.sort_values(by='Y')
        #
        # X_valid = pd_valid.drop(['Y'], axis=1).values
        # y_pred = self.gbm.predict(X_valid)
        # y_valid = pd_valid.Y.values
        #
        # x = range(len(y_pred))
        # plt.plot(x, y_pred, 'r-*', label='y_pred')
        # plt.plot(x, y_valid, 'b-o', label='y_valid')
        # plt.show()

        # importance = pd.Series(self.gbm.feature_importances_, index=X_train.columns)
        #
        # self.important_feature = []
        # for index in range(len(importance)):
        #     if importance[index] > 0:
        #         self.important_feature.append(importance.index[index])
        # print(len(self.important_feature))

    def get_important_features(self):
        return self.important_feature

    def predict(self, test_X):
        a_pred = self.gbm.predict(test_X)
        gbm_a_pattern = pd.read_csv('input/b_pattern.csv', names=['id'])
        gbm_a_pattern['Y'] = a_pred
        gbm_a_pattern.to_csv('output/gbm_b_pattern.csv', index=None, header=None)
        print(gbm_a_pattern.head())
        return a_pred
