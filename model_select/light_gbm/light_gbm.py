import lightgbm as lgm
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error

from model_select.predict_model import PredictModel


class LightGBM(PredictModel):
    gbm = None

    def create_predict_model(self):
        self.gbm = lgm.LGBMRegressor(objective='regression',
                                     boosting_type='gbdt',
                                     metric='l2',
                                     max_depth=8,
                                     min_sum_hessian_in_leaf=5.0,
                                     tree_learner='voting',
                                     num_trees=10000,
                                     feature_fraction=0.9,
                                     bagging_freq=5,
                                     bagging_fraction=0.8,
                                     min_data_in_leaf=100,
                                     learning_rate=0.05,
                                     n_estimators=200,
                                     metric_freq=1)

    def run(self, X_train, y_train, X_valid, y_valid):
        self.gbm.fit(X_train, y_train,
                     eval_set=[(X_valid, y_valid)],
                     eval_metric='l2',
                     early_stopping_rounds=200)
        y_pred = self.gbm.predict(X_valid)
        print("mean_squared_error:", mean_squared_error(y_pred, y_valid))

        x = range(len(y_pred))
        plt.plot(x, y_pred, 'r-*', label='y_pred')
        plt.plot(x, y_valid, 'b-o', label='y_valid')
        plt.legend()
        plt.show()
