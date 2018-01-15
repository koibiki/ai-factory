import xgboost as xgb
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error

from model_select.predict_model import PredictModel


class Xgb(PredictModel):
    xgb = None
    sort_valid = None

    def create_predict_model(self):
        self.xgb = xgb.XGBRegressor(max_depth=8,
                                    learning_rate=0.05,
                                    n_estimators=500,
                                    silent=True,
                                    objective='reg:linear',
                                    nthread=-1,
                                    min_child_weight=2,
                                    max_delta_step=0,
                                    subsample=0.8,
                                    colsample_bytree=0.7,
                                    colsample_bylevel=1,
                                    reg_alpha=0,
                                    reg_lambda=1,
                                    scale_pos_weight=1,
                                    seed=1440,
                                    missing=None)

    def run(self, X_train, y_train, X_valid, y_valid):
        self.create_predict_model()
        self.xgb.fit(X_train, y_train,
                     eval_set=[(X_valid, y_valid)],
                     eval_metric='rmse',
                     early_stopping_rounds=200)
        y_pred = self.xgb.predict(X_valid)
        print("xgb mean_squared_error:", mean_squared_error(y_pred, y_valid))

        # x = range(len(y_pred))
        # plt.plot(x, y_pred, 'r-*', label='y_pred')
        # plt.plot(x, y_valid, 'b-o', label='y_valid')
        # plt.legend()
        # plt.show()
        #
        # pd_valid = pd.concat([X_valid, y_valid], axis=1)
        # sort_pd_valid = pd_valid.sort_values(by='Y')
        #
        # sort_X_valid = sort_pd_valid.drop(['Y'], axis=1)
        # sort_y_pred = self.xgb.predict(sort_X_valid)
        # sort_y_valid = sort_pd_valid.Y.values
        #
        # x = range(len(sort_y_pred))
        # plt.plot(x, sort_y_pred, 'r-*', label='y_pred')
        # plt.plot(x, sort_y_valid, 'b-o', label='y_valid')
        # plt.legend()
        # plt.show()

    def predict(self, test_X):
        a_pred = self.xgb.predict(test_X)
        return a_pred
