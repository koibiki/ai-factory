from model_select.gbr.gradient_boosting import GBR
from model_select.xgboost.xgb import Xgb
from model_select.knn.knn import KNR
from model_select.skflow_dnn.skflow_dnn import TensorDNN
from model_select.skflow_lr.skflow_lr import TensorLr
from model_select.light_gbm.light_gbm import LightGBM
from model_select.linear.linear import LinearR
from model_select.svm.svm import Svr


class ModelFactory(object):

    MODEL_LIGHET_GBM = 0
    MODEL_XGBOOST = 1
    MODEL_GBR = 2
    MODEL_SVR = 3
    MODEL_TENSOR_DNN = 4
    MODEL_TENSOR_LR = 5
    MODEL_KNR = 6
    MODEL_LINEAR = 7

    def create_model(self, argument):
        method_name = 'model_' + str(argument)
        method = getattr(self, method_name, lambda: "nothing")
        return method()

    def model_0(self):
        return LightGBM()

    def model_1(self):
        return Xgb()

    def model_2(self):
        return GBR()

    def model_3(self):
        return Svr()

    def model_4(self):
        return TensorDNN()

    def model_5(self):
        return TensorLr()

    def model_6(self):
        return KNR()

    def model_7(self):
        return LinearR()

    def get_model_name(self, argument):
        method_name = 'get_model_name_' + str(argument)
        method = getattr(self, method_name, lambda: "nothing")
        return method()

    def get_model_name_0(self):
        return 'light_gbm'

    def get_model_name_1(self):
        return 'xgboost'

    def get_model_name_2(self):
        return 'gbr'

    def get_model_name_3(self):
        return 'svr'

    def get_model_name_4(self):
        return 'dnn'

    def get_model_name_5(self):
        return 'lr'

    def get_model_name_6(self):
        return 'knr'

    def get_model_name_7(self):
        return 'lr'
