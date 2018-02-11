from sklearn.metrics import mean_squared_error
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.model_selection import KFold

from feature_engineering.create_sample import create_sample
from model_selection.multi_classifier_model_factory import MultiClassifierModelFactory
from model_selection.classifier_model_factory import ClassifierModelFactory
from model_selection.regressor_model_factory import RegressorModelFactory
from utils.utils import *

mcmf = MultiClassifierModelFactory()
cmf = ClassifierModelFactory()
rmf = RegressorModelFactory()


def classify(x):
    if x >= 3.2:
        return 0
    elif x <= 2.6:
        return 1
    else:
        return 2


def k_fold_smote_regressor(train_x, train_y, test_x, model_num, cv=5, important_level=2):
    print('开始CV{}折训练...'.format(cv))
    kf = KFold(n_splits=cv, shuffle=True, random_state=520)
    y_pred = np.zeros(train_x.shape[0])
    test_y_preds = []
    cv_indexs = {}
    importances = []
    for i, (train_index, test_index) in enumerate(kf.split(train_x)):
        print('第{}次训练...'.format(i))
        model = rmf.create_model(model_num)
        kf_X_train = train_x.iloc[train_index]
        kf_y_train = train_y.iloc[train_index]
        X_y_train = pd.concat([kf_X_train, kf_y_train], axis=1)

        X_y_train['class'] = X_y_train.Y.apply(lambda x: classify(x))
        high_num = X_y_train[X_y_train['class'] == 0].shape[0]
        median_num = X_y_train[X_y_train['class'] == 2].shape[0]
        low_num = X_y_train[X_y_train['class'] == 1].shape[0]

        high_scale = int(np.floor(median_num / high_num))
        low_scale = int(np.floor(median_num / low_num))

        high_data = X_y_train[X_y_train['class'] == 0].iloc[:, :-1].reset_index(drop=True)
        low_data = X_y_train[X_y_train['class'] == 1].iloc[:, :-1].reset_index(drop=True)
        median_data = X_y_train[X_y_train['class'] == 2].iloc[:, :-1].reset_index(drop=True)

        high_samples = []
        if high_scale >= 2:
            high_samples = [high_data] + [create_sample(high_data) for time in range(high_scale - 1)]

        low_samples = []
        if low_scale >= 2:
            low_samples = [low_data] + [create_sample(low_data) for time in range(low_scale - 1)]
        print('high:' + str(len(high_samples)) + ' median:' + str(len(median_data)) + ' low:' + str(len(low_samples)))

        new_data = pd.concat(high_samples + [median_data] + low_samples, axis=0).reset_index(drop=True)

        kf_X_train = new_data.iloc[:, :-1]
        kf_y_train = new_data.iloc[:, -1]
        kf_X_valid = train_x.iloc[test_index]
        kf_y_valid = train_y.iloc[test_index]
        model.fit(kf_X_train, kf_X_valid, kf_y_train, kf_y_valid)
        y_pred[test_index] += model.predict(kf_X_valid)
        test_y_pred = model.predict(test_x)
        test_y_preds.append(test_y_pred)
        cv_indexs[i] = [train_index, test_index]
        if model.can_get_feature_importance():
            importances.append(model.feature_importance(important_level))
    print(rmf.get_model_name(model_num) + ' k fold validation:', mean_squared_error(train_y, y_pred))
    predict = calculate_mean(test_y_preds)
    return predict, cv_indexs, cv_important_feature2(importances)
