from sklearn.metrics import mean_squared_error
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.model_selection import KFold

from model_selection.multi_classifier_model_factory import MultiClassifierModelFactory
from model_selection.classifier_model_factory import ClassifierModelFactory
from model_selection.regressor_model_factory import RegressorModelFactory
from utils.utils import *

mcmf = MultiClassifierModelFactory()
cmf = ClassifierModelFactory()
rmf = RegressorModelFactory()


def k_fold_multi_classifier(train_x, train_y, test_x, model_num, cv=5):
    print('开始CV{}折训练...'.format(cv))
    kf = KFold(n_splits=cv, shuffle=True, random_state=20)

    test_y_preds = []
    cv_indexs = {}
    for i, (train_index, test_index) in enumerate(kf.split(train_x)):
        print('第{}次训练...'.format(i))
        model = mcmf.create_model(model_num)
        kf_X_train = train_x.iloc[train_index]
        kf_y_train = train_y.iloc[train_index]
        kf_X_valid = train_x.iloc[test_index]
        kf_y_valid = train_y.iloc[test_index]
        model.fit(kf_X_train, kf_X_valid, kf_y_train, kf_y_valid)
        kf_y_pred = model.predict(kf_X_valid)
        print(kf_y_pred)
        print(softmax_to_class(kf_y_pred))
        print(classification_report(kf_y_valid, softmax_to_class(kf_y_pred)))
        test_y_pred = model.predict(test_x)
        test_y_preds.append(test_y_pred)
        cv_indexs[i] = [train_index, test_index]
    predict = calculate_multi_mean(test_y_preds)
    return predict


def k_fold_classifier(train_x, train_y, test_x, model_num, cv=5):
    print('开始CV{}折训练...'.format(cv))
    kf = KFold(n_splits=cv, shuffle=True, random_state=33)
    ass = []
    test_y_preds = []
    cv_indexs = {}
    for i, (train_index, test_index) in enumerate(kf.split(train_x)):
        print('第{}次训练...'.format(i))
        model = cmf.create_model(model_num)
        kf_X_train = train_x.iloc[train_index]
        kf_y_train = train_y.iloc[train_index]
        kf_X_valid = train_x.iloc[test_index]
        kf_y_valid = train_y.iloc[test_index]
        model.fit(kf_X_train, kf_X_valid, kf_y_train, kf_y_valid)
        kf_y_pred = model.predict(kf_X_valid)
        ass.append(accuracy_score(kf_y_valid, logloss_to_class(kf_y_pred)))
        test_y_pred = model.predict(test_x)
        test_y_preds.append(test_y_pred)
        cv_indexs[i] = [train_index, test_index]
    print(cmf.get_model_name(model_num) + ' k fold validation:', sum(ass) / len(ass))
    predict = calculate_mean(test_y_preds)
    return predict, sum(ass) / len(ass)


def k_fold_regressor(train_x, train_y, test_x, model_num, cv=5, important_level=2):
    print('开始CV{}折训练...'.format(cv))
    kf = KFold(n_splits=cv, shuffle=True, random_state=520)
    y_pred = np.zeros(train_x.shape[0])
    test_y_preds = []
    cv_indexs ={}
    importances = []
    for i, (train_index, test_index) in enumerate(kf.split(train_x)):
        print('第{}次训练...'.format(i))
        model = rmf.create_model(model_num)
        kf_X_train = train_x.iloc[train_index]
        kf_y_train = train_y.iloc[train_index]
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
    return predict, cv_indexs, calculate_importance_feature(importances)


