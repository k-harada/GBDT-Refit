import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_breast_cancer

from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.ensemble import GradientBoostingClassifier

from retrainer import ReTrainer


def get_toy_data(dataset, objective, random_state, test_size=0.2):
    data = load_breast_cancer()
    X = data['data']
    y = data['target']
    task = 'binary'
    stratify = y if task in ['binary', 'multiclass'] else None
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size,
                                                        random_state=random_state,
                                                        stratify=stratify)
    return X_train, X_test, y_train, y_test


def test_xgb():
    X_train, X_test, y_train, y_test = get_toy_data(
        'breast_cancer', 'binary', 0
    )
    model = XGBClassifier(
        base_score=0.5,  # necessary to reproduce
        reg_lambda=1.0,  # not necessary, but you must not set 0
    )
    model.fit(X_train, y_train)
    new_model = ReTrainer(model)
    new_model.fit(X_train, y_train)
    pred_test_0 = model.predict_proba(X_test)[:, 1]
    pred_test_1 = new_model.predict(X_test)
    # print(np.abs(pred_test_0 - pred_test_1).max())


def test_lgb():
    X_train, X_test, y_train, y_test = get_toy_data(
        'breast_cancer', 'binary', 0
    )
    model = LGBMClassifier()
    model.fit(X_train, y_train)

    inf_values = calc_influence(model, X_train, y_train, X_test, y_test)


def test_cb():
    X_train, X_test, y_train, y_test = get_toy_data(
        'breast_cancer', 'binary', 0
    )
    model = CatBoostClassifier(
        leaf_estimation_iterations=1
    )
    model.fit(X_train, y_train)

    inf_values = calc_influence(model, X_train, y_train, X_test, y_test)


def test_shb():
    X_train, X_test, y_train, y_test = get_toy_data(
        'breast_cancer', 'binary', 0
    )
    model = HistGradientBoostingClassifier()
    model.fit(X_train, y_train)

    inf_values = calc_influence(model, X_train, y_train, X_test, y_test)


def test_sgb():
    X_train, X_test, y_train, y_test = get_toy_data(
        'breast_cancer', 'binary', 0
    )
    model = GradientBoostingClassifier()
    model.fit(X_train, y_train)

    inf_values = calc_influence(model, X_train, y_train, X_test, y_test)


if __name__ == "__main__":
    # test_sgb()
    # test_shb()
    test_xgb()
    # test_lgb()
    # test_cb()
