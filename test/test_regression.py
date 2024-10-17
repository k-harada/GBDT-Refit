import numpy as np
import json
from sklearn.model_selection import train_test_split
from sklearn.datasets import fetch_california_housing

from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor

from retrainer import ReTrainer


def get_toy_data(dataset, objective, random_state, test_size=0.2):
    data = fetch_california_housing()
    X = data['data']
    y = data['target']
    task = 'regression'
    stratify = y if task in ['binary', 'multiclass'] else None
    X_train, X_test, y_train, y_test = train_test_split(X[:500], y[:500], test_size=test_size,
                                                        random_state=random_state,
                                                        stratify=stratify)
    return X_train, X_test, y_train, y_test


def test_xgb():
    X_train, X_test, y_train, y_test = get_toy_data(
        'breast_cancer', 'binary', 0
    )
    model = XGBRegressor()
    model.fit(X_train, y_train)
    # get base_score
    # necessary to reproduce
    if "XGB" in str(model):  # sklearn api
        booster = model.get_booster()
    else:
        # xgboost.core.Booster
        booster = model
    xgb_config = json.loads(booster.save_config())
    base_score = float(xgb_config["learner"]["learner_model_param"]["base_score"])

    new_model = ReTrainer(model)
    new_model.fit(X_train, y_train, base_score)
    pred_test_0 = model.predict(X_test)
    pred_test_1 = new_model.predict(X_test)
    print(np.abs(pred_test_0 - pred_test_1).max())


def test_lgb():
    X_train, X_test, y_train, y_test = get_toy_data(
        'breast_cancer', 'binary', 0
    )
    model = LGBMRegressor()
    model.fit(X_train, y_train)
    new_model = ReTrainer(model)
    new_model.fit(X_train, y_train)
    pred_test_0 = model.predict(X_test)
    pred_test_1 = new_model.predict(X_test)
    print(np.abs(pred_test_0 - pred_test_1).max())


def test_cb():
    X_train, X_test, y_train, y_test = get_toy_data(
        'breast_cancer', 'binary', 0
    )
    model = CatBoostRegressor(
        leaf_estimation_iterations=1, n_estimators=100
    )
    model.fit(X_train, y_train)
    new_model = ReTrainer(model)
    new_model.fit(X_train, y_train)
    pred_test_0 = model.predict(X_test)
    pred_test_1 = new_model.predict(X_test)
    print(np.abs(pred_test_0 - pred_test_1).max())


if __name__ == "__main__":
    test_xgb()
    test_lgb()
    test_cb()
