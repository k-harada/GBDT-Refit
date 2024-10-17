import json

import numpy as np
import catboost as cb

from .util import Parameters


def params_cb(model):
    assert "CatBoost" in str(model)  # sklearn api
    model_params = model.get_all_params()

    # get params
    if "CatBoostClassifier" in str(model):
        num_class = len(model_params['class_names'])
        if num_class == 2:
            objective = "binary"
        else:
            objective = "multiclass"
    elif "CatBoostRegressor" in str(model):
        objective = "regression"
        num_class = 1
    else:
        raise NotImplementedError(f"{str(model)}")

    print(model_params)

    params = Parameters(
        objective, num_class, "newton", "cb", model_params['learning_rate'],
        model_params['iterations'], model_params['l2_leaf_reg']
    )

    return params


def pred_leaf_cb(model, x):
    leaves = model.calc_leaf_indexes(x)
    model_params = model.get_all_params()
    num_trees = model_params['iterations']
    if num_trees == 1:
        leaves = leaves.reshape((-1, 1))  # force 2d
    return leaves
