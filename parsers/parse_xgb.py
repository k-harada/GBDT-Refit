import json

import numpy as np
import xgboost as xgb

from .util import Parameters


def params_xgb(model):
    if "XGB" in str(model):  # sklearn api
        booster = model.get_booster()
    else:
        # xgboost.core.Booster
        booster = model
    xgb_config = json.loads(booster.save_config())

    best = booster.attr("best_iteration")
    if best is not None:
        num_trees = booster.best_iteration + 1
    else:
        num_trees = booster.num_boosted_rounds()

    # get params
    objective_ = xgb_config["learner"]["objective"]["name"]
    if objective_ == "reg:squarederror":
        objective = "regression"
        num_class = 1
    elif objective_ == "binary:logistic":
        objective = "binary"
        num_class = 2
    elif objective_ == "multi:softprob":
        objective = "multiclass"
        num_class = int(xgb_config["learner"]["learner_model_param"]["num_class"])
    else:
        raise NotImplementedError(f"objective: {objective_}")

    if "tree_train_param" in xgb_config["learner"]["gradient_booster"].keys():
        # >= v2.0.0
        train_params = xgb_config["learner"]["gradient_booster"]["tree_train_param"]
    else:
        # < v2.0.0
        train_params = xgb_config["learner"]["gradient_booster"]["updater"]['grow_colmaker']["train_param"]
    learning_rate = float(train_params["learning_rate"])
    reg_lambda = float(train_params["reg_lambda"])

    params = Parameters(
        objective, num_class, "newton", "xgb", learning_rate, num_trees, reg_lambda
    )

    return params


def pred_leaf_xgb(model, x):
    if "XGB" in str(model):  # sklearn api
        booster = model.get_booster()
    else:
        booster = model
    # leaves
    best = booster.attr("best_iteration")
    if best is not None:
        num_trees = booster.best_iteration + 1
    else:
        num_trees = booster.num_boosted_rounds()

    leaves = booster.predict(
        xgb.DMatrix(x), pred_leaf=True, iteration_range=(0, num_trees)
    ).astype(np.int32)

    if num_trees == 1:
        leaves = leaves.reshape((-1, 1))  # force 2d

    return leaves
