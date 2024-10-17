import numpy as np
import lightgbm as lgb

from .util import Parameters


def params_lgb(model):
    if "LGBM" in str(model):  # sklearn api
        booster = model._Booster
    else:
        booster = model
    model_params = booster.params

    num_trees = booster.num_trees()

    # get params
    objective_ = model_params["objective"]
    if objective_ == "regression":
        objective = "regression"
        num_class = 1
    elif objective_ == "binary":
        objective = "binary"
        num_class = 2
    elif objective_ == "multiclass":
        objective = "multiclass"
        num_class = booster.num_model_per_iteration()
    else:
        raise NotImplementedError(f"objective: {objective_}")

    learning_rate = float(model_params["learning_rate"])
    reg_lambda = float(model_params["reg_lambda"])

    params = Parameters(
        objective, num_class, "newton", "lgb", learning_rate, num_trees, reg_lambda
    )

    return params


def pred_leaf_lgb(model, x):
    if "LGBM" in str(model):  # sklearn api
        booster = model._Booster
    else:
        booster = model
    # leaves
    num_trees = booster.num_trees()
    leaves = booster.predict(x, pred_leaf=True).astype(np.int32)
    if num_trees == 1:
        leaves = leaves.reshape((-1, 1))  # force 2d

    return leaves
