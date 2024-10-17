from typing import Optional
from dataclasses import asdict
import numpy as np

from parsers.util import Parameters
from parsers.parse_xgb import params_xgb, pred_leaf_xgb
from parsers.parse_lgb import params_lgb, pred_leaf_lgb
from parsers.parse_cb import params_cb, pred_leaf_cb


class ReTrainer:
    def __init__(self, model):
        """
        :param model: Tree ensemble object.
        """
        self.model = model
        self.is_fit = False
        # get parameters
        if "LGBM" in str(model):
            self.params = asdict(params_lgb(model))
        elif "XGB" in str(model) or "xgb" in str(model):
            self.params = asdict(params_xgb(model))
        elif "HistGradientBoosting" in str(model):
            raise NotImplementedError
        elif "GradientBoosting" in str(model):
            raise NotImplementedError
            # self.leaves_train, self.params = parse_sgb(model, x_train, y_train, train=True)
        elif 'CatBoost' in str(model):
            self.params = asdict(params_cb(model))
        else:
            raise NotImplementedError

    def get_leaves(self, x):
        model = self.model
        # get leaves
        if "LGBM" in str(model):
            return pred_leaf_lgb(model, x)
        elif "XGB" in str(model) or "xgb" in str(model):
            return pred_leaf_xgb(model, x)
        elif "HistGradientBoosting" in str(model):
            raise NotImplementedError
        elif "GradientBoosting" in str(model):
            raise NotImplementedError
            # self.leaves_train, self.params = parse_sgb(model, x_train, y_train, train=True)
        elif 'CatBoost' in str(model):
            return pred_leaf_cb(model, x)
        else:
            raise NotImplementedError

    def fit(self, x_train, y_train, base_score=None):
        self.leaves_train = self.get_leaves(x_train)
        # re-fit
        self.x_train = x_train
        self.y_train = y_train
        self.f_train = np.zeros(self.leaves_train.shape).astype(np.float64)  # assigned value on each step
        self.g_train = np.zeros(self.leaves_train.shape).astype(np.float64)  # gradient on each step
        self.h_train = np.ones(self.leaves_train.shape).astype(np.float64)  # hessian on each step
        self.hh_train = np.zeros(self.leaves_train.shape).astype(np.float64)  # sum of hessian of same leaf on each step
        self.k_train = np.zeros(self.leaves_train.shape).astype(np.float64)  # third derivative on each step

        # assertion
        if self.params["objective"] == "regression":
            assert len(y_train.shape) <= 2
            if len(y_train.shape) == 2:
                assert y_train.shape[1] == 1
                y_train_ = y_train.reshape((-1, ))
            else:
                y_train_ = y_train
            self.refit_1d(y_train_, base_score)
        elif self.params["objective"] == "binary":
            assert len(y_train.shape) <= 2
            if len(y_train.shape) == 2:
                assert y_train.shape[1] == 2 and y_train.max() == 1
                y_train_ = y_train[:, 1]
            else:
                y_train_ = y_train
            self.refit_1d(y_train_, base_score)
        elif self.params["objective"] == "multiclass":
            assert len(y_train.shape) <= 2
            if len(y_train.shape) == 2:
                assert y_train.shape[1] == self.params["n_class"] and y_train.max() == 1
                y_train_ = y_train.astype(np.float64)
            else:
                assert y_train.min() >= 0
                assert y_train.max() <= self.params["n_class"] - 1
                y_train_ = np.identity(self.params["n_class"])[y_train].astype(np.float64)
            self.refit_2d(y_train_, base_score)
        else:
            raise NotImplementedError(f"objective {self.params['objective']} is not implemented.")
        self.is_fit = True
        return self

    def refit_1d(self, y_train, base_score):

        n_data, n_trees = self.leaves_train.shape
        gg_train = np.zeros(self.leaves_train.shape).astype(np.float64)  # gradient on each step
        if base_score is None:
            self.params["base_score"] = y_train.mean()
        else:
            self.params["base_score"] = base_score
        p = self.params["base_score"] * np.ones(y_train.shape[0])

        if self.params["objective"] == "regression":
            z = p
        else:  # self.params.objective == "binary":
            z = np.log(p / (1 - p))

        for t in range(n_trees):
            node_id_min, node_id_max = self.leaves_train[:, t].min(), self.leaves_train[:, t].max()
            n_leaves = node_id_max - node_id_min + 1
            map_leaves_train = np.zeros((n_data, n_leaves)).astype(np.float64)

            for i, k in enumerate(self.leaves_train[:, t]):
                map_leaves_train[i, k - node_id_min] = 1
                if self.params["objective"] == "regression":
                    self.h_train[i, t] = 1.0
                    # self.k_train[i, t] = 0.0
                elif self.params["objective"] == "binary":
                    self.h_train[i, t] = p[i] * (1.0 - p[i])
                    self.k_train[i, t] = p[i] * (1.0 - p[i]) * (1.0 - 2.0 * p[i])
                else:
                    raise NotImplementedError

            if self.params["formula"] == "gradient":
                weight_by_leaves = self.params["reg_lambda"] + map_leaves_train.sum(axis=0, keepdims=True)
            else:
                weight_by_leaves = self.params["reg_lambda"] + (
                        map_leaves_train * self.h_train[:, t].reshape((-1, 1))
                ).sum(axis=0, keepdims=True)
            self.hh_train[:, t] = (map_leaves_train * weight_by_leaves).sum(axis=1)
            self.g_train[:, t] = p - y_train
            gg_train[:, t] = np.matmul(
                map_leaves_train, np.matmul(map_leaves_train.transpose(), self.g_train[:, t])
            )
            self.f_train[:, t] = - gg_train[:, t] / self.hh_train[:, t]
            z += self.params["learning_rate"] * self.f_train[:, t]
            if self.params["objective"] == "regression":
                p = z
            elif self.params["objective"] == "binary":
                p = 1.0 / (1.0 + np.exp(-z))
            else:
                raise NotImplementedError
        # check

    def refit_2d(self, y_train, base_score):

        n_data, n_trees = self.leaves_train.shape
        gg_train = np.zeros(self.leaves_train.shape).astype(np.float64)  # gradient on each step
        z = np.zeros(y_train.shape).astype(np.float64)
        p = np.ones(y_train.shape).astype(np.float64) / self.params["n_class"]

        for t in range(n_trees):
            t_ = t % self.params["n_class"]
            node_id_min, node_id_max = self.leaves_train[:, t].min(), self.leaves_train[:, t].max()
            n_leaves = node_id_max - node_id_min + 1
            map_leaves_train = np.zeros((n_data, n_leaves)).astype(np.float32)

            for i, k in enumerate(self.leaves_train[:, t]):
                map_leaves_train[i, k - node_id_min] = 1
                self.h_train[i, t] = 2.0 * p[i, t_] * (1.0 - p[i, t_])
                self.k_train[i, t] = 2.0 * p[i, t_] * (1.0 - p[i, t_]) * (1.0 - 2.0 * p[i, t_])

            if self.params["formula"] == "gradient":
                weight_by_leaves = self.params["reg_lambda"] + map_leaves_train.sum(axis=0, keepdims=True)
            else:
                weight_by_leaves = self.params["reg_lambda"] + (
                        map_leaves_train * self.h_train[:, t].reshape((-1, 1))
                ).sum(axis=0, keepdims=True)
            self.hh_train[:, t] = (map_leaves_train * weight_by_leaves).sum(axis=1)
            self.g_train[:, t] = p[:, t_] - y_train[:, t_]
            gg_train[:, t] = np.matmul(
                map_leaves_train, np.matmul(map_leaves_train.transpose(), self.g_train[:, t])
            )
            self.f_train[:, t] = - gg_train[:, t] / self.hh_train[:, t]
            z[:, t_] += self.params["learning_rate"] * self.f_train[:, t]

            if t_ == self.params["n_class"] - 1:
                z -= z.max(axis=1, keepdims=True)
                p = np.exp(z) / np.exp(z).sum(axis=1, keepdims=True)

        # check

    def predict(self, x):
        assert self.is_fit
        n_trees = self.leaves_train.shape[1]
        p = self.params["base_score"] * np.ones(x.shape[0])
        if self.params["objective"] == "regression":
            z = p
        else:  # self.params.objective == "binary":
            z = np.log(p / (1 - p))
        pred_leaves = self.get_leaves(x)

        for t in range(n_trees):
            pred_f = np.zeros(x.shape[0])
            n_leaves = max(self.leaves_train[:, t].max(), pred_leaves[:, t].max())
            f_leaves = np.zeros(n_leaves + 1).astype(float)

            for i, k in enumerate(self.leaves_train[:, t]):
                f_leaves[k] = self.f_train[i, t]
            for i, k in enumerate(pred_leaves[:, t]):
                pred_f[i] = f_leaves[k]

            z += self.params["learning_rate"] * pred_f
            if self.params["objective"] == "regression":
                p = z
            elif self.params["objective"] == "binary":
                p = 1.0 / (1.0 + np.exp(-z))
            else:
                raise NotImplementedError

        return p
