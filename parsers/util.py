from dataclasses import dataclass


@dataclass
class Parameters:
    """Class for GBDT parameters."""
    objective: str  # regression / binary / multiclass
    num_class: int
    formula: str  # gradient / newton
    model_type: str  # xgb / lgb / cb
    learning_rate: float
    num_trees: int
    reg_lambda: float
