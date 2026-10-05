"""Fixed-configuration XGBoost one-step forecaster with validation-only early stopping and SHAP."""

from __future__ import annotations

from datetime import datetime

from .baseline import evaluate_predictions, preview_predictions

FEATURE_NAMES = ["lag_1", "lag_48", "target_slot", "weekday", "month"]

PARAMS = {
    "objective": "reg:squarederror",
    "tree_method": "hist",
    "max_depth": 4,
    "learning_rate": 0.05,
    "subsample": 0.8,
    "seed": 42,
    "nthread": 2,
    "eval_metric": "mae",
}
MAX_ROUNDS = 500
EARLY_STOPPING_ROUNDS = 30


def feature_row(example: dict, case: str) -> list[float]:
    """Five causal features: previous actual, previous-day actual, known target slot/weekday/month."""

    key = example["key"]
    if case == "aemo":
        if not isinstance(key, datetime):
            raise ValueError("AEMO example key must be a timestamp")
        slot = key.hour * 2 + key.minute // 30
        day = key.date()
    elif case == "ausgrid":
        day, slot = key
    else:
        raise ValueError(f"Unknown case: {case}")
    return [float(example["lag_1"]), float(example["lag_48"]), float(slot), float(day.weekday()), float(day.month)]


def _matrix(examples: list[dict], case: str):
    import numpy as np

    return (
        np.array([feature_row(example, case) for example in examples], dtype=float),
        np.array([example["target"] for example in examples], dtype=float),
    )


def fit_and_validate(parts: dict, case: str, include_preview: bool = False) -> dict:
    """Train on ``train``, pick the tree count on ``validation``; the test part is never touched."""

    try:
        import numpy as np
        import xgboost as xgb
    except ImportError as error:
        raise RuntimeError(f"XGBoost dependencies are not installed: {error}") from error

    x_train, y_train = _matrix(parts["train"], case)
    x_val, y_val = _matrix(parts["validation"], case)
    train = xgb.DMatrix(x_train, label=y_train, feature_names=FEATURE_NAMES)
    validation = xgb.DMatrix(x_val, label=y_val, feature_names=FEATURE_NAMES)

    booster = xgb.train(
        PARAMS, train, num_boost_round=MAX_ROUNDS,
        evals=[(validation, "validation")], early_stopping_rounds=EARLY_STOPPING_ROUNDS, verbose_eval=False,
    )
    best_iteration = booster.best_iteration
    iteration_range = (0, best_iteration + 1)

    raw = booster.predict(validation, iteration_range=iteration_range)
    predictions = [max(0.0, float(value)) for value in raw]
    metrics = evaluate_predictions(parts["validation"], predictions)

    contributions = booster.predict(validation, iteration_range=iteration_range, pred_contribs=True)
    mean_abs = np.abs(contributions[:, :-1]).mean(axis=0)
    first = contributions[0]
    result = {
        "model": "XGBoost",
        "validation_metrics": metrics,
        "best_iteration": int(best_iteration),
        "max_rounds": MAX_ROUNDS,
        "parameters": {**PARAMS, "early_stopping_rounds": EARLY_STOPPING_ROUNDS},
        "features": FEATURE_NAMES,
        "validation_mean_absolute_SHAP_by_feature": {
            name: float(value) for name, value in zip(FEATURE_NAMES, mean_abs)
        },
        "validation_first_local_SHAP": {
            "bias": float(first[-1]),
            "feature_contributions": {name: float(value) for name, value in zip(FEATURE_NAMES, first[:-1])},
            "raw_prediction_before_clipping": float(raw[0]),
        },
    }
    result["validation_prediction_rows"] = (
        preview_predictions(parts["validation"], predictions, limit=len(predictions)) if include_preview else []
    )
    return result
