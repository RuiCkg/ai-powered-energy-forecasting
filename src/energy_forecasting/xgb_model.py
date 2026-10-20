
"""Fixed-configuration XGBoost one-step forecaster with validation-only early stopping and SHAP."""

from __future__ import annotations

from datetime import datetime

from .baseline import evaluate_predictions, preview_predictions
from .explainability import compute_shap_contributions, global_feature_importance, local_explanation


# --------------------------------------------------
# XGBOOST CONFIGURATION
# --------------------------------------------------

FEATURE_NAMES = [
    "lag_1",
    "lag_48",
    "target_slot",
    "weekday",
    "month",
]

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

# Names required by the experiment-record module
FIXED_PARAMETERS = PARAMS.copy()
NUM_BOOST_ROUND = MAX_ROUNDS

# Maximum permitted relative SHAP reconstruction error
SHAP_ADDITIVITY_TOLERANCE = 1e-3


# --------------------------------------------------
# FEATURE ENGINEERING
# --------------------------------------------------

def feature_row(example: dict, case: str) -> list[float]:
    """Prepare causal lag, calendar and optional known-covariate features."""

    key = example["key"]

    if case in {"aemo", "custom"}:
        if not isinstance(key, datetime):
            raise ValueError(f"{case} example key must be a timestamp")

        slot = example.get("target_slot", key.hour * 2 + key.minute // 30)
        day = key.date()

    elif case == "ausgrid":
        day, slot = key

    else:
        raise ValueError(f"Unknown case: {case}")

    row = [
        float(example["lag_1"]),
        float(example["lag_48"]),
        float(slot),
        float(day.weekday()),
        float(day.month),
    ]
    if case == "custom":
        row.extend(
            float(value)
            for _, value in sorted(example.get("custom_covariates", {}).items())
        )
    return row


def feature_names_for(examples: list[dict], case: str) -> list[str]:
    """Return the matrix column names for a forecasting case."""

    names = list(FEATURE_NAMES)
    if case == "custom":
        names[1] = "lag_seasonal"
        if not examples:
            raise ValueError("No custom examples were provided")
        names.extend(sorted(examples[0].get("custom_covariates", {})))
    return names


# --------------------------------------------------
# CREATE TRAINING MATRIX
# --------------------------------------------------

def _matrix(examples: list[dict], case: str):
    """Convert forecasting examples into XGBoost input arrays."""

    import numpy as np

    features = np.array(
        [feature_row(example, case) for example in examples],
        dtype=float,
    )

    targets = np.array(
        [example["target"] for example in examples],
        dtype=float,
    )

    return features, targets


# --------------------------------------------------
# SHAP ADDITIVITY VALIDATION
# --------------------------------------------------

def shap_additivity_error(contributions, raw_predictions):
    """
    Return the maximum scaled error between SHAP
    reconstruction and raw model predictions.

    The final column of each contribution row is the bias.
    All contributions, including the bias, are summed.

    This function checks every prediction row.
    """

    import numpy as np

    values = np.asarray(contributions, dtype=float)
    predictions = np.asarray(raw_predictions, dtype=float)

    if predictions.size == 0:
        raise ValueError("No predictions provided")


    if values.ndim != 2:
        raise ValueError(
            "SHAP contributions must be a two-dimensional array"
        )

    if predictions.ndim != 1:
        raise ValueError(
            "Raw predictions must be a one-dimensional array"
        )

    if values.shape[0] == 0 or values.shape[1] == 0:
        raise ValueError("SHAP contributions cannot be empty")

    if values.shape[0] != predictions.shape[0]:
        raise ValueError(
            "SHAP contributions and predictions are not aligned"
        )

    if not np.all(np.isfinite(values)):
        raise ValueError("SHAP contributions contain invalid values")

    if not np.all(np.isfinite(predictions)):
        raise ValueError("Raw predictions contain invalid values")

    reconstructed = values.sum(axis=1)

    absolute_errors = np.abs(reconstructed - predictions)

    scale = np.maximum(1.0, np.abs(predictions))

    scaled_errors = absolute_errors / scale

    return float(np.max(scaled_errors))


def verify_shap_additivity(
    result: dict,
    tolerance: float = SHAP_ADDITIVITY_TOLERANCE,
) -> bool:
    """Verify SHAP additivity for the stored local explanation."""

    local = result["validation_first_local_SHAP"]

    reconstructed = (
        float(local["bias"])
        + sum(
            float(value)
            for value in local["feature_contributions"].values()
        )
    )

    prediction = float(local["raw_prediction_before_clipping"])

    error = abs(reconstructed - prediction) / max(
        1.0,
        abs(prediction),
    )

    return error <= tolerance


# --------------------------------------------------
# TRAIN AND VALIDATE XGBOOST
# --------------------------------------------------

def fit_and_validate(
    parts: dict,
    case: str,
    include_preview: bool = False,
) -> dict:
    """
    Train on the training data.

    Select the best tree count using validation data.

    The held-out test data is not used.
    """

    try:
        import numpy as np
        import xgboost as xgb

    except ImportError as error:
        raise RuntimeError(
            f"XGBoost dependencies are not installed: {error}"
        ) from error

    # Prepare training and validation data
    feature_names = feature_names_for(parts["train"], case)
    x_train, y_train = _matrix(parts["train"], case)
    x_val, y_val = _matrix(parts["validation"], case)

    train = xgb.DMatrix(
        x_train,
        label=y_train,
        feature_names=feature_names,
    )

    validation = xgb.DMatrix(
        x_val,
        label=y_val,
        feature_names=feature_names,
    )

    # Train model
    booster = xgb.train(
        PARAMS,
        train,
        num_boost_round=MAX_ROUNDS,
        evals=[(validation, "validation")],
        early_stopping_rounds=EARLY_STOPPING_ROUNDS,
        verbose_eval=False,
    )

    # Best validation iteration
    best_iteration = booster.best_iteration

    iteration_range = (0, best_iteration + 1)

    # Generate raw predictions
    raw = booster.predict(
        validation,
        iteration_range=iteration_range,
    )

    # Energy values cannot be negative
    predictions = [
        max(0.0, float(value))
        for value in raw
    ]

    # Calculate validation metrics
    metrics = evaluate_predictions(
        parts["validation"],
        predictions,
    )

    # --------------------------------------------------
    # SHAP FEATURE CONTRIBUTIONS
    # --------------------------------------------------

    contributions = compute_shap_contributions(
        booster,
        validation,
        iteration_range=iteration_range,
    )

    # Validate SHAP reconstruction against raw predictions
    shap_error = shap_additivity_error(
        contributions,
        raw,
    )

    if shap_error > SHAP_ADDITIVITY_TOLERANCE:
        raise RuntimeError(
            "SHAP additivity verification failed. "
            f"Maximum scaled error: {shap_error:.8f}"
        )

    mean_abs = global_feature_importance(contributions).reindex(feature_names)
    first = local_explanation(contributions)

    # --------------------------------------------------
    # RESULT INFORMATION
    # --------------------------------------------------

    result = {
        "model": "XGBoost",

        "validation_metrics": metrics,

        "best_iteration": int(best_iteration),

        "max_rounds": MAX_ROUNDS,

        "parameters": {
            **PARAMS,
            "early_stopping_rounds": EARLY_STOPPING_ROUNDS,
        },

        "features": feature_names,
        "feature_names": feature_names,

        "validation_mean_absolute_SHAP_by_feature": {
            name: float(value)
            for name, value in mean_abs.items()
        },

        "validation_first_local_SHAP": {
            "bias": first["bias"],
            "feature_contributions": {
                name: float(value)
                for name, value in first["feature_contributions"].items()
            },
            "raw_prediction_before_clipping": float(raw[0]),
        },

        "shap_additivity_error": float(shap_error),
    }

    # --------------------------------------------------
    # VALIDATION PREDICTION DOWNLOAD DATA
    # --------------------------------------------------

    result["validation_prediction_rows"] = (
        preview_predictions(
            parts["validation"],
            predictions,
            limit=len(predictions),
        )
        if include_preview
        else []
    )

    return result


def fit_and_evaluate_test(
    parts: dict,
    case: str,
    num_boost_round: int,
    include_preview: bool = False,
) -> dict:
    """Retrain on train+validation and evaluate the untouched test partition."""

    try:
        import xgboost as xgb
    except ImportError as error:
        raise RuntimeError(f"XGBoost dependencies are not installed: {error}") from error
    if not isinstance(num_boost_round, int) or num_boost_round < 1:
        raise ValueError("Final XGBoost tree count must be a positive integer")

    development = list(parts["train"]) + list(parts["validation"])
    feature_names = feature_names_for(development, case)
    x_development, y_development = _matrix(development, case)
    x_test, y_test = _matrix(parts["test"], case)
    train = xgb.DMatrix(x_development, label=y_development, feature_names=feature_names)
    test = xgb.DMatrix(x_test, label=y_test, feature_names=feature_names)
    booster = xgb.train(PARAMS, train, num_boost_round=num_boost_round, verbose_eval=False)
    predictions = [max(0.0, float(value)) for value in booster.predict(test)]
    result = {
        "model": "XGBoost",
        "test_metrics": evaluate_predictions(parts["test"], predictions),
        "num_boost_round_selected_on_validation": num_boost_round,
        "feature_names": feature_names,
        "training_partition": "train plus validation",
        "evaluation_partition": "reserved test",
    }
    result["test_prediction_rows"] = (
        preview_predictions(parts["test"], predictions, limit=len(predictions))
        if include_preview else []
    )
    return result
