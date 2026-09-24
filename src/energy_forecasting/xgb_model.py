"""Small, fixed-config XGBoost experiment using only causal source features."""

from __future__ import annotations

from .baseline import evaluate_predictions, preview_predictions


FEATURE_NAMES = ("lag_1", "lag_48", "slot_of_day", "day_of_week", "month")


def feature_row(example: dict, case: str) -> list[float]:
    """Calendar fields describe the known target slot; lag values are past actuals."""

    if case == "aemo":
        timestamp = example["key"]
        slot = timestamp.hour * 2 + timestamp.minute // 30
        source_date = timestamp.date()
    elif case == "ausgrid":
        source_date, slot = example["key"]
    else:
        raise ValueError(f"Unknown XGBoost case: {case}")
    return [
        float(example["lag_1"]),
        float(example["lag_48"]),
        float(slot),
        float(source_date.weekday()),
        float(source_date.month),
    ]


def fit_and_validate(parts: dict[str, list[dict]], case: str, include_preview: bool = False) -> dict:
    """Fit on train, select tree count on validation, leave test unscored."""

    try:
        import xgboost as xgb
    except ImportError as error:
        raise RuntimeError("Install requirements.txt in a local virtual environment before training") from error

    train, validation = parts["train"], parts["validation"]
    if not train or not validation or not parts["test"]:
        raise ValueError("Chronological train/validation/test examples are required")
    train_matrix = xgb.DMatrix(
        [feature_row(example, case) for example in train],
        label=[example["target"] for example in train],
        feature_names=list(FEATURE_NAMES),
    )
    validation_matrix = xgb.DMatrix(
        [feature_row(example, case) for example in validation],
        label=[example["target"] for example in validation],
        feature_names=list(FEATURE_NAMES),
    )
    parameters = {
        "objective": "reg:squarederror",
        "tree_method": "hist",
        "eval_metric": "mae",
        "max_depth": 4,
        "eta": 0.05,
        "subsample": 0.8,
        "seed": 42,
        "nthread": 2,
    }
    booster = xgb.train(
        parameters,
        train_matrix,
        num_boost_round=500,
        evals=[(validation_matrix, "validation")],
        early_stopping_rounds=30,
        verbose_eval=False,
    )
    raw_predictions = booster.predict(validation_matrix, iteration_range=(0, booster.best_iteration + 1))
    predictions = [max(0.0, float(value)) for value in raw_predictions]
    contributions = booster.predict(
        validation_matrix,
        pred_contribs=True,
        iteration_range=(0, booster.best_iteration + 1),
    )
    if contributions.shape[1] != len(FEATURE_NAMES) + 1:
        raise ValueError("Unexpected XGBoost SHAP contribution shape")
    if abs(float(sum(contributions[0])) - float(raw_predictions[0])) > 1e-3:
        raise ValueError("XGBoost SHAP contributions do not sum to the raw prediction")
    mean_absolute_shap = {
        name: float(sum(abs(float(row[index])) for row in contributions) / len(contributions))
        for index, name in enumerate(FEATURE_NAMES)
    }
    result = {
        "model": "XGBoost CPU, squared-error trees",
        "xgboost_version": xgb.__version__,
        "fixed_parameters": parameters,
        "best_boosting_iteration_selected_on_validation": booster.best_iteration,
        "feature_names": list(FEATURE_NAMES),
        "nonnegative_predictions": True,
        "validation_metrics": evaluate_predictions(validation, predictions),
        "validation_mean_absolute_SHAP_by_feature": mean_absolute_shap,
        "SHAP_scope": "XGBoost raw predictions before nonnegative clipping; validation rows only",
        "validation_first_local_SHAP": {
            "feature_contributions": {name: float(contributions[0][index]) for index, name in enumerate(FEATURE_NAMES)},
            "bias": float(contributions[0][-1]),
            "raw_prediction_before_clipping": float(raw_predictions[0]),
        },
        "validation_seasonal_naive_metrics": evaluate_predictions(
            validation, [example["seasonal_naive"] for example in validation]
        ),
        "validation_persistence_metrics": evaluate_predictions(
            validation, [example["lag_1"] for example in validation]
        ),
        "test_status": "reserved; no test target used in training, early stopping or reported metrics",
    }
    if include_preview:
        result["validation_prediction_rows"] = preview_predictions(validation, predictions, len(validation))
    return result
