"""SHAP-based explainability for XGBoost forecasting.

Author: Nguyen Hoang Viet (Explainability & QA Lead)
Purpose: Provide global and local feature attributions for the XGBoost
forecasting model using XGBoost's native pred_contribs (exact for trees).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import xgboost as xgb


def compute_shap_contributions(
    booster: xgb.Booster,
    dmatrix: xgb.DMatrix,
) -> pd.DataFrame:
    """Return per-feature SHAP contributions for XGBoost predictions.

    Uses XGBoost's native pred_contribs, which is exact for tree models.
    The final column is the bias term (base value).
    """
    contributions = booster.predict(dmatrix, pred_contribs=True)
    feature_names = list(booster.feature_names or []) + ["bias"]
    if contributions.shape[1] != len(feature_names):
        raise ValueError(
            f"SHAP shape {contributions.shape} does not match "
            f"feature names {len(feature_names)}"
        )
    return pd.DataFrame(contributions, columns=feature_names)


def global_feature_importance(contributions: pd.DataFrame) -> pd.Series:
    """Mean absolute SHAP value per feature, excluding the bias term.

    Higher values indicate features that influence predictions more
    strongly on average across the evaluated rows.
    """
    if "bias" in contributions.columns:
        features = contributions.drop(columns=["bias"])
    else:
        features = contributions
    return features.abs().mean().sort_values(ascending=False)


def top_features(contributions: pd.DataFrame, n: int = 5) -> list[str]:
    """Return the top-n feature names by mean absolute SHAP value."""
    return global_feature_importance(contributions).head(n).index.tolist()


def local_explanation(contributions: pd.DataFrame, row: int = 0) -> dict[str, Any]:
    """Return one row's feature contributions plus the bias (base value).

    These are associations from the fitted model, not causal effects.
    """
    if row < 0 or row >= len(contributions):
        raise IndexError(f"Row {row} out of range for {len(contributions)} rows")
    row_values = contributions.iloc[row]
    return {
        "feature_contributions": row_values.drop(labels=["bias"]).to_dict(),
        "bias": float(row_values.get("bias", 0.0)),
        "raw_prediction": float(row_values.sum()),
    }


def export_global_importance(
    contributions: pd.DataFrame,
    output_path: str | Path,
) -> None:
    """Save global feature importance to CSV for reporting."""
    importance = global_feature_importance(contributions)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    importance.to_csv(output_path, header=["mean_absolute_shap"])