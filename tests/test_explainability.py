"""Unit tests for the SHAP explainability module.

Author: Nguyen Hoang Viet (Explainability & QA Lead)
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from energy_forecasting.explainability import (
    compute_shap_contributions,
    global_feature_importance,
    local_explanation,
    top_features,
)


def _small_booster() -> tuple[xgb.Booster, xgb.DMatrix]:
    """Train a tiny XGBoost model on synthetic data for testing."""
    rng = np.random.default_rng(42)
    n = 200
    X = rng.normal(size=(n, 3)).astype(np.float32)
    y = 2.0 * X[:, 0] + 1.5 * X[:, 1] - X[:, 2] + rng.normal(scale=0.1, size=n)
    dmatrix = xgb.DMatrix(X, label=y, feature_names=["f0", "f1", "f2"])
    params = {"objective": "reg:squarederror", "max_depth": 3, "seed": 42}
    booster = xgb.train(params, dmatrix, num_boost_round=20)
    return booster, dmatrix


class ExplainabilityTest(unittest.TestCase):
    def test_shap_shape_matches_features_plus_bias(self):
        booster, dmatrix = _small_booster()
        contributions = compute_shap_contributions(booster, dmatrix)
        self.assertEqual(contributions.shape[1], 4)  # 3 features + bias
        self.assertIn("bias", contributions.columns)
        self.assertEqual(len(contributions), dmatrix.num_row())

    def test_shap_contributions_sum_to_prediction(self):
        booster, dmatrix = _small_booster()
        contributions = compute_shap_contributions(booster, dmatrix)
        raw_pred = booster.predict(dmatrix, output_margin=True)
        summed = contributions.sum(axis=1).to_numpy()
        np.testing.assert_allclose(summed, raw_pred, atol=1e-4)

    def test_global_importance_is_sorted_descending(self):
        booster, dmatrix = _small_booster()
        contributions = compute_shap_contributions(booster, dmatrix)
        importance = global_feature_importance(contributions)
        self.assertNotIn("bias", importance.index)
        self.assertTrue(importance.is_monotonic_decreasing)

    def test_top_features_returns_correct_count(self):
        booster, dmatrix = _small_booster()
        contributions = compute_shap_contributions(booster, dmatrix)
        self.assertEqual(len(top_features(contributions, n=2)), 2)

    def test_local_explanation_structure(self):
        booster, dmatrix = _small_booster()
        contributions = compute_shap_contributions(booster, dmatrix)
        explanation = local_explanation(contributions, row=0)
        self.assertIn("feature_contributions", explanation)
        self.assertIn("bias", explanation)
        self.assertIn("raw_prediction", explanation)
        self.assertEqual(len(explanation["feature_contributions"]), 3)


if __name__ == "__main__":
    unittest.main()