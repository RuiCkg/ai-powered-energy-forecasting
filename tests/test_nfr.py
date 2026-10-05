"""NFR verification tests (Week 2 Activity 3 requirements).

Author: Nguyen Hoang Viet (Explainability & QA Lead)

Verifies:
- NFR2: training < 60 seconds for 10,000 records per model
- SHAP generation < 120 seconds
- Reproducibility with fixed seed
"""

from __future__ import annotations

import sys
import time
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from energy_forecasting.explainability import compute_shap_contributions


class NFRVerificationTest(unittest.TestCase):
    def test_nfr2_xgboost_training_under_60_seconds(self):
        """NFR2: XGBoost training on ~10,000 records < 60s."""
        rng = np.random.default_rng(42)
        n = 10_000
        X = rng.normal(size=(n, 5)).astype(np.float32)
        y = X[:, 0] + 0.5 * X[:, 1] + rng.normal(scale=0.1, size=n)
        dmatrix = xgb.DMatrix(X, label=y, feature_names=[f"f{i}" for i in range(5)])
        params = {"objective": "reg:squarederror", "max_depth": 4, "seed": 42}

        start = time.perf_counter()
        xgb.train(params, dmatrix, num_boost_round=100)
        elapsed = time.perf_counter() - start

        self.assertLess(
            elapsed, 60.0,
            f"NFR2 violation: XGBoost training took {elapsed:.2f}s (>60s)",
        )

    def test_shap_generation_under_120_seconds(self):
        """SHAP for 10,000 rows should complete within 120 seconds."""
        rng = np.random.default_rng(42)
        n = 10_000
        X = rng.normal(size=(n, 5)).astype(np.float32)
        y = X[:, 0] + rng.normal(scale=0.1, size=n)
        dmatrix = xgb.DMatrix(X, label=y, feature_names=[f"f{i}" for i in range(5)])
        params = {"objective": "reg:squarederror", "max_depth": 3, "seed": 42}
        booster = xgb.train(params, dmatrix, num_boost_round=50)

        start = time.perf_counter()
        contributions = compute_shap_contributions(booster, dmatrix)
        elapsed = time.perf_counter() - start

        self.assertEqual(contributions.shape[0], n)
        self.assertLess(
            elapsed, 120.0,
            f"SHAP generation took {elapsed:.2f}s (>120s)",
        )

    def test_reproducibility_with_fixed_seed(self):
        """Same seed must produce identical SHAP values."""
        rng = np.random.default_rng(42)
        n = 100
        X = rng.normal(size=(n, 3)).astype(np.float32)
        y = X[:, 0] + rng.normal(scale=0.1, size=n)
        dmatrix = xgb.DMatrix(X, label=y, feature_names=["a", "b", "c"])
        params = {"objective": "reg:squarederror", "max_depth": 3, "seed": 42}
        booster = xgb.train(params, dmatrix, num_boost_round=20)

        first = compute_shap_contributions(booster, dmatrix)
        second = compute_shap_contributions(booster, dmatrix)
        pd.testing.assert_frame_equal(first, second)


if __name__ == "__main__":
    unittest.main()