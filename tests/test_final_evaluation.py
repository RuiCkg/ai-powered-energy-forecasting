from __future__ import annotations

from datetime import datetime, timedelta, timezone
import importlib.util
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from energy_forecasting.lstm_model import fit_and_evaluate_test_lstm, fit_and_validate_lstm
from energy_forecasting.xgb_model import fit_and_evaluate_test, fit_and_validate


def example(index: int) -> dict:
    seasonal_period = 8
    target = 1.0 + (index % seasonal_period) / 10
    return {
        "key": datetime(2026, 1, 1, tzinfo=timezone.utc) + timedelta(hours=index),
        "target": target,
        "lag_1": 1.0 + ((index - 1) % seasonal_period) / 10,
        "lag_48": target,
        "seasonal_naive": target,
        "history_48": tuple(1.0 + ((index - offset) % seasonal_period) / 10 for offset in range(8, 0, -1)),
        "target_slot": index % seasonal_period,
        "slots_per_season": seasonal_period,
        "custom_covariates": {"temperature": 20.0 + index / 100},
    }


def parts() -> dict:
    examples = [example(index) for index in range(90)]
    return {"train": examples[:50], "validation": examples[50:70], "test": examples[70:]}


class FinalEvaluationTest(unittest.TestCase):
    def test_xgboost_uses_validation_selected_rounds_on_reserved_test(self):
        validation = fit_and_validate(parts(), "custom")
        result = fit_and_evaluate_test(
            parts(), "custom", validation["best_iteration"] + 1, include_preview=True
        )
        self.assertEqual(result["test_metrics"]["example_count"], 20)
        self.assertEqual(result["evaluation_partition"], "reserved test")
        self.assertEqual(len(result["test_prediction_rows"]), 20)

    @unittest.skipUnless(importlib.util.find_spec("torch"), "PyTorch is an optional test dependency")
    def test_lstm_uses_validation_selected_epochs_on_reserved_test(self):
        validation = fit_and_validate_lstm(parts(), "custom")
        result = fit_and_evaluate_test_lstm(
            parts(), "custom", validation["best_epoch"], include_preview=True
        )
        self.assertEqual(result["test_metrics"]["example_count"], 20)
        self.assertEqual(result["evaluation_partition"], "reserved test")
        self.assertEqual(len(result["test_prediction_rows"]), 20)


if __name__ == "__main__":
    unittest.main()
