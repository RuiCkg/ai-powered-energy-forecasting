from datetime import date, datetime, timedelta
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from energy_forecasting.aemo import NEM_TIME
from energy_forecasting.baseline import aemo_one_step_examples, ausgrid_next_slot_examples, evaluate_persistence, evaluate_seasonal_naive


class BaselineTest(unittest.TestCase):
    def test_aemo_uses_only_prior_day_for_prediction(self):
        start = datetime(2026, 1, 1, tzinfo=NEM_TIME)
        records = [{"timestamp": start + timedelta(minutes=30 * i), "target": float(i)} for i in range(50)]
        examples = aemo_one_step_examples(records)
        self.assertEqual(len(examples), 2)
        self.assertEqual(examples[0]["target"], 48.0)
        self.assertEqual(examples[0]["lag_1"], 47.0)
        self.assertEqual(examples[0]["seasonal_naive"], 0.0)
        self.assertAlmostEqual(evaluate_persistence(examples)["MAE"], 1.0)

    def test_pv_within_day_origin_and_zero_mape_policy(self):
        rows = [
            {"date": date(2013, 1, day), "values_kWh": tuple([0.0] + [float(day)] * 47),
             "clock_labels": tuple(str(i) for i in range(48)), "row_quality": ""}
            for day in (1, 2)
        ]
        examples = ausgrid_next_slot_examples(rows)
        self.assertEqual(len(examples), 47)
        self.assertEqual(examples[0]["lag_1"], 0.0)
        self.assertEqual(examples[0]["seasonal_naive"], 1.0)
        result = evaluate_seasonal_naive(examples)
        self.assertEqual(result["zero_actual_count_excluded_from_MAPE"], 0)
        self.assertAlmostEqual(result["MAE"], 1.0)
        zero_result = evaluate_seasonal_naive([{"target": 0.0, "seasonal_naive": 1.0}])
        self.assertIsNone(zero_result["MAPE_positive_actual_percent"])
        self.assertEqual(zero_result["zero_actual_count_excluded_from_MAPE"], 1)


if __name__ == "__main__":
    unittest.main()
