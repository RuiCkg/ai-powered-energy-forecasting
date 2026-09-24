from datetime import date, datetime, timedelta
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from energy_forecasting.aemo import NEM_TIME
from energy_forecasting.baseline import aemo_one_step_examples, ausgrid_next_slot_examples
from energy_forecasting.sequences import aemo_history_windows, ausgrid_history_windows


class SequenceWindowsTest(unittest.TestCase):
    def test_aemo_window_is_past_only(self):
        start = datetime(2026, 1, 1, tzinfo=NEM_TIME)
        records = [{"timestamp": start + timedelta(minutes=30 * i), "target": float(i)} for i in range(49)]
        example = aemo_history_windows(records, aemo_one_step_examples(records))[0]
        self.assertEqual(example["history_48"], tuple(float(i) for i in range(48)))
        self.assertNotIn(example["target"], example["history_48"])

    def test_pv_window_ends_at_previous_source_slot(self):
        rows = [
            {"date": date(2013, 1, day), "values_kWh": tuple(float(i + 48 * (day - 1)) for i in range(48)),
             "clock_labels": tuple(str(i) for i in range(48)), "row_quality": ""}
            for day in (1, 2)
        ]
        example = ausgrid_history_windows(rows, ausgrid_next_slot_examples(rows))[0]
        self.assertEqual(example["target"], 49.0)
        self.assertEqual(example["history_48"][0], 1.0)
        self.assertEqual(example["history_48"][-1], 48.0)


if __name__ == "__main__":
    unittest.main()
