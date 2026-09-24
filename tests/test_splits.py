from datetime import date, datetime, timedelta, timezone
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from energy_forecasting.aemo import NEM_TIME
from energy_forecasting.splits import split_aemo_records, split_ausgrid_days


class ChronologicalSplitsTest(unittest.TestCase):
    def test_aemo_timestamp_boundaries(self):
        start = datetime(2026, 1, 1, tzinfo=NEM_TIME)
        records = [{"timestamp": start + timedelta(minutes=30 * i)} for i in range(6)]
        parts = split_aemo_records(records, records[2]["timestamp"], records[4]["timestamp"])
        self.assertEqual([len(parts[name]) for name in ("train", "validation", "test")], [2, 2, 2])
        self.assertLess(parts["train"][-1]["timestamp"], parts["validation"][0]["timestamp"])
        with self.assertRaisesRegex(ValueError, r"fixed NEM UTC\+10"):
            split_aemo_records(records, records[2]["timestamp"].astimezone(timezone.utc), records[4]["timestamp"])

    def test_ausgrid_whole_source_days(self):
        rows = [{"date": date(2013, 1, i)} for i in range(1, 7)]
        parts = split_ausgrid_days(rows, date(2013, 1, 3), date(2013, 1, 5))
        self.assertEqual([len(parts[name]) for name in ("train", "validation", "test")], [2, 2, 2])
        with self.assertRaisesRegex(ValueError, "unique, chronological"):
            split_ausgrid_days(rows[::-1], date(2013, 1, 3), date(2013, 1, 5))


if __name__ == "__main__":
    unittest.main()
