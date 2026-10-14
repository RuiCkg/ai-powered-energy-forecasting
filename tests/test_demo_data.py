from datetime import date, datetime, timedelta
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from energy_forecasting.demo_data import (
    AEMO_BOUNDARIES,
    AUSGRID_BOUNDARIES,
    EXPECTED_AEMO_NAMES,
    prepare_uploaded_sources,
)


class DemoInputContractTest(unittest.TestCase):
    def test_rejects_missing_or_unexpected_monthly_archive_names_before_reading(self):
        self.assertEqual(len(EXPECTED_AEMO_NAMES), 12)
        with self.assertRaisesRegex(ValueError, "Missing:"):
            prepare_uploaded_sources([], None)

    def test_recorded_boundaries_match_each_dataset_partition(self):
        self.assertIsInstance(AEMO_BOUNDARIES["validation_start"], datetime)
        self.assertEqual(AEMO_BOUNDARIES["validation_start"].utcoffset(), timedelta(hours=10))
        self.assertEqual(AEMO_BOUNDARIES["test_start"].date(), date(2026, 6, 1))

        self.assertNotIsInstance(AUSGRID_BOUNDARIES["validation_start"], datetime)
        self.assertEqual(AUSGRID_BOUNDARIES["validation_start"], date(2013, 1, 1))
        self.assertEqual(AUSGRID_BOUNDARIES["test_start"], date(2013, 4, 1))


if __name__ == "__main__":
    unittest.main()
