from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from energy_forecasting.demo_data import EXPECTED_AEMO_NAMES, prepare_uploaded_sources


class DemoInputContractTest(unittest.TestCase):
    def test_rejects_missing_or_unexpected_monthly_archive_names_before_reading(self):
        self.assertEqual(len(EXPECTED_AEMO_NAMES), 12)
        with self.assertRaisesRegex(ValueError, "Missing:"):
            prepare_uploaded_sources([], None)


if __name__ == "__main__":
    unittest.main()
