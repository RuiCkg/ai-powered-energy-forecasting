from datetime import date, datetime
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from energy_forecasting.aemo import NEM_TIME
from energy_forecasting.xgb_model import FEATURE_NAMES, feature_row


class XgbFeaturesTest(unittest.TestCase):
    def test_known_calendar_and_past_lags(self):
        aemo = {"key": datetime(2026, 4, 1, 4, 30, tzinfo=NEM_TIME), "lag_1": 10, "lag_48": 20}
        self.assertEqual(feature_row(aemo, "aemo"), [10.0, 20.0, 9.0, 2.0, 4.0])
        pv = {"key": (date(2013, 1, 1), 2), "lag_1": 0, "lag_48": 0.5}
        self.assertEqual(feature_row(pv, "ausgrid"), [0.0, 0.5, 2.0, 1.0, 1.0])
        self.assertEqual(len(FEATURE_NAMES), 5)


if __name__ == "__main__":
    unittest.main()
