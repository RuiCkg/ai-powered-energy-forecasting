"""Tests for dashboard-only presentation conversions."""

from datetime import datetime
import unittest

from energy_forecasting.dashboard import forecast_plot_timestamp


class ForecastPlotTimestampTest(unittest.TestCase):
    def test_aemo_iso_timestamp_is_preserved(self):
        result = forecast_plot_timestamp("aemo", "2026-04-01T04:30:00+10:00")
        self.assertEqual(result.isoformat(), "2026-04-01T04:30:00+10:00")

    def test_ausgrid_slot_is_converted_to_a_real_half_hour_timestamp(self):
        result = forecast_plot_timestamp("ausgrid", "2012-07-01 slot 03")
        self.assertEqual(result, datetime(2012, 7, 1, 1, 30))

    def test_invalid_ausgrid_label_is_rejected_instead_of_silently_blank(self):
        with self.assertRaisesRegex(ValueError, "Invalid Ausgrid target key"):
            forecast_plot_timestamp("ausgrid", "2012-07-01")


if __name__ == "__main__":
    unittest.main()
