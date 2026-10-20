from __future__ import annotations

import unittest
from pathlib import Path
import sys

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from energy_forecasting.custom_data import (
    build_synthetic_demo_csv,
    infer_daily_seasonal_period,
    inspect_custom_csv,
    prepare_custom_csv,
)


def sample_csv(rows: int = 160) -> bytes:
    timestamps = pd.date_range("2026-01-01", periods=rows, freq="30min")
    frame = pd.DataFrame({
        "when": timestamps.astype(str),
        "energy": [float(index % 48) for index in range(rows)],
        "temperature": [20.0 + index / 100 for index in range(rows)],
    })
    return frame.to_csv(index=False).encode("utf-8")


class CustomCSVTest(unittest.TestCase):
    def test_synthetic_demo_is_a_complete_valid_example(self):
        data = build_synthetic_demo_csv()
        inspection = inspect_custom_csv("synthetic_energy_demo.csv", data)
        self.assertEqual(inspection.row_count, 672)
        self.assertEqual(
            inspection.columns,
            ("timestamp", "energy_kwh", "temperature_c"),
        )
        case, evidence = prepare_custom_csv(
            "synthetic_energy_demo.csv",
            data,
            "timestamp",
            "energy_kwh",
            ["temperature_c"],
            seasonal_period=48,
            unit="kWh / half-hour",
        )
        self.assertTrue(all(case["parts"].values()))
        self.assertEqual(evidence["accepted_upload_count"], 1)

    def test_inspection_and_daily_period_inference(self):
        data = sample_csv()
        inspection = inspect_custom_csv("sample.csv", data)
        self.assertEqual(inspection.columns, ("when", "energy", "temperature"))
        self.assertEqual(inspection.row_count, 160)
        self.assertEqual(infer_daily_seasonal_period("sample.csv", data, "when"), 48)

    def test_prepare_maps_covariates_and_keeps_test_reserved(self):
        case, evidence = prepare_custom_csv(
            "sample.csv",
            sample_csv(),
            "when",
            "energy",
            ["temperature"],
            seasonal_period=48,
            unit="kWh",
        )
        self.assertEqual(case["profile"]["forecast_horizon_rows"], 1)
        self.assertEqual(case["unit"], "kWh")
        self.assertEqual(set(case["parts"]), {"train", "validation", "test"})
        self.assertEqual(case["parts"]["train"][0]["custom_covariates"], {"temperature": 20.48})
        self.assertEqual(len(case["parts"]["train"][0]["history_48"]), 48)
        self.assertEqual(evidence["accepted_upload_count"], 1)

    def test_duplicate_timestamp_is_rejected(self):
        data = sample_csv().decode("utf-8").splitlines()
        data[2] = data[1]
        with self.assertRaisesRegex(ValueError, "duplicate timestamps"):
            prepare_custom_csv(
                "sample.csv",
                ("\n".join(data) + "\n").encode("utf-8"),
                "when",
                "energy",
                seasonal_period=48,
            )

    def test_missing_target_is_rejected(self):
        data = sample_csv().decode("utf-8").replace(",1.0,", ",,")
        with self.assertRaisesRegex(ValueError, "blank or non-numeric"):
            prepare_custom_csv(
                "sample.csv", data.encode("utf-8"), "when", "energy", seasonal_period=48
            )


if __name__ == "__main__":
    unittest.main()
