import io
from pathlib import Path
import sys
import unittest
import zipfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from energy_forecasting.aemo import profile_records, read_operational_demand, read_operational_demand_months


CSV = "\n".join(
    [
        "C,NEMP.WORLD,ACTUAL_OPERATIONAL_DEMAND_DAILY,AEMO,PUBLIC,2026/07/02,04:40:01,1,2,3",
        "I,OPERATIONAL_DEMAND,ACTUAL,3,REGIONID,INTERVAL_DATETIME,OPERATIONAL_DEMAND,OPERATIONAL_DEMAND_ADJUSTMENT,WDR_ESTIMATE,LASTCHANGED",
        'D,OPERATIONAL_DEMAND,ACTUAL,3,NSW1,"2026/07/01 04:30:00",7135,0,0,"2026/07/01 04:30:02"',
        'D,OPERATIONAL_DEMAND,ACTUAL,3,NSW1,"2026/07/01 05:00:00",7200,0,0,"2026/07/01 05:00:02"',
        'D,OPERATIONAL_DEMAND,ACTUAL,3,QLD1,"2026/07/01 04:30:00",6000,0,0,"2026/07/01 04:30:02"',
    ]
)


class AemoParserTest(unittest.TestCase):
    def test_nested_archive_and_region_filter(self):
        daily_bytes = io.BytesIO()
        with zipfile.ZipFile(daily_bytes, "w") as daily:
            daily.writestr("daily.CSV", CSV)
        monthly_bytes = io.BytesIO()
        with zipfile.ZipFile(monthly_bytes, "w") as monthly:
            monthly.writestr("day.zip", daily_bytes.getvalue())
        monthly_bytes.seek(0)
        records = read_operational_demand(monthly_bytes)
        self.assertEqual(len(records), 2)
        self.assertEqual([record["target"] for record in records], [7135.0, 7200.0])
        self.assertEqual(records[0]["timestamp"].isoformat(), "2026-07-01T04:30:00+10:00")
        self.assertEqual(profile_records(records)["non_30_minute_gaps"], 0)
        monthly_bytes.seek(0)
        with self.assertRaisesRegex(ValueError, "across AEMO monthly archives"):
            read_operational_demand_months([monthly_bytes, monthly_bytes])

    def test_rejects_non_finite_or_negative_demand(self):
        for bad_value in ("nan", "-1"):
            with self.subTest(bad_value=bad_value):
                daily_bytes = io.BytesIO()
                with zipfile.ZipFile(daily_bytes, "w") as daily:
                    daily.writestr("daily.CSV", CSV.replace(",7135,", f",{bad_value},"))
                monthly_bytes = io.BytesIO()
                with zipfile.ZipFile(monthly_bytes, "w") as monthly:
                    monthly.writestr("day.zip", daily_bytes.getvalue())
                monthly_bytes.seek(0)
                with self.assertRaisesRegex(ValueError, "Non-finite or negative AEMO demand"):
                    read_operational_demand(monthly_bytes)


if __name__ == "__main__":
    unittest.main()
