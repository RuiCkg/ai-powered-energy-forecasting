import io
from pathlib import Path
import sys
import unittest
import zipfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from energy_forecasting.ausgrid import profile_annual_archive, profile_daily_rows, read_daily_rows


class AusgridParserTest(unittest.TestCase):
    def test_wide_row_and_quality_without_absolute_timestamp(self):
        clocks = [f"{(i + 1) // 2 % 24}:{'30' if i % 2 == 0 else '00'}" for i in range(48)]
        header = ["Customer", "Generator Capacity", "Postcode", "Consumption Category", "date", *clocks, "Row Quality"]
        data = ["1", "3.78", "2076", "GG", "1/07/2012", *(["0.5"] * 48), "NA"]
        csv_data = "notice\n" + ",".join(header) + "\n" + ",".join(data) + "\n"
        archive = io.BytesIO()
        with zipfile.ZipFile(archive, "w") as z:
            z.writestr("Solar home 2012-2013.csv", csv_data)
        archive.seek(0)
        rows = read_daily_rows(archive)
        profile = profile_daily_rows(rows)
        self.assertEqual(profile["interval_count"], 48)
        self.assertEqual(profile["non_actual_row_count"], 1)
        self.assertEqual(rows[0]["clock_labels"][0], "0:30")
        self.assertIn("pending", profile["absolute_timestamp_status"])

        archive.seek(0)
        annual = profile_annual_archive(archive)
        self.assertEqual(annual["category_totals"]["GG"]["non_actual"], 1)
        self.assertEqual(annual["gg_full_year_no_blank_actual_customer_count"], 0)

    def test_rejects_non_finite_or_negative_pv_energy(self):
        clocks = [f"slot{i}" for i in range(48)]
        header = ["Customer", "Generator Capacity", "Postcode", "Consumption Category", "date", *clocks, "Row Quality"]
        for bad_value in ("inf", "-0.5"):
            with self.subTest(bad_value=bad_value):
                data = ["1", "3.78", "2076", "GG", "1/07/2012", bad_value, *(["0.5"] * 47), ""]
                archive = io.BytesIO()
                with zipfile.ZipFile(archive, "w") as z:
                    z.writestr("Solar home 2012-2013.csv", "notice\n" + ",".join(header) + "\n" + ",".join(data) + "\n")
                archive.seek(0)
                with self.assertRaisesRegex(ValueError, "Non-finite or negative Ausgrid energy reading"):
                    read_daily_rows(archive)


if __name__ == "__main__":
    unittest.main()
