"""Run: python scripts/profile_splits.py AEMO_DIRECTORY AUSGRID_ARCHIVE."""

import json
from datetime import date, datetime
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from energy_forecasting.aemo import NEM_TIME, read_operational_demand_months
from energy_forecasting.ausgrid import profile_daily_rows, read_customer_years
from energy_forecasting.splits import split_aemo_records, split_ausgrid_days


def describe(parts: dict, key: str, multiplier: int = 1) -> dict:
    return {
        name: {
            "source_row_count": len(rows),
            "half_hour_slot_count": len(rows) * multiplier,
            "first": rows[0][key].isoformat(),
            "last": rows[-1][key].isoformat(),
        }
        for name, rows in parts.items()
    }


if __name__ == "__main__":
    if len(sys.argv) != 3:
        raise SystemExit("Usage: python scripts/profile_splits.py AEMO_DIRECTORY AUSGRID_ARCHIVE")
    aemo_archives = sorted(Path(sys.argv[1]).glob("PUBLIC_ACTUAL_OPERATIONAL_DEMAND_DAILY_????????.zip"))
    if len(aemo_archives) != 12:
        raise SystemExit("Expected the documented 12 AEMO monthly archives, August 2025 through July 2026")
    aemo = read_operational_demand_months(aemo_archives, "NSW1")
    pv = read_customer_years(sys.argv[2], 1, "GG")
    pv_quality = profile_daily_rows(pv)
    if any(pv_quality[key] for key in ("non_1_day_gaps", "blank_interval_count", "non_actual_row_count", "row_quality_not_provided_count")):
        raise SystemExit("The documented Ausgrid candidate does not meet its source-quality selection rule")
    aemo_parts = split_aemo_records(
        aemo,
        datetime(2026, 4, 1, 4, 30, tzinfo=NEM_TIME),
        datetime(2026, 6, 1, 4, 30, tzinfo=NEM_TIME),
    )
    pv_parts = split_ausgrid_days(pv, date(2013, 1, 1), date(2013, 4, 1))
    print(json.dumps({
        "status": "provisional source-record split; no forecast windows or model training",
        "aemo_nsw1": describe(aemo_parts, "timestamp"),
        "ausgrid_customer_1_gg": describe(pv_parts, "date", 48),
        "ausgrid_timestamp_status": "source dates and slots only; DST policy pending",
    }, indent=2))
