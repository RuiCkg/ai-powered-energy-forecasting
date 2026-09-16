"""Run: python scripts/run_seasonal_baseline.py AEMO_DIRECTORY AUSGRID_ARCHIVE."""

import json
from datetime import date, datetime
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from energy_forecasting.aemo import NEM_TIME, read_operational_demand_months
from energy_forecasting.ausgrid import read_customer_years
from energy_forecasting.baseline import aemo_one_step_examples, ausgrid_next_slot_examples, evaluate_persistence, evaluate_seasonal_naive, partition_examples


def score(parts: dict) -> dict:
    return {
        name: {
            "seasonal_naive_previous_day": evaluate_seasonal_naive(examples),
            "persistence_previous_slot": evaluate_persistence(examples),
        }
        for name, examples in parts.items()
    }


if __name__ == "__main__":
    if len(sys.argv) != 3:
        raise SystemExit("Usage: python scripts/run_seasonal_baseline.py AEMO_DIRECTORY AUSGRID_ARCHIVE")
    aemo_archives = sorted(Path(sys.argv[1]).glob("PUBLIC_ACTUAL_OPERATIONAL_DEMAND_DAILY_????????.zip"))
    if len(aemo_archives) != 12:
        raise SystemExit("Expected the documented 12 AEMO monthly archives, August 2025 through July 2026")
    aemo = read_operational_demand_months(aemo_archives, "NSW1")
    pv = read_customer_years(sys.argv[2], 1, "GG")
    aemo_examples = aemo_one_step_examples(aemo)
    pv_examples = ausgrid_next_slot_examples(pv)
    aemo_parts = partition_examples(aemo_examples, datetime(2026, 4, 1, 4, 30, tzinfo=NEM_TIME), datetime(2026, 6, 1, 4, 30, tzinfo=NEM_TIME))
    # Source-date partition: all target slots of a day stay together.
    pv_parts = partition_examples(pv_examples, date(2013, 1, 1), date(2013, 4, 1), source_date_key=True)
    print(json.dumps({
        "baselines": "previous-day same half-hour/source slot; previous immediate half-hour/source slot",
        "forecast_task": "one-step AEMO half-hour; next within-day Ausgrid source slot",
        "aemo_NSW1_MW": score(aemo_parts),
        "ausgrid_customer_1_GG_kWh": score(pv_parts),
        "caveat": "Ausgrid absolute DST timestamps unresolved; ordinary PV MAPE excludes zero actuals",
    }, indent=2))
