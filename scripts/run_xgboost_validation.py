"""Run: python scripts/run_xgboost_validation.py AEMO_DIRECTORY AUSGRID_ARCHIVE."""

import json
from datetime import date, datetime
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from energy_forecasting.aemo import NEM_TIME, read_operational_demand_months
from energy_forecasting.ausgrid import read_customer_years
from energy_forecasting.baseline import aemo_one_step_examples, ausgrid_next_slot_examples, partition_examples
from energy_forecasting.xgb_model import fit_and_validate


if __name__ == "__main__":
    if len(sys.argv) != 3:
        raise SystemExit("Usage: python scripts/run_xgboost_validation.py AEMO_DIRECTORY AUSGRID_ARCHIVE")
    aemo_archives = sorted(Path(sys.argv[1]).glob("PUBLIC_ACTUAL_OPERATIONAL_DEMAND_DAILY_????????.zip"))
    if len(aemo_archives) != 12:
        raise SystemExit("Expected the documented 12 AEMO monthly archives, August 2025 through July 2026")
    aemo = aemo_one_step_examples(read_operational_demand_months(aemo_archives, "NSW1"))
    pv = ausgrid_next_slot_examples(read_customer_years(sys.argv[2], 1, "GG"))
    aemo_parts = partition_examples(aemo, datetime(2026, 4, 1, 4, 30, tzinfo=NEM_TIME), datetime(2026, 6, 1, 4, 30, tzinfo=NEM_TIME))
    pv_parts = partition_examples(pv, date(2013, 1, 1), date(2013, 4, 1), source_date_key=True)
    print(json.dumps({
        "status": "first fixed-configuration ML experiment; validation only",
        "aemo_NSW1": fit_and_validate(aemo_parts, "aemo"),
        "ausgrid_customer_1_GG": fit_and_validate(pv_parts, "ausgrid"),
        "ausgrid_clock_caveat": "source slots retained; absolute daylight-saving timestamps unresolved",
    }, indent=2))
