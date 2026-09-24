"""Run: python scripts/profile_aemo_months.py ARCHIVE_DIRECTORY [REGION]."""

import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from energy_forecasting.aemo import profile_records, read_operational_demand_months


if __name__ == "__main__":
    if not 2 <= len(sys.argv) <= 3:
        raise SystemExit("Usage: python scripts/profile_aemo_months.py ARCHIVE_DIRECTORY [REGION]")
    directory = Path(sys.argv[1])
    archives = sorted(directory.glob("PUBLIC_ACTUAL_OPERATIONAL_DEMAND_DAILY_????????.zip"))
    if not archives:
        raise SystemExit("No AEMO ACTUAL_DAILY monthly ZIP files found")
    region = sys.argv[2] if len(sys.argv) == 3 else "NSW1"
    result = profile_records(read_operational_demand_months(archives, region))
    result["archive_count"] = len(archives)
    result["archive_months"] = [archive.stem[-8:-2] for archive in archives]
    print(json.dumps(result, indent=2))
