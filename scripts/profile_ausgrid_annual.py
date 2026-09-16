"""Run: python scripts/profile_ausgrid_annual.py ARCHIVE [YEAR_FILE]."""

import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from energy_forecasting.ausgrid import profile_annual_archive


if __name__ == "__main__":
    if not 2 <= len(sys.argv) <= 3:
        raise SystemExit("Usage: python scripts/profile_ausgrid_annual.py ARCHIVE [YEAR_FILE]")
    year_file = sys.argv[2] if len(sys.argv) == 3 else "Solar home 2012-2013.csv"
    print(json.dumps(profile_annual_archive(sys.argv[1], year_file), indent=2))
