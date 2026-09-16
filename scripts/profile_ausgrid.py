"""Run: python scripts/profile_ausgrid.py ARCHIVE [CUSTOMER] [CATEGORY] [YEAR_FILE]."""

from __future__ import annotations

import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from energy_forecasting.ausgrid import profile_daily_rows, read_daily_rows


def main() -> None:
    if not 2 <= len(sys.argv) <= 5:
        raise SystemExit("Usage: python scripts/profile_ausgrid.py ARCHIVE [CUSTOMER] [CATEGORY] [YEAR_FILE]")
    archive = sys.argv[1]
    customer = int(sys.argv[2]) if len(sys.argv) >= 3 else 1
    category = sys.argv[3] if len(sys.argv) >= 4 else "GG"
    year_file = sys.argv[4] if len(sys.argv) >= 5 else "Solar home 2012-2013.csv"
    print(json.dumps(profile_daily_rows(read_daily_rows(archive, customer, category, year_file)), indent=2))


if __name__ == "__main__":
    main()
