"""Run: python scripts/profile_aemo.py PATH_TO_MONTHLY_ZIP [REGION]."""

from __future__ import annotations

import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from energy_forecasting.aemo import profile_records, read_operational_demand


def main() -> None:
    if len(sys.argv) not in (2, 3):
        raise SystemExit("Usage: python scripts/profile_aemo.py PATH_TO_MONTHLY_ZIP [REGION]")
    region = sys.argv[2] if len(sys.argv) == 3 else "NSW1"
    print(json.dumps(profile_records(read_operational_demand(sys.argv[1], region)), indent=2))


if __name__ == "__main__":
    main()
