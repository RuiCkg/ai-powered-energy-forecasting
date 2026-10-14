"""Run: python scripts/run_xgboost_validation.py AEMO_DIRECTORY AUSGRID_ARCHIVE."""

import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from energy_forecasting.demo_data import prepare_uploaded_sources
from energy_forecasting.xgb_model import fit_and_validate


class DiskFile:
    """Minimal stand-in for a Streamlit upload."""

    def __init__(self, path: Path):
        self.name, self._path = path.name, path

    def getvalue(self) -> bytes:
        return self._path.read_bytes()


if __name__ == "__main__":
    if len(sys.argv) != 3:
        raise SystemExit("Usage: python scripts/run_xgboost_validation.py AEMO_DIRECTORY AUSGRID_ARCHIVE")
    aemo = [DiskFile(p) for p in sorted(Path(sys.argv[1]).glob("PUBLIC_ACTUAL_OPERATIONAL_DEMAND_DAILY_????????.zip"))]
    prepared = prepare_uploaded_sources(aemo, DiskFile(Path(sys.argv[2])))
    output = {"scope": "validation only; test partition not scored"}
    for case in ("aemo", "ausgrid"):
        result = fit_and_validate(prepared[case]["parts"], case)
        result.pop("validation_prediction_rows")
        output[case] = result
    print(json.dumps(output, indent=2))
