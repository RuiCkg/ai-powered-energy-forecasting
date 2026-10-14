"""Check NFR5: replay a recorded configuration and compare the figures exactly.

Two ways to run it.

Compare two records already downloaded from the dashboard:

    python scripts/replay_check.py FIRST_RECORD.json SECOND_RECORD.json

Run the whole pipeline twice from source archives and compare the two records it
produces, which needs no dashboard and no manual steps:

    python scripts/replay_check.py --from-sources SOURCE_DIRECTORY [--case aemo|ausgrid]

The criterion is exact. Equal configuration digests with equal metric digests
pass. Equal configuration with unequal metrics is a reproducibility failure and
the differing field is named. Unequal configuration means the two runs were not
the same experiment, which is a different finding and is reported as such.

Exit status is 0 only on an exact reproduction, so this works in a check as well
as by hand.
"""

import json
from pathlib import Path
import sys

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from energy_forecasting.experiment_record import build_experiment_record, compare_records  # noqa: E402


class Upload:
    """Minimal stand-in for a Streamlit upload: a name and its bytes."""

    def __init__(self, path: Path) -> None:
        self.name = path.name
        self._data = path.read_bytes()

    def getvalue(self) -> bytes:
        return self._data


def run_once(source_directory: Path, case: str) -> dict:
    """Walk the same path the app walks, and return the resulting record."""

    from energy_forecasting.demo_data import EXPECTED_AEMO_NAMES, prepare_uploaded_sources
    from energy_forecasting.upload_validation import validate_documented_sources
    from energy_forecasting.xgb_model import fit_and_validate

    aemo_paths = sorted(source_directory / name for name in EXPECTED_AEMO_NAMES)
    missing = [path.name for path in aemo_paths if not path.exists()]
    if missing:
        raise SystemExit(f"Missing {len(missing)} documented AEMO archive(s) in {source_directory}, first: {missing[0]}")
    ausgrid_candidates = [
        path for path in sorted(source_directory.glob("*.zip"))
        if path.name not in EXPECTED_AEMO_NAMES
    ]
    if len(ausgrid_candidates) != 1:
        raise SystemExit(
            f"Expected exactly one non-AEMO ZIP in {source_directory} for the Ausgrid archive, "
            f"found {len(ausgrid_candidates)}"
        )

    aemo_files = [Upload(path) for path in aemo_paths]
    ausgrid_file = Upload(ausgrid_candidates[0])

    evidence = validate_documented_sources(aemo_files, ausgrid_file)
    prepared = prepare_uploaded_sources(aemo_files, ausgrid_file)
    case_data = prepared[case]
    parts = case_data["parts"]
    results = {"XGBoost": fit_and_validate(parts, case, include_preview=False)}
    return build_experiment_record(
        case,
        evidence,
        case_data["profile"],
        case_data["boundaries"],
        {name: len(rows) for name, rows in parts.items()},
        results,
    )


def report(first: dict, second: dict, labels: tuple[str, str]) -> int:
    outcome = compare_records(first, second)
    print(f"{'':22}{labels[0]:<68}{labels[1]}")
    for field in ("configuration_digest", "metrics_digest"):
        print(f"{field:<22}{first.get(field, '(absent)'):<68}{second.get(field, '(absent)')}")
    print()
    print(f"configuration matches : {outcome['configuration_matches']}")
    print(f"metrics match         : {outcome['metrics_match']}")
    print(f"replay reproduced     : {outcome['replay_reproduced']}")

    if outcome["replay_reproduced"]:
        for name, model in sorted(first.get("metrics", {}).items()):
            metrics = model.get("validation_metrics", {})
            if metrics:
                print(f"\n{name} validation MAE {metrics.get('MAE')!r}, RMSE {metrics.get('RMSE')!r} "
                      f"(identical in both runs)")
        print("\nNFR5 satisfied for this pair: the recorded configuration replayed to the same figures.")
        return 0

    print(f"\n{len(outcome['differing_paths'])} differing field(s):")
    for path in outcome["differing_paths"][:40]:
        print(f"  {path}")
    if len(outcome["differing_paths"]) > 40:
        print(f"  ... and {len(outcome['differing_paths']) - 40} more")
    if outcome["configuration_matches"] and not outcome["metrics_match"]:
        print("\nREPRODUCIBILITY FAILURE: same configuration, different figures.")
    else:
        print("\nThe two runs are not the same experiment, so their figures were never "
              "required to match. Compare the configuration fields above first.")
    return 1


if __name__ == "__main__":
    arguments = sys.argv[1:]
    if "--from-sources" in arguments:
        index = arguments.index("--from-sources")
        try:
            source_directory = Path(arguments[index + 1])
        except IndexError:
            raise SystemExit("Usage: python scripts/replay_check.py --from-sources SOURCE_DIRECTORY [--case CASE]")
        case = "aemo"
        if "--case" in arguments:
            case = arguments[arguments.index("--case") + 1]
        if case not in ("aemo", "ausgrid"):
            raise SystemExit("--case must be aemo or ausgrid")
        print(f"Running the {case} pipeline twice from {source_directory}\n")
        print("run 1 ...")
        first = run_once(source_directory, case)
        print("run 2 ...\n")
        second = run_once(source_directory, case)
        raise SystemExit(report(first, second, ("run 1", "run 2")))

    if len(arguments) != 2:
        raise SystemExit(
            "Usage:\n"
            "  python scripts/replay_check.py FIRST_RECORD.json SECOND_RECORD.json\n"
            "  python scripts/replay_check.py --from-sources SOURCE_DIRECTORY [--case aemo|ausgrid]"
        )
    paths = [Path(argument) for argument in arguments]
    records = []
    for path in paths:
        if not path.exists():
            raise SystemExit(f"No such record file: {path}")
        records.append(json.loads(path.read_text(encoding="utf-8")))
    raise SystemExit(report(records[0], records[1], (paths[0].name, paths[1].name)))
