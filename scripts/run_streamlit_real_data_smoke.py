"""Exercise the Streamlit workflow with the two documented real datasets.

Run:
    python scripts/run_streamlit_real_data_smoke.py AEMO_DIRECTORY AUSGRID_ARCHIVE

The raw archives remain outside Git. The script uses Streamlit's official
AppTest interface to perform the same widget actions as the local browser app.
"""

from __future__ import annotations

import json
from pathlib import Path
import sys

from streamlit.testing.v1 import AppTest


def uploaded_tuple(path: Path) -> tuple[str, bytes, str]:
    return path.name, path.read_bytes(), "application/zip"


def fail_on_app_error(app: AppTest, stage: str) -> None:
    if app.error:
        messages = [str(item.value) for item in app.error]
        raise RuntimeError(f"{stage} displayed Streamlit errors: {messages}")
    if app.exception:
        messages = [str(item.value) for item in app.exception]
        raise RuntimeError(f"{stage} raised Streamlit exceptions: {messages}")


def model_summary(result: dict) -> dict:
    return {
        "model": result["model"],
        "validation_metrics": result["validation_metrics"],
        "selection": {
            key: result[key]
            for key in ("best_iteration", "best_epoch", "epochs_run")
            if key in result
        },
    }


def main() -> None:
    if len(sys.argv) != 3:
        raise SystemExit(
            "Usage: python scripts/run_streamlit_real_data_smoke.py "
            "AEMO_DIRECTORY AUSGRID_ARCHIVE"
        )

    repository = Path(__file__).resolve().parents[1]
    aemo_directory = Path(sys.argv[1])
    ausgrid_archive = Path(sys.argv[2])
    aemo_paths = sorted(
        aemo_directory.glob("PUBLIC_ACTUAL_OPERATIONAL_DEMAND_DAILY_????????.zip")
    )
    if len(aemo_paths) != 12:
        raise ValueError(f"Expected 12 AEMO archives, found {len(aemo_paths)}")

    app = AppTest.from_file(repository / "streamlit_app.py", default_timeout=900)
    app.run()
    fail_on_app_error(app, "initial render")

    app.file_uploader[0].set_value([uploaded_tuple(path) for path in aemo_paths])
    app.file_uploader[1].set_value(uploaded_tuple(ausgrid_archive))
    app.run()
    fail_on_app_error(app, "upload selection")

    app.button[0].click().run(timeout=900)
    fail_on_app_error(app, "source preparation")

    evidence = {
        "workflow": "Streamlit real-data widget smoke test",
        "sources_prepared": True,
        "cases": {},
    }

    for case in ("aemo", "ausgrid"):
        app.selectbox[0].set_value(case).run()
        fail_on_app_error(app, f"{case} selection")

        app.button(key=f"xgb_{case}").click().run(timeout=900)
        fail_on_app_error(app, f"{case} XGBoost")

        app.button(key=f"lstm_{case}").click().run(timeout=900)
        fail_on_app_error(app, f"{case} LSTM")

        results = app.session_state["results"][case]
        if set(results) != {"XGBoost", "LSTM"}:
            raise RuntimeError(f"{case} did not retain both model results: {sorted(results)}")

        download_labels = [button.label for button in app.download_button]
        required_downloads = (
            "Download XGBoost validation forecasts (CSV)",
            "Download LSTM validation forecasts (CSV)",
            "Download validation summary (JSON)",
            "Download reproducibility record (JSON)",
        )
        missing = [label for label in required_downloads if label not in download_labels]
        if missing:
            raise RuntimeError(f"{case} is missing downloads: {missing}")

        evidence["cases"][case] = {
            "xgboost": model_summary(results["XGBoost"]),
            "lstm": model_summary(results["LSTM"]),
            "comparison_table_present": bool(app.dataframe),
            "required_downloads_present": True,
        }

    print(json.dumps(evidence, indent=2))


if __name__ == "__main__":
    main()
