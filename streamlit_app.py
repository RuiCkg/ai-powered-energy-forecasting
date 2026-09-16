"""Local, historical-validation demo for the two provisional energy cases."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from energy_forecasting.baseline import (
    evaluate_persistence,
    evaluate_seasonal_naive,
)
from energy_forecasting.demo_data import prepare_uploaded_sources
from energy_forecasting.lstm_model import fit_and_validate_lstm
from energy_forecasting.xgb_model import fit_and_validate


CASE_LABELS = {"aemo": "AEMO NSW1 operational demand", "ausgrid": "Ausgrid customer 1 gross PV"}
UNITS = {"aemo": "MW", "ausgrid": "kWh / source half-hour slot"}


def metric_table(case_data, model_results):
    validation = case_data["parts"]["validation"]
    results = {
        "Previous-day seasonal naive": evaluate_seasonal_naive(validation),
        "Previous-slot persistence": evaluate_persistence(validation),
    }
    results.update({name: value["validation_metrics"] for name, value in model_results.items()})
    return pd.DataFrame([
        {
            "Approach": name,
            "MAE": values["MAE"],
            "RMSE": values["RMSE"],
            "MAPE on positive actuals (%)": values["MAPE_positive_actual_percent"],
            "Validation targets": values["example_count"],
            "Zero actuals excluded from MAPE": values["zero_actual_count_excluded_from_MAPE"],
        }
        for name, values in results.items()
    ])


def show_model_result(case: str, model_name: str, result: dict):
    rows = result["validation_prediction_rows"]
    with st.expander(f"{model_name}: predictions and diagnostics", expanded=True):
        st.caption("Historical validation backtest; chart displays at most 192 consecutive forecast slots.")
        frame = pd.DataFrame(rows)
        sample = frame.head(192).set_index("target_key")[["actual", "predicted"]]
        st.line_chart(sample, x_label="Validation target", y_label=UNITS[case])
        st.dataframe(frame.head(20), hide_index=True)
        st.download_button(
            f"Download {model_name} validation forecasts (CSV)",
            data=frame.to_csv(index=False).encode("utf-8"),
            file_name=f"{case}_{model_name.lower()}_validation_forecasts.csv",
            mime="text/csv",
            key=f"download_{case}_{model_name}",
        )
        if model_name == "XGBoost":
            st.subheader("XGBoost feature contributions (SHAP)")
            global_shap = pd.Series(result["validation_mean_absolute_SHAP_by_feature"], name="Mean absolute SHAP")
            st.bar_chart(global_shap, y_label=UNITS[case])
            local = result["validation_first_local_SHAP"]
            st.caption("Local example: raw prediction = bias + feature contributions. These are associations, not causal effects. Nonnegative clipping is applied afterwards.")
            st.dataframe(pd.DataFrame([{"bias": local["bias"], **local["feature_contributions"],
                                       "raw prediction": local["raw_prediction_before_clipping"]}]), hide_index=True)


st.set_page_config(page_title="Energy forecasting validation demo", layout="wide")
st.title("Energy forecasting: local validation demo")
st.write("A provisional school-project prototype comparing two separately evaluated time series. Upload the documented source archives; no files are sent to GitHub by this app.")
st.info("This shows one-step **historical validation** forecasts, not a live grid/PV prediction service. The held-out test period is reserved and is not displayed or scored here.")

with st.expander("Source files and forecasting rules", expanded=True):
    st.markdown("""
- **Load:** AEMO NSW1, 12 `ACTUAL_DAILY` monthly ZIPs from August 2025 to July 2026; units MW and NEM fixed UTC+10 half-hour timestamps.
- **PV:** Ausgrid solar-home ZIP, customer 1 `GG`, July 2011–June 2013; kWh in each of 48 original source slots per day. Absolute daylight-saving timestamps are deliberately unresolved.
- **Forecast origin:** immediately preceding observed half-hour/source slot. These input values must actually be available at inference time; operational latency has not been confirmed.
- **Selection:** train/validation/test follow chronology. Tree count and LSTM checkpoint use validation, so validation scores are development estimates, not unbiased final performance.
- **MAPE:** ordinary MAPE is undefined at zero PV actuals. The displayed positive-actual MAPE excludes zeros and must be read with that count; MAE and RMSE include all targets.
""")

aemo_files = st.file_uploader("Upload 12 AEMO monthly ZIP archives", type="zip", accept_multiple_files=True)
ausgrid_file = st.file_uploader("Upload Ausgrid solar-home ZIP archive", type="zip")
if st.button("Read sources and prepare validation", type="primary", disabled=not (aemo_files and ausgrid_file)):
    try:
        with st.spinner("Reading source archives and checking chronology and quality..."):
            st.session_state["prepared"] = prepare_uploaded_sources(aemo_files, ausgrid_file)
            st.session_state["results"] = {"aemo": {}, "ausgrid": {}}
    except (ValueError, KeyError, OSError, TypeError) as error:
        st.session_state.pop("prepared", None)
        st.error(f"Source validation failed: {error}")

if "prepared" in st.session_state:
    case = st.selectbox("Case", list(CASE_LABELS), format_func=lambda key: CASE_LABELS[key])
    case_data = st.session_state["prepared"][case]
    parts = case_data["parts"]
    st.caption(f"Units: {UNITS[case]}. Train {len(parts['train']):,}; validation {len(parts['validation']):,}; test reserved {len(parts['test']):,} forecast targets.")
    with st.expander("Source profile"):
        st.json(case_data["profile"])

    results = st.session_state["results"][case]
    left, right = st.columns(2)
    if left.button("Train and validate XGBoost", key=f"xgb_{case}"):
        try:
            with st.spinner("Training XGBoost and calculating validation SHAP contributions..."):
                results["XGBoost"] = fit_and_validate(parts, case, include_preview=True)
        except (RuntimeError, ValueError, OSError) as error:
            st.error(f"XGBoost failed: {error}")
    if right.button("Train and validate LSTM (may take several minutes)", key=f"lstm_{case}"):
        try:
            with st.spinner("Training a small CPU LSTM; checkpoint selection uses validation only..."):
                results["LSTM"] = fit_and_validate_lstm(parts, case, include_preview=True)
        except (RuntimeError, ValueError, OSError) as error:
            st.error(f"LSTM failed: {error}")

    st.subheader("Validation comparison")
    st.dataframe(metric_table(case_data, results), hide_index=True)
    for name, result in results.items():
        show_model_result(case, name, result)
    summary = {"case": CASE_LABELS[case], "unit": UNITS[case], "source_profile": case_data["profile"],
               "split_counts": {name: len(rows) for name, rows in parts.items()},
               "validation_comparison": metric_table(case_data, results).to_dict(orient="records"),
               "models": {name: {key: value for key, value in result.items() if key != "validation_prediction_rows"}
                          for name, result in results.items()},
               "test_status": "held out, not evaluated in this local demo"}
    st.download_button("Download validation summary (JSON)", json.dumps(summary, indent=2).encode("utf-8"),
                       file_name=f"{case}_validation_summary.json", mime="application/json")
