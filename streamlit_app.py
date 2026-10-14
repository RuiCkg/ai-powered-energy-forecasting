"""Accessible local validation dashboard for the two provisional energy cases."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from energy_forecasting.baseline import evaluate_persistence, evaluate_seasonal_naive
from energy_forecasting.dashboard import forecast_plot_timestamp
from energy_forecasting.demo_data import prepare_uploaded_sources
from energy_forecasting.experiment_record import build_experiment_record
from energy_forecasting.lstm_model import fit_and_validate_lstm
from energy_forecasting.upload_validation import UploadRejected, validate_documented_sources
from energy_forecasting.xgb_model import fit_and_validate


CASE_LABELS = {
    "aemo": "AEMO NSW1 operational demand",
    "ausgrid": "Ausgrid customer 1 gross PV",
}
UNITS = {"aemo": "MW", "ausgrid": "kWh / source half-hour slot"}
MODEL_ORDER = ("XGBoost", "LSTM")


def run_validation_model(case: str, model_name: str) -> None:
    """Run a model before the next page render to avoid duplicate widget rows."""
    case_data = st.session_state["prepared"][case]
    parts = case_data["parts"]
    messages = {
        "XGBoost": "Training XGBoost and calculating SHAP contributions…",
        "LSTM": "Training the CPU LSTM; this may take several minutes…",
    }
    trainers = {
        "XGBoost": fit_and_validate,
        "LSTM": fit_and_validate_lstm,
    }
    try:
        with st.spinner(messages[model_name]):
            st.session_state["results"][case][model_name] = trainers[model_name](
                parts,
                case,
                include_preview=True,
            )
        st.session_state["completion_notice"] = {
            "case": case,
            "model": model_name,
        }
        st.session_state.pop("model_error", None)
    except (RuntimeError, ValueError, OSError) as error:
        st.session_state["model_error"] = {
            "case": case,
            "model": model_name,
            "message": str(error),
        }


def metric_table(case_data: dict, model_results: dict) -> pd.DataFrame:
    """Return full-precision validation metrics for evidence and downloads."""
    validation = case_data["parts"]["validation"]
    results = {
        "Previous-day seasonal naive": evaluate_seasonal_naive(validation),
        "Previous-slot persistence": evaluate_persistence(validation),
    }
    results.update({
        name: value["validation_metrics"]
        for name, value in model_results.items()
    })
    return pd.DataFrame([
        {
            "Approach": name,
            "MAE": values["MAE"],
            "RMSE": values["RMSE"],
            "Positive-actual MAPE (%)": values["MAPE_positive_actual_percent"],
            "Validation targets": values["example_count"],
            "Zeros excluded from MAPE": values["zero_actual_count_excluded_from_MAPE"],
        }
        for name, values in results.items()
    ])


def display_metric_table(case: str, table: pd.DataFrame) -> pd.DataFrame:
    """Keep the interface readable while retaining exact values in downloads."""
    displayed = table.copy()
    metric_decimals = 2 if case == "aemo" else 4
    displayed["MAE"] = displayed["MAE"].round(metric_decimals)
    displayed["RMSE"] = displayed["RMSE"].round(metric_decimals)
    displayed["Positive-actual MAPE (%)"] = displayed[
        "Positive-actual MAPE (%)"
    ].round(2)
    return displayed


def workflow_status(aemo_files, ausgrid_file) -> tuple[int, str]:
    """Summarise the current five-step workflow without relying on colour."""
    uploaded = bool(aemo_files and ausgrid_file)
    prepared = "prepared" in st.session_state
    active_results = []
    if prepared:
        active_case = st.session_state.get("active_case", "aemo")
        active_results = list(st.session_state.get("results", {}).get(active_case, {}))
    if active_results:
        return 5, "Results ready: compare, explain and download"
    if prepared:
        return 3, "Sources prepared: select a case and run a model"
    if uploaded:
        return 2, "Files selected: validate and prepare the sources"
    return 1, "Upload the two documented source sets"


def show_workflow_progress(aemo_files, ausgrid_file) -> None:
    step, message = workflow_status(aemo_files, ausgrid_file)
    st.markdown("**Workflow**")
    st.progress(step / 5, text=f"Step {step} of 5 — {message}")
    st.caption(
        "1 Upload  →  2 Validate  →  3 Select dataset  →  "
        "4 Run models  →  5 Compare and export"
    )


def forecast_chart(case: str, frame: pd.DataFrame, model_name: str) -> alt.Chart:
    """Use both colour and line style so the series are not colour-dependent."""
    chart_frame = frame.head(96).copy()
    chart_frame["plot_timestamp"] = chart_frame["target_key"].map(
        lambda target_key: forecast_plot_timestamp(case, target_key)
    )
    chart_rows = chart_frame.melt(
        id_vars=["target_key", "plot_timestamp"],
        value_vars=["actual", "predicted"],
        var_name="Series",
        value_name="Value",
    )
    return (
        alt.Chart(chart_rows)
        .mark_line(strokeWidth=2.5)
        .encode(
              x=alt.X("plot_timestamp:T", title="Validation target"),
            y=alt.Y("Value:Q", title=UNITS[case], scale=alt.Scale(zero=False)),
            color=alt.Color(
                "Series:N",
                scale=alt.Scale(
                    domain=["actual", "predicted"],
                    range=["#2563EB", "#D97706"],
                ),
                legend=alt.Legend(title=None),
            ),
            strokeDash=alt.StrokeDash(
                "Series:N",
                scale=alt.Scale(
                    domain=["actual", "predicted"],
                    range=[[1, 0], [7, 4]],
                ),
                legend=alt.Legend(title=None),
            ),
            tooltip=[
                  alt.Tooltip("target_key:N", title="Target"),
                alt.Tooltip("Series:N"),
                alt.Tooltip("Value:Q", format=".3f"),
            ],
        )
        .properties(
            title=f"{model_name}: actual (solid) and predicted (dashed)",
            height=360,
        )
        .interactive()
    )


def show_forecast_result(case: str, model_name: str, result: dict) -> None:
    metrics = result["validation_metrics"]
    frame = pd.DataFrame(result["validation_prediction_rows"])
    decimals = 2 if case == "aemo" else 4
    st.subheader(model_name)
    st.write(
        f"Across {metrics['example_count']:,} validation targets, {model_name} "
        f"achieved MAE {metrics['MAE']:.{decimals}f} and "
        f"RMSE {metrics['RMSE']:.{decimals}f} {UNITS[case]}. "
          "The chart shows the first 96 validation targets for readability."
    )
    st.altair_chart(
        forecast_chart(case, frame, model_name),
        width="stretch",
        theme="streamlit",
    )
    with st.expander("View the first 20 forecast rows"):
        displayed = frame.head(20).copy()
        displayed["actual"] = displayed["actual"].round(decimals)
        displayed["predicted"] = displayed["predicted"].round(decimals)
        st.dataframe(displayed, hide_index=True, width="stretch")


def show_explainability(case: str, result: dict) -> None:
    global_shap = pd.Series(
        result["validation_mean_absolute_SHAP_by_feature"],
        name="Mean absolute SHAP",
    ).sort_values(ascending=False)
    feature_summary = ", ".join(
        f"{name} ({value:.3f})" for name, value in global_shap.head(3).items()
    )
    st.subheader("XGBoost feature contributions (SHAP)")
    st.info(
        "This tab explains the XGBoost model only. LSTM metrics and forecasts "
        "remain available in Overview and Forecasts; a separate deep-learning "
        "attribution method is outside the current prototype scope."
    )
    st.write(
        "The three strongest average model associations are "
        f"{feature_summary}. SHAP explains the fitted model; it does not prove "
        "that a feature causes the energy outcome."
    )
    st.bar_chart(global_shap, y_label=UNITS[case], color="#2563EB")
    local = result["validation_first_local_SHAP"]
    with st.expander("View one local SHAP calculation"):
        st.caption(
            "Raw prediction = bias + feature contributions. Nonnegative clipping "
            "is applied afterwards."
        )
        local_frame = pd.DataFrame([{
            "bias": local["bias"],
            **local["feature_contributions"],
            "raw prediction": local["raw_prediction_before_clipping"],
        }]).round(4)
        st.dataframe(local_frame, hide_index=True, width="stretch")


def model_summary_cards(case: str, comparison: pd.DataFrame, results: dict) -> None:
    if not results:
        st.info("Run XGBoost or LSTM to add a trained model to the comparison.")
        return
    model_rows = comparison[comparison["Approach"].isin(results)]
    best_row = model_rows.loc[model_rows["MAE"].idxmin()]
    persistence_mae = comparison.loc[
        comparison["Approach"] == "Previous-slot persistence", "MAE"
    ].iloc[0]
    improvement = 100 * (1 - best_row["MAE"] / persistence_mae)
    decimals = 2 if case == "aemo" else 4
    first, second, third, fourth = st.columns(4)
    first.metric("Best model by MAE", best_row["Approach"])
    second.metric("MAE", f"{best_row['MAE']:.{decimals}f} {UNITS[case]}")
    third.metric("vs persistence", f"{improvement:.1f}% lower")
    fourth.metric("Validation targets", f"{int(best_row['Validation targets']):,}")
    st.caption(
        "These cards show the completed trained model with the lowest validation "
        "MAE, not the model that was run most recently."
    )


st.set_page_config(page_title="Energy forecasting validation demo", layout="wide")
st.title("Energy forecasting validation demo")
st.write(
    "Compare interpretable machine-learning and sequence-model forecasts on "
    "two documented Australian energy datasets."
)
st.info(
    "This is a local **historical-validation prototype**, not a live prediction "
    "service. The reserved test period is not displayed or scored."
)

with st.expander("Source files, forecasting rules and metric definitions"):
    st.markdown("""
- **Demand:** AEMO NSW1, 12 `ACTUAL_DAILY` monthly ZIPs from August 2025 to July 2026; MW at fixed UTC+10 half-hour timestamps.
- **PV:** Ausgrid customer 1 `GG`, July 2011–June 2013; kWh in each original half-hour source slot.
- **Forecast origin:** the immediately preceding observed source slot.
- **Selection:** training, validation and reserved test partitions follow chronology. Validation scores are development estimates.
- **MAPE:** positive-actual MAPE excludes zero PV targets. MAE and RMSE include every valid target.
""")

with st.expander(
    "1. Upload source archives",
    expanded="prepared" not in st.session_state,
):
    aemo_files = st.file_uploader(
        "Upload 12 AEMO monthly ZIP archives",
        type="zip",
        accept_multiple_files=True,
        help="Select the 12 consecutive monthly ACTUAL_DAILY archives together.",
    )
    ausgrid_file = st.file_uploader(
        "Upload the Ausgrid solar-home ZIP archive",
        type="zip",
        help="Use the original documented archive; the data remain local.",
    )
    aemo_count = len(aemo_files) if aemo_files else 0
    left, right = st.columns(2)
    left.metric("AEMO archives selected", f"{aemo_count}/12")
    right.metric("Ausgrid archive selected", "1/1" if ausgrid_file else "0/1")
    if aemo_count and aemo_count != 12:
        st.warning(f"Select exactly 12 AEMO monthly archives; currently {aemo_count} selected.")

show_workflow_progress(aemo_files, ausgrid_file)

if st.button(
    "Validate files and prepare forecasting cases",
    type="primary",
    disabled=not (aemo_files and ausgrid_file),
):
    try:
        with st.spinner("Validating archives, chronology and data quality…"):
            evidence = validate_documented_sources(aemo_files, ausgrid_file)
            prepared = prepare_uploaded_sources(aemo_files, ausgrid_file)
            st.session_state["upload_evidence"] = evidence
            st.session_state["prepared"] = prepared
            st.session_state["results"] = {"aemo": {}, "ausgrid": {}}
        st.success("Sources validated and both forecasting cases are ready.")
    except UploadRejected as rejection:
        st.session_state.pop("prepared", None)
        st.session_state.pop("upload_evidence", None)
        st.session_state.pop("results", None)
        st.error(f"Upload rejected ({rejection.code}): {rejection.message}")
    except (ValueError, KeyError, OSError, TypeError) as error:
        st.session_state.pop("prepared", None)
        st.session_state.pop("upload_evidence", None)
        st.session_state.pop("results", None)
        st.error(f"Source validation failed: {error}")

if "prepared" in st.session_state:
    st.divider()
    st.subheader("2. Select a forecasting case")
    case = st.selectbox(
        "Dataset",
        list(CASE_LABELS),
        format_func=lambda key: CASE_LABELS[key],
        key="active_case",
    )
    case_data = st.session_state["prepared"][case]
    parts = case_data["parts"]
    st.write(
        f"**Unit:** {UNITS[case]}  ·  **Train:** {len(parts['train']):,}  ·  "
        f"**Validation:** {len(parts['validation']):,}  ·  "
        f"**Reserved test:** {len(parts['test']):,} targets"
    )
    with st.expander("View source profile"):
        st.json(case_data["profile"])

    results = st.session_state["results"][case]
    st.subheader("3. Run validation models")
    completion_notice = st.session_state.pop("completion_notice", None)
    if completion_notice and completion_notice["case"] == case:
        st.success(
            f"{completion_notice['model']} completed. Results are available below."
        )
    model_error = st.session_state.pop("model_error", None)
    if model_error and model_error["case"] == case:
        st.error(f"{model_error['model']} failed: {model_error['message']}")
    xgb_column, lstm_column = st.columns(2)
    xgb_label = "Re-run XGBoost" if "XGBoost" in results else "Run XGBoost"
    lstm_label = "Re-run LSTM" if "LSTM" in results else "Run LSTM"
    xgb_column.button(
        xgb_label,
        key=f"xgb_{case}",
        width="stretch",
        on_click=run_validation_model,
        args=(case, "XGBoost"),
    )
    lstm_column.button(
        lstm_label,
        key=f"lstm_{case}",
        width="stretch",
        on_click=run_validation_model,
        args=(case, "LSTM"),
    )
    completed = [name for name in MODEL_ORDER if name in results]
    if completed:
        st.caption("Completed models: " + ", ".join(completed))
    else:
        st.caption("No trained model yet. Baseline results are already available.")

    comparison = metric_table(case_data, results)
    summary = {
        "case": CASE_LABELS[case],
        "unit": UNITS[case],
        "source_profile": case_data["profile"],
        "split_counts": {name: len(rows) for name, rows in parts.items()},
        "validation_comparison": comparison.to_dict(orient="records"),
        "models": {
            name: {
                key: value
                for key, value in result.items()
                if key != "validation_prediction_rows"
            }
            for name, result in results.items()
        },
        "test_status": "held out, not evaluated in this local demo",
    }
    experiment_record = build_experiment_record(
        case=case,
        upload_evidence=st.session_state.get("upload_evidence", {}),
        source_profile=case_data["profile"],
        split_boundaries=case_data["boundaries"],
        part_counts={name: len(rows) for name, rows in parts.items()},
        model_results=results,
    )

    st.subheader("4. Review and export results")
    overview_tab, forecast_tab, explain_tab, downloads_tab = st.tabs([
        "Overview", "Forecasts", "Explainability (XGBoost)", "Downloads"
    ])
    with overview_tab:
        model_summary_cards(case, comparison, results)
        st.markdown("#### Validation comparison")
        st.dataframe(
            display_metric_table(case, comparison),
            hide_index=True,
            width="stretch",
        )
        st.caption(
            "Smaller MAE and RMSE are better. Positive-actual MAPE is secondary "
            "for PV because zero-generation targets are excluded."
        )
    with forecast_tab:
        if not results:
            st.info("Run a model to display historical validation forecasts.")
        for name in MODEL_ORDER:
            if name in results:
                show_forecast_result(case, name, results[name])
    with explain_tab:
        if "XGBoost" in results:
            show_explainability(case, results["XGBoost"])
        else:
            st.info("Run XGBoost to display SHAP explanations.")
    with downloads_tab:
        st.write(
            "Download full-precision evidence. The on-screen tables are rounded "
            "for readability; exported values are not rounded."
        )
        for name in MODEL_ORDER:
            if name in results:
                frame = pd.DataFrame(results[name]["validation_prediction_rows"])
                st.download_button(
                    f"Download {name} validation forecasts (CSV)",
                    data=frame.to_csv(index=False).encode("utf-8"),
                    file_name=f"{case}_{name.lower()}_validation_forecasts.csv",
                    mime="text/csv",
                    key=f"download_{case}_{name}",
                    width="stretch",
                )
        st.download_button(
            "Download validation summary (JSON)",
            json.dumps(summary, indent=2).encode("utf-8"),
            file_name=f"{case}_validation_summary.json",
            mime="application/json",
            width="stretch",
        )
        st.download_button(
            "Download reproducibility record (JSON)",
            json.dumps(experiment_record, indent=2).encode("utf-8"),
            file_name=f"{case}_experiment_record.json",
            mime="application/json",
            width="stretch",
        )
