"""Accessible local validation dashboard for the two provisional energy cases."""

from __future__ import annotations

import io
import json
import sys
from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from energy_forecasting.baseline import evaluate_persistence, evaluate_seasonal_naive
from energy_forecasting.custom_data import (
    build_synthetic_demo_csv,
    infer_daily_seasonal_period,
    inspect_custom_csv,
    prepare_custom_csv,
)
from energy_forecasting.dashboard import forecast_plot_timestamp
from energy_forecasting.demo_data import prepare_uploaded_sources
from energy_forecasting.experiment_record import build_experiment_record
from energy_forecasting.lstm_model import fit_and_evaluate_test_lstm, fit_and_validate_lstm
from energy_forecasting.upload_validation import UploadRejected, validate_documented_sources
from energy_forecasting.xgb_model import fit_and_evaluate_test, fit_and_validate


CASE_LABELS = {
    "aemo": "AEMO NSW1 operational demand",
    "ausgrid": "Ausgrid customer 1 gross PV",
}
UNITS = {"aemo": "MW", "ausgrid": "kWh / source half-hour slot"}
MODEL_ORDER = ("XGBoost", "LSTM")


def apply_interface_theme() -> None:
    """Apply a restrained product-style visual system without changing behaviour."""
    st.markdown(
        """
<style>
    :root {
        --energy-navy: #10243e;
        --energy-blue: #1d5f8f;
        --energy-teal: #0f8b8d;
        --energy-gold: #e2a72e;
        --energy-ink: #172033;
        --energy-muted: #5f6b7a;
        --energy-line: #dce4ec;
        --energy-surface: #f6f9fc;
    }

    .stApp { background: linear-gradient(180deg, #f7fafc 0, #ffffff 23rem); }
    .block-container { max-width: 1240px; padding-top: 1.5rem; padding-bottom: 4rem; }

    .energy-hero {
        position: relative;
        overflow: hidden;
        color: #ffffff;
        background: linear-gradient(125deg, var(--energy-navy) 0%, #153f63 58%, var(--energy-teal) 135%);
        border-radius: 22px;
        padding: 2.1rem 2.3rem 2rem;
        margin: 0 0 1.25rem;
        box-shadow: 0 18px 45px rgba(16, 36, 62, 0.16);
    }
    .energy-hero::after {
        content: "";
        position: absolute;
        width: 260px;
        height: 260px;
        border-radius: 50%;
        right: -85px;
        top: -120px;
        background: rgba(255,255,255,0.09);
    }
    .energy-eyebrow {
        display: inline-flex;
        align-items: center;
        gap: .45rem;
        font-size: .76rem;
        font-weight: 750;
        letter-spacing: .12em;
        text-transform: uppercase;
        color: #d6f3f2;
        margin-bottom: .75rem;
    }
    .energy-hero h1 {
        color: #ffffff;
        font-size: clamp(2rem, 4vw, 3.25rem);
        line-height: 1.06;
        letter-spacing: -.035em;
        max-width: 850px;
        margin: 0 0 .8rem;
    }
    .energy-hero p {
        color: #e6eef6;
        font-size: 1.03rem;
        line-height: 1.6;
        max-width: 830px;
        margin: 0;
    }
    .hero-badges { display: flex; flex-wrap: wrap; gap: .55rem; margin-top: 1.25rem; }
    .hero-badge {
        border: 1px solid rgba(255,255,255,.24);
        background: rgba(255,255,255,.09);
        color: #ffffff;
        border-radius: 999px;
        padding: .38rem .72rem;
        font-size: .79rem;
        font-weight: 650;
    }

    .section-label {
        color: var(--energy-blue);
        font-size: .76rem;
        font-weight: 800;
        letter-spacing: .1em;
        text-transform: uppercase;
        margin: 1.6rem 0 .25rem;
    }
    .section-title { color: var(--energy-ink); font-size: 1.45rem; font-weight: 750; margin: 0 0 .2rem; }
    .section-copy { color: var(--energy-muted); margin: 0 0 .85rem; }

    .workflow-shell {
        border: 1px solid var(--energy-line);
        background: #ffffff;
        border-radius: 16px;
        padding: 1rem 1.05rem .9rem;
        margin: 1rem 0 .8rem;
        box-shadow: 0 5px 18px rgba(16, 36, 62, .045);
    }
    .workflow-head { display: flex; justify-content: space-between; gap: 1rem; align-items: baseline; }
    .workflow-head strong { color: var(--energy-ink); }
    .workflow-head span { color: var(--energy-muted); font-size: .86rem; }
    .workflow-steps { display: grid; grid-template-columns: repeat(5, 1fr); gap: .55rem; margin-top: .85rem; }
    .workflow-step {
        border-top: 4px solid #d8e0e8;
        color: #788494;
        padding: .52rem .2rem 0;
        font-size: .78rem;
        font-weight: 650;
    }
    .workflow-step.done { border-color: var(--energy-teal); color: #17696b; }
    .workflow-step.active { border-color: var(--energy-gold); color: var(--energy-ink); }

    div[data-testid="stExpander"] {
        background: rgba(255,255,255,.92);
        border: 1px solid var(--energy-line);
        border-radius: 15px;
        box-shadow: 0 5px 18px rgba(16, 36, 62, .04);
        overflow: hidden;
    }
    div[data-testid="stFileUploader"] section {
        border: 1.5px dashed #9fb4c7;
        border-radius: 14px;
        background: var(--energy-surface);
    }
    div[data-testid="stMetric"] {
        background: #ffffff;
        border: 1px solid var(--energy-line);
        border-radius: 14px;
        padding: .9rem 1rem;
        box-shadow: 0 5px 16px rgba(16, 36, 62, .045);
    }
    div[data-testid="stMetricLabel"] { color: var(--energy-muted); }
    div[data-testid="stMetricValue"] { color: var(--energy-navy); }
    .stButton > button, .stDownloadButton > button {
        min-height: 2.8rem;
        border-radius: 10px;
        font-weight: 700;
    }
    .stButton > button[kind="primary"] {
        border: 0;
        background: linear-gradient(100deg, var(--energy-blue), var(--energy-teal));
        box-shadow: 0 7px 18px rgba(29, 95, 143, .18);
    }
    .stButton > button:disabled,
    .stButton > button[kind="primary"]:disabled {
        color: #637083 !important;
        background: #e8eef3 !important;
        border: 1px solid #d3dde6 !important;
        box-shadow: none !important;
        opacity: 1 !important;
    }
    div[data-testid="stSegmentedControl"] button {
        min-height: 2.75rem;
        font-weight: 700;
    }
    button[data-baseweb="tab"] { font-weight: 700; }
    div[data-testid="stDataFrame"] { border: 1px solid var(--energy-line); border-radius: 12px; overflow: hidden; }
    .energy-note {
        border-left: 4px solid var(--energy-gold);
        background: #fffaf0;
        border-radius: 0 12px 12px 0;
        color: #57481f;
        padding: .85rem 1rem;
        margin: .5rem 0 1rem;
        font-size: .9rem;
    }
    .energy-footer {
        border-top: 1px solid var(--energy-line);
        color: var(--energy-muted);
        font-size: .82rem;
        margin-top: 2.5rem;
        padding-top: 1rem;
        text-align: center;
    }

    @media (max-width: 760px) {
        .block-container { padding: .8rem .75rem 2.5rem; }
        .energy-hero { padding: 1.45rem 1.25rem; border-radius: 16px; }
        .energy-hero h1 { font-size: 2rem; }
        .workflow-steps {
            grid-template-columns: repeat(5, minmax(74px, 1fr));
            gap: .35rem;
            overflow-x: auto;
            padding-bottom: .25rem;
        }
        .workflow-step {
            border-top: 4px solid #d8e0e8;
            border-left: 0;
            padding: .42rem .1rem 0;
            white-space: nowrap;
        }
        .workflow-head { display: block; }
    }
</style>
        """,
        unsafe_allow_html=True,
    )


def show_product_header() -> None:
    st.markdown(
        """
<section class="energy-hero">
  <div class="energy-eyebrow">● ICT30016 · Energy forecasting workspace</div>
  <h1>Explainable energy forecasting, from data to decision.</h1>
  <p>Prepare an energy time series, compare transparent forecasting models and export reproducible evidence through one guided workflow.</p>
  <div class="hero-badges">
    <span class="hero-badge">AEMO demand</span>
    <span class="hero-badge">Ausgrid solar PV</span>
    <span class="hero-badge">Custom CSV</span>
    <span class="hero-badge">XGBoost + LSTM</span>
  </div>
</section>
        """,
        unsafe_allow_html=True,
    )


def show_section_heading(step: str, title: str, copy: str) -> None:
    st.markdown(
        f'<div class="section-label">{step}</div>'
        f'<div class="section-title">{title}</div>'
        f'<div class="section-copy">{copy}</div>',
        unsafe_allow_html=True,
    )


def case_label(case: str) -> str:
    if case in CASE_LABELS:
        return CASE_LABELS[case]
    return st.session_state.get("prepared", {}).get(case, {}).get("label", case)


def unit_for(case: str) -> str:
    if case in UNITS:
        return UNITS[case]
    return st.session_state.get("prepared", {}).get(case, {}).get("unit", "energy units")


def metric_decimals_for(case: str) -> int:
    return 2 if case == "aemo" else 4 if case == "ausgrid" else 3


def run_validation_model(case: str, model_name: str) -> None:
    """Run a model before the next page render to avoid duplicate widget rows."""
    if case in st.session_state.get("final_results", {}):
        st.session_state["model_error"] = {
            "case": case,
            "model": model_name,
            "message": "Validation settings are locked because the reserved test has been used.",
        }
        return
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


def run_final_evaluation(case: str) -> None:
    """Use validation-selected settings once on the reserved test partition."""

    parts = st.session_state["prepared"][case]["parts"]
    validation_results = st.session_state["results"][case]
    try:
        with st.spinner("Retraining on train plus validation and scoring the reserved test…"):
            final = {
                "XGBoost": fit_and_evaluate_test(
                    parts,
                    case,
                    int(validation_results["XGBoost"]["best_iteration"]) + 1,
                    include_preview=True,
                ),
                "LSTM": fit_and_evaluate_test_lstm(
                    parts,
                    case,
                    int(validation_results["LSTM"]["best_epoch"]),
                    include_preview=True,
                ),
            }
            st.session_state.setdefault("final_results", {})[case] = final
        st.session_state["final_notice"] = case
    except (RuntimeError, ValueError, OSError) as error:
        st.session_state["final_error"] = {"case": case, "message": str(error)}


def metric_table(case_data: dict, model_results: dict) -> pd.DataFrame:
    """Return full-precision validation metrics for evidence and downloads."""
    validation = case_data["parts"]["validation"]
    results = {
        "Seasonal naive": evaluate_seasonal_naive(validation),
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


def final_metric_table(case_data: dict, final_results: dict) -> pd.DataFrame:
    test = case_data["parts"]["test"]
    values = {
        "Seasonal naive": evaluate_seasonal_naive(test),
        "Previous-slot persistence": evaluate_persistence(test),
        **{name: result["test_metrics"] for name, result in final_results.items()},
    }
    return pd.DataFrame([
        {
            "Approach": name,
            "MAE": metrics["MAE"],
            "RMSE": metrics["RMSE"],
            "Positive-actual MAPE (%)": metrics["MAPE_positive_actual_percent"],
            "Test targets": metrics["example_count"],
            "Zeros excluded from MAPE": metrics["zero_actual_count_excluded_from_MAPE"],
        }
        for name, metrics in values.items()
    ])


def display_metric_table(case: str, table: pd.DataFrame) -> pd.DataFrame:
    """Keep the interface readable while retaining exact values in downloads."""
    displayed = table.copy()
    metric_decimals = metric_decimals_for(case)
    displayed["MAE"] = displayed["MAE"].round(metric_decimals)
    displayed["RMSE"] = displayed["RMSE"].round(metric_decimals)
    displayed["Positive-actual MAPE (%)"] = displayed[
        "Positive-actual MAPE (%)"
    ].round(2)
    return displayed


def workflow_status(upload_ready: bool, input_mode: str) -> tuple[int, str]:
    """Summarise the current five-step workflow without relying on colour."""
    prepared = (
        "prepared" in st.session_state
        and st.session_state.get("prepared_input_mode") == input_mode
    )
    active_results = []
    if prepared:
        active_case = st.session_state.get("active_case", "aemo")
        active_results = list(st.session_state.get("results", {}).get(active_case, {}))
    if active_results:
        return 5, "Results ready: compare, explain and download"
    if prepared:
        return 3, "Sources prepared: select a case and run a model"
    if upload_ready:
        return 2, "Files selected: validate and prepare the sources"
    return 1, "Upload a supported source dataset"


def show_workflow_progress(upload_ready: bool, input_mode: str) -> None:
    step, message = workflow_status(upload_ready, input_mode)
    labels = ("Upload", "Validate", "Select", "Model", "Review")
    rendered_steps = []
    for index, label in enumerate(labels, start=1):
        state_class = "done" if index < step else "active" if index == step else ""
        rendered_steps.append(
            f'<div class="workflow-step {state_class}">{index} · {label}</div>'
        )
    step_markup = "".join(rendered_steps)
    st.markdown(
        f"""
<div class="workflow-shell">
  <div class="workflow-head">
    <strong>Workflow progress</strong>
    <span>Step {step} of 5 · {message}</span>
  </div>
  <div class="workflow-steps">{step_markup}</div>
</div>
        """,
        unsafe_allow_html=True,
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
              x=alt.X("plot_timestamp:T", title="Forecast target"),
            y=alt.Y("Value:Q", title=unit_for(case), scale=alt.Scale(zero=False)),
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
    decimals = metric_decimals_for(case)
    st.subheader(model_name)
    st.write(
        f"Across {metrics['example_count']:,} validation targets, {model_name} "
        f"achieved MAE {metrics['MAE']:.{decimals}f} and "
        f"RMSE {metrics['RMSE']:.{decimals}f} {unit_for(case)}. "
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

    residuals = frame.copy()
    residuals["residual"] = residuals["actual"] - residuals["predicted"]
    residuals["absolute_error"] = residuals["residual"].abs()
    residuals["plot_timestamp"] = residuals["target_key"].map(
        lambda target_key: forecast_plot_timestamp(case, target_key)
    )
    st.markdown("##### Residual analysis")
    st.caption(
        "Residual = actual - predicted. Values distributed around zero indicate "
        "less systematic over- or under-prediction."
    )
    residual_chart = (
        alt.Chart(residuals.head(96))
        .mark_line(color="#7C3AED", strokeWidth=2)
        .encode(
            x=alt.X("plot_timestamp:T", title="Forecast target"),
            y=alt.Y("residual:Q", title=f"Residual ({unit_for(case)})"),
            tooltip=[
                alt.Tooltip("target_key:N", title="Target"),
                alt.Tooltip("residual:Q", format=".3f"),
            ],
        )
        .properties(height=220)
    )
    st.altair_chart(residual_chart, width="stretch", theme="streamlit")
    seasonal = (
        residuals.assign(hour=residuals["plot_timestamp"].dt.hour)
        .groupby("hour", as_index=False)["absolute_error"]
        .mean()
        .rename(columns={"absolute_error": "mean_absolute_error"})
    )
    with st.expander("View mean absolute error by hour"):
        st.bar_chart(seasonal, x="hour", y="mean_absolute_error", color="#2563EB")


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
    st.bar_chart(global_shap, y_label=unit_for(case), color="#2563EB")
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
    decimals = metric_decimals_for(case)
    first, second, third, fourth = st.columns(4)
    first.metric("Best model by MAE", best_row["Approach"])
    second.metric("MAE", f"{best_row['MAE']:.{decimals}f} {unit_for(case)}")
    third.metric("vs persistence", f"{improvement:.1f}% lower")
    fourth.metric("Validation targets", f"{int(best_row['Validation targets']):,}")
    st.caption(
        "These cards show the completed trained model with the lowest validation "
        "MAE, not the model that was run most recently."
    )


st.set_page_config(
    page_title="Explainable energy forecasting platform",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="collapsed",
)
apply_interface_theme()
show_product_header()
st.markdown(
    """
<div class="energy-note"><strong>Evaluation safeguard:</strong> Validation supports model development. The reserved test partition remains untouched until the team confirms the final model settings.</div>
    """,
    unsafe_allow_html=True,
)

with st.expander("Source files, forecasting rules and metric definitions"):
    st.markdown("""
- **Demand:** AEMO NSW1, 12 `ACTUAL_DAILY` monthly ZIPs from August 2025 to July 2026; MW at fixed UTC+10 half-hour timestamps.
- **PV:** Ausgrid customer 1 `GG`, July 2011–June 2013; kWh in each original half-hour source slot.
- **Custom CSV:** map one timestamp column, one non-negative energy target and optional numeric covariates known at prediction time.
- **Forecast origin:** the immediately preceding observed source slot.
- **Selection:** training, validation and reserved test partitions follow chronology. Validation scores are development estimates.
- **MAPE:** positive-actual MAPE excludes zero PV targets. MAE and RMSE include every valid target.
""")

show_section_heading(
    "Get started",
    "Choose a data source",
    "Start instantly, upload your own data or reproduce the documented research benchmark.",
)
input_mode = st.segmented_control(
    "Input method",
    ("Try demo data", "Upload your CSV", "Official benchmark · Advanced"),
    default="Try demo data",
    label_visibility="collapsed",
)
mode_descriptions = {
    "Try demo data": (
        "No upload required. Explore the complete workflow with a clearly labelled "
        "synthetic two-week household-energy time series."
    ),
    "Upload your CSV": (
        "Map your timestamp, energy target and optional known numeric covariates. "
        "The file is validated before model preparation."
    ),
    "Official benchmark · Advanced": (
        "Reproduce the documented AEMO demand and Ausgrid solar-PV cases from the "
        "original 12 monthly AEMO archives and one Ausgrid archive."
    ),
}
st.info(mode_descriptions[input_mode])

source_step_titles = {
    "Try demo data": "1. Prepare demonstration data",
    "Upload your CSV": "1. Upload and map your CSV",
    "Official benchmark · Advanced": "1. Upload official source archives",
}

with st.expander(
    source_step_titles[input_mode],
    expanded="prepared" not in st.session_state,
):
    aemo_files = None
    ausgrid_file = None
    custom_file = None
    custom_configuration = None
    demo_bytes = None
    if input_mode == "Try demo data":
        demo_bytes = build_synthetic_demo_csv()
        st.markdown(
            "**Synthetic demonstration dataset**  \n"
            "14 days · half-hourly observations · household energy target · "
            "temperature covariate · no personal or operational data"
        )
        preview = pd.read_csv(io.BytesIO(demo_bytes)).head(8)
        st.dataframe(preview, hide_index=True, width="stretch")
        upload_ready = True
    elif input_mode == "Official benchmark · Advanced":
        st.caption(
            "This advanced path is only for reproducing the two official reference "
            "cases. General users do not need these 13 source archives."
        )
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
        upload_ready = bool(aemo_files and ausgrid_file)
    else:
        template_column, guidance_column = st.columns([1, 2])
        template_column.download_button(
            "Download valid sample CSV",
            data=build_synthetic_demo_csv(),
            file_name="valid_energy_forecast_sample.csv",
            mime="text/csv",
            width="stretch",
        )
        guidance_column.markdown(
            "**Minimum structure:** one timestamp column, one non-negative energy "
            "target and enough regular observations for chronological splitting."
        )
        custom_file = st.file_uploader(
            "Upload an approved-schema energy CSV",
            type="csv",
            help="Maximum 20 MB, UTF-8, one header row and chronological observations.",
        )
        st.caption("The application validates a maximum custom CSV size of 20 MB.")
        upload_ready = custom_file is not None
        if custom_file is not None:
            try:
                custom_bytes = custom_file.getvalue()
                inspection = inspect_custom_csv(custom_file.name, custom_bytes)
                st.caption(f"{inspection.row_count:,} rows · {inspection.byte_size / 1024:.1f} KB")
                timestamp_column = st.selectbox("Timestamp column", inspection.columns)
                target_options = [column for column in inspection.columns if column != timestamp_column]
                target_column = st.selectbox("Target column", target_options)
                covariate_options = [
                    column for column in inspection.columns
                    if column not in {timestamp_column, target_column}
                ]
                covariates = st.multiselect(
                    "Optional numeric covariates known at prediction time",
                    covariate_options,
                    help="Do not select measurements that would only be known after the forecast is made.",
                )
                inferred_period = infer_daily_seasonal_period(
                    custom_file.name, custom_bytes, timestamp_column
                )
                seasonal_period = st.number_input(
                    "Seasonal period in rows",
                    min_value=2,
                    value=inferred_period or 48,
                    step=1,
                    help="For half-hourly data, a daily season contains 48 rows.",
                )
                unit = st.text_input("Target unit", value="kWh")
                custom_configuration = {
                    "timestamp_column": timestamp_column,
                    "target_column": target_column,
                    "covariate_columns": covariates,
                    "seasonal_period": int(seasonal_period),
                    "unit": unit,
                }
                st.dataframe(
                    pd.read_csv(custom_file).head(10), hide_index=True, width="stretch"
                )
            except (ValueError, pd.errors.ParserError) as error:
                upload_ready = False
                st.error(f"CSV inspection failed: {error}")

show_workflow_progress(upload_ready, input_mode)

button_disabled = not upload_ready or (
    input_mode == "Upload your CSV" and custom_configuration is None
)
if input_mode == "Try demo data":
    action_label = "Prepare demo dataset"
elif input_mode == "Upload your CSV":
    action_label = "Validate and prepare CSV" if not button_disabled else "Upload a CSV to continue"
else:
    action_label = (
        "Validate and prepare official sources"
        if not button_disabled else "Select all 13 official archives to continue"
    )

if st.button(
    action_label,
    type="primary",
    disabled=button_disabled,
):
    try:
        with st.spinner("Validating schema, chronology and data quality…"):
            if input_mode == "Official benchmark · Advanced":
                evidence = validate_documented_sources(aemo_files, ausgrid_file)
                prepared = prepare_uploaded_sources(aemo_files, ausgrid_file)
            elif input_mode == "Try demo data":
                demo_case, evidence = prepare_custom_csv(
                    "synthetic_energy_demo.csv",
                    demo_bytes,
                    timestamp_column="timestamp",
                    target_column="energy_kwh",
                    covariate_columns=["temperature_c"],
                    seasonal_period=48,
                    unit="kWh / half-hour",
                )
                demo_case["label"] = "Synthetic household energy demo"
                demo_case["profile"]["source_type"] = "synthetic demonstration data"
                evidence["source_type"] = "synthetic demonstration data"
                prepared = {"demo": demo_case}
            else:
                custom_case, evidence = prepare_custom_csv(
                    custom_file.name,
                    custom_file.getvalue(),
                    **custom_configuration,
                )
                prepared = {"custom": custom_case}
            st.session_state["upload_evidence"] = evidence
            st.session_state["prepared"] = prepared
            st.session_state["prepared_input_mode"] = input_mode
            st.session_state["results"] = {key: {} for key in prepared}
            st.session_state["final_results"] = {}
        st.success("Source data validated and forecasting cases are ready.")
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

if (
    "prepared" in st.session_state
    and st.session_state.get("prepared_input_mode") == input_mode
):
    st.divider()
    show_section_heading(
        "Step 2",
        "Select a forecasting case",
        "Review the chronological split before training or comparing models.",
    )
    if st.session_state.get("active_case") not in st.session_state["prepared"]:
        st.session_state["active_case"] = next(iter(st.session_state["prepared"]))
    case = st.selectbox(
        "Dataset",
        list(st.session_state["prepared"]),
        format_func=case_label,
        key="active_case",
    )
    case_data = st.session_state["prepared"][case]
    parts = case_data["parts"]
    st.write(
        f"**Unit:** {unit_for(case)}  ·  **Train:** {len(parts['train']):,}  ·  "
        f"**Validation:** {len(parts['validation']):,}  ·  "
        f"**Reserved test:** {len(parts['test']):,} targets"
    )
    with st.expander("View source profile"):
        st.json(case_data["profile"])

    results = st.session_state["results"][case]
    final_locked = case in st.session_state.get("final_results", {})
    show_section_heading(
        "Step 3",
        "Run validation models",
        "Train XGBoost and LSTM on the same evidence and compare them with transparent baselines.",
    )
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
        disabled=final_locked,
    )
    lstm_column.button(
        lstm_label,
        key=f"lstm_{case}",
        width="stretch",
        on_click=run_validation_model,
        args=(case, "LSTM"),
        disabled=final_locked,
    )
    completed = [name for name in MODEL_ORDER if name in results]
    if completed:
        st.caption("Completed models: " + ", ".join(completed))
    else:
        st.caption("No trained model yet. Baseline results are already available.")
    if final_locked:
        st.caption("Model settings are locked because the reserved test has been evaluated.")

    comparison = metric_table(case_data, results)
    completed_final = st.session_state.get("final_results", {}).get(case, {})
    summary = {
        "case": case_label(case),
        "unit": unit_for(case),
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
        "test_status": (
            "evaluated once after validation settings were confirmed"
            if completed_final else "reserved and not yet evaluated"
        ),
        "test_comparison": (
            final_metric_table(case_data, completed_final).to_dict(orient="records")
            if completed_final else []
        ),
    }
    experiment_record = build_experiment_record(
        case=case,
        upload_evidence=st.session_state.get("upload_evidence", {}),
        source_profile=case_data["profile"],
        split_boundaries=case_data["boundaries"],
        part_counts={name: len(rows) for name, rows in parts.items()},
        model_results=results,
        final_results=completed_final,
    )

    show_section_heading(
        "Steps 4–5",
        "Review, explain and export",
        "Compare model quality, inspect forecast behaviour and download reproducible evidence.",
    )
    overview_tab, forecast_tab, explain_tab, final_tab, downloads_tab = st.tabs([
        "Overview", "Forecasts", "Explainability (XGBoost)", "Final test", "Downloads"
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
    with final_tab:
        final_results = st.session_state.get("final_results", {}).get(case)
        final_notice = st.session_state.pop("final_notice", None)
        if final_notice == case:
            st.success("Final test evaluation completed and locked for this session.")
        final_error = st.session_state.pop("final_error", None)
        if final_error and final_error["case"] == case:
            st.error(f"Final evaluation failed: {final_error['message']}")
        if final_results:
            st.warning(
                "The reserved test has now been used. Do not tune models or choose "
                "settings from these results."
            )
            test_comparison = final_metric_table(case_data, final_results)
            st.dataframe(
                display_metric_table(case, test_comparison),
                hide_index=True,
                width="stretch",
            )
            for name in MODEL_ORDER:
                frame = pd.DataFrame(final_results[name]["test_prediction_rows"])
                st.altair_chart(
                    forecast_chart(case, frame, f"{name} final test"),
                    width="stretch",
                    theme="streamlit",
                )
        else:
            st.write(
                "Run both models on validation first. When the team is ready to "
                "finalise the comparison, confirm below and evaluate the reserved test once."
            )
            confirmed = st.checkbox(
                "I confirm model settings are final and the reserved test may now be used",
                key=f"confirm_final_{case}",
            )
            st.button(
                "Run final test once",
                key=f"final_test_{case}",
                type="primary",
                disabled=not confirmed or not all(name in results for name in MODEL_ORDER),
                on_click=run_final_evaluation,
                args=(case,),
            )
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
        final_results = st.session_state.get("final_results", {}).get(case, {})
        for name in MODEL_ORDER:
            if name in final_results:
                frame = pd.DataFrame(final_results[name]["test_prediction_rows"])
                st.download_button(
                    f"Download {name} final-test forecasts (CSV)",
                    data=frame.to_csv(index=False).encode("utf-8"),
                    file_name=f"{case}_{name.lower()}_test_forecasts.csv",
                    mime="text/csv",
                    key=f"download_test_{case}_{name}",
                    width="stretch",
                )
        st.download_button(
            "Download full validation comparison (CSV)",
            data=comparison.to_csv(index=False).encode("utf-8"),
            file_name=f"{case}_validation_comparison.csv",
            mime="text/csv",
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

st.markdown(
    """
<div class="energy-footer">ICT30016 Innovation Project · Explainable energy forecasting prototype · Results support analysis and are not operational dispatch advice.</div>
    """,
    unsafe_allow_html=True,
)
