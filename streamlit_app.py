"""Accessible local validation dashboard for the two provisional energy cases."""

from __future__ import annotations

import io
import json
import sys
import time
from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st

ROOT_DIR = Path(__file__).resolve().parent
for candidate in (ROOT_DIR / "src", ROOT_DIR):
    candidate = candidate.resolve()
    if candidate.exists() and str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

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
