# ai-powered-energy-forecasting
Local prototype for separately forecasting electricity load and photovoltaic energy with reference methods, XGBoost and a small LSTM.

Two provisional cases use real files: AEMO `NSW1` operational demand (12 months, MW) and Ausgrid customer `1` gross PV generation (`GG`; two quality-flagged years, kWh per original half-hour source slot). Both are one-step historical **validation** backtests. Neither the operational forecast horizon nor Ausgrid daylight-saving conversion has been approved; the app is not a live prediction service. See [data-source evidence](docs/data_sources.md), [XGBoost experiment](docs/first_ml_experiment.md), [LSTM experiment](docs/lstm_experiment.md) and [MVP status](docs/mvp_status.md).

Run the 12 archive, source-value, input-contract, chronology and feature tests with:

```text
python -m unittest discover -s tests -v
```

Create a Python 3.12+ virtual environment, then install `requirements.txt`, `requirements-lstm.txt` and `requirements-app.txt`. On Windows, XGBoost's native library requires the Microsoft Visual C++ runtime; see the [official install instructions](https://xgboost.readthedocs.io/en/stable/install.html). Do not copy a local runtime DLL into Git.

Run the local guided dashboard from this repository:

```text
python -m streamlit run streamlit_app.py
```

Open the local address printed by Streamlit, upload the **12 documented AEMO monthly ZIPs** (August 2025–July 2026) and the Ausgrid solar-home ZIP, then choose a case and train each model. The app validates source coverage, compares MAE/RMSE and positive-actual MAPE, charts all downloadable validation forecasts, and shows XGBoost's native SHAP feature contributions. Training the CPU LSTM can take several minutes. Its result is a development estimate selected on validation, not held-out final accuracy.

Do not commit raw downloaded archives. Keep them outside this repository or under the ignored `data/` directory. Data source terms and licences must be reviewed before anyone redistributes the files. The provisional implementation should be reviewed by the team before merging it to `main` or describing it as a client-approved platform.

For data/integration handoff, see [the copy-specific archive manifest and quality flags](docs/data_manifest.md) and [Rui's role handoff with facilitator questions and proposed backlog items](docs/rui_data_handoff.md). They distinguish local evidence from team/client decisions still needed.
