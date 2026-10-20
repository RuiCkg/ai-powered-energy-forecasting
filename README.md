# ai-powered-energy-forecasting
Explainable prototype for forecasting electricity load or photovoltaic energy with a seasonal benchmark, XGBoost and a small LSTM.

The guided workflow accepts the two documented real-data cases or an approved-schema UTF-8 CSV. A custom CSV user maps a timestamp, a non-negative energy target and optional numeric covariates that are known at prediction time. The adapter checks chronology, duplicates, missing values and regular frequency before creating causal lag and calendar features. Every case uses one-step historical forecasting and chronological train, validation and reserved-test partitions. The app is not a live prediction service. See [requirements traceability](docs/client_requirements_traceability.md), [model cards](docs/model_cards.md), [data-source evidence](docs/data_sources.md), [XGBoost experiment](docs/first_ml_experiment.md) and [LSTM experiment](docs/lstm_experiment.md).

Run the full archive, source-value, input-contract, chronology, model, explainability, security and reproducibility test suite with:

```text
python -m unittest discover -s tests -v
```

Create a Python 3.12+ virtual environment, then install `requirements.txt`. The two smaller requirements files remain for component-specific local setup. On Windows, XGBoost's native library requires the Microsoft Visual C++ runtime; see the [official install instructions](https://xgboost.readthedocs.io/en/stable/install.html). Do not copy a local runtime DLL into Git.

Run the local guided dashboard from this repository:

```text
python -m streamlit run streamlit_app.py
```

Open the address printed by Streamlit and follow the three screens: **Data**, **Models** and **Results**. Start with the labelled synthetic demo, upload a regular UTF-8 CSV and map its fields, or use the advanced benchmark path with the **12 documented AEMO monthly ZIPs** (August 2025-July 2026) and the Ausgrid solar-home ZIP. Run both models, compare MAE/RMSE/positive-actual MAPE, review XGBoost SHAP and export the evidence. Demo scores are illustrative rather than evidence of expected real-world performance. Training the CPU LSTM and the first cloud start can take longer than subsequent interactions. Use the reserved-test action only after model settings are final.

## Streamlit Community Cloud deployment

Current deployment: [ICT30016 AI Energy Forecasting](https://ict30016-ai-energy-forecasting.streamlit.app/)

1. Push the repository to GitHub without raw datasets.
2. In Streamlit Community Cloud, create an app from the repository and select `streamlit_app.py` as the entry point.
3. In **Advanced settings**, select Python 3.12, then deploy with the root `requirements.txt` and `.streamlit/config.toml`.
4. After deployment, run a custom CSV smoke test and one documented-source smoke test. Confirm XGBoost, LSTM, SHAP and all downloads before sharing the URL.

Do not commit raw downloaded archives. Keep them outside this repository or under the ignored `data/` directory. Data source terms and licences must be reviewed before anyone redistributes the files. The provisional implementation should be reviewed by the team before merging it to `main` or describing it as a client-approved platform.

For data/integration handoff, see [the copy-specific archive manifest and quality flags](docs/data_manifest.md) and [Rui's role handoff with facilitator questions and proposed backlog items](docs/rui_data_handoff.md). They distinguish local evidence from team/client decisions still needed.
