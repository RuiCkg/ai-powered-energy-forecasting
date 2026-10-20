# Technical MVP status, 20 October 2026

## Implemented and verified

- Accepts either the documented AEMO and Ausgrid archives or an approved-schema UTF-8 custom CSV.
- Maps a custom timestamp, non-negative energy target and optional numeric covariates known at forecast time.
- Rejects oversize, malformed, missing, duplicate, unordered, irregular or non-numeric inputs with correction messages.
- Creates causal lag, seasonal, calendar and sequence features, followed by chronological train, validation and reserved-test partitions.
- Compares seasonal-naive and persistence references with XGBoost and a PyTorch LSTM using MAE, RMSE and positive-actual MAPE.
- Presents actual-versus-predicted charts, residual plots and hourly mean absolute error.
- Provides verified global and local native SHAP contributions for XGBoost.
- Exports validation and final-test forecasts, comparison data, a summary and a reproducibility record.
- Allows the reserved test to be run only after both model configurations have been selected on validation; final training uses train plus validation.
- Includes Streamlit Community Cloud dependency and theme configuration; Python 3.12 is selected during deployment.
- Passes 87 automated tests in the project virtual environment, including both final-evaluation paths.

## Deployment and acceptance status

- The repository is deployment-ready, but a public Streamlit URL has not yet been created and smoke-tested.
- Two independent human usability reviews and recorded client/facilitator acceptance remain pending.
- AEMO and Ausgrid licence and provenance restrictions still apply; raw archives remain outside Git.
- The fixed Ausgrid source-slot treatment does not claim an absolute daylight-saving timestamp policy.

## Explicit prototype boundaries

- One-step historical forecasting, not real-time market ingestion or autonomous grid control.
- One mapped target at a time; no multi-target or recursive multi-step forecasting.
- Regular time series only; arbitrary irregular schemas require a new adapter.
- SHAP for XGBoost only; LSTM attribution remains an extension.
- Internal reusable model services are used instead of Darts, and validation early stopping is used instead of a full Optuna study.

See the [client requirements traceability matrix](client_requirements_traceability.md) and [model cards](model_cards.md) before describing scope in the presentation.
