# Forecasting Model Cards

## Seasonal naive

**Purpose.** Provide a transparent benchmark using the target from one configured seasonal period earlier.

**Appropriate use.** Regular energy series with a meaningful repeated cycle, such as 48 half-hour rows per day.

**Limitations.** It cannot react to unusual conditions or use covariates. Its role is comparison, not deployment selection by itself.

## XGBoost

**Purpose.** Produce a one-step non-negative forecast from the previous value, seasonal lag, calendar fields and optional known-at-forecast-time numeric covariates.

**Selection.** The number of trees is selected with validation early stopping. Final evaluation retrains on train plus validation using that tree count and then scores the reserved test once.

**Explanation.** Native SHAP contributions provide global mean absolute associations and one local additive explanation. They explain the fitted model and do not prove causation.

**Limitations.** Results depend on data coverage, the chosen seasonal period and covariate availability. The model is not validated for live dispatch or operational control.

## LSTM

**Purpose.** Compare a small sequence model with XGBoost using the same one-step targets, partitions and metrics.

**Selection.** Training-only scaling prevents leakage. The best epoch is selected on validation MAE. Final evaluation retrains on train plus validation for that epoch count before scoring the reserved test.

**Limitations.** CPU training can take several minutes, performance is configuration-dependent and no deep-learning attribution method is included. Validation performance is not final test performance.

## Shared trust conditions

- Uploaded data must be public or approved for processing.
- Timestamps must be unique, chronological and regularly spaced.
- Selected covariates must be available when the forecast would be made.
- Near-zero energy targets make ordinary MAPE unstable, so the interface reports positive-actual MAPE and the excluded-zero count.
- A lower error alone does not establish operational value; users must consider the dataset, horizon, residuals and known limitations.
