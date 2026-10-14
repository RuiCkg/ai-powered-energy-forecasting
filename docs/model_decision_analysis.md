# Model decision analysis

**Owner:** Khang Huynh (AI/ML Model Lead)

**Reviewed:** 14 October 2026

## Decision

The prototype retains a small one-layer LSTM as the advanced sequence-model comparator and uses XGBoost as the current primary validation candidate. This decision is based on the implemented, reproducible comparison below—not on an unsupported claim that one architecture is universally superior.

GRU and TCN remain possible future alternatives. They were considered conceptually but were not implemented or empirically evaluated in this project, so no performance claim is made for them.

## Implemented comparison

Both models use the same chronological train/validation partitions and the same one-step target definition. Model selection uses validation only; the reserved test targets are not reported here.

| Case | Metric | XGBoost | LSTM | Better validation result |
| --- | --- | ---: | ---: | --- |
| AEMO NSW1 demand | MAE (MW) | **90.37** | 164.51 | XGBoost |
| AEMO NSW1 demand | RMSE (MW) | **122.69** | 202.09 | XGBoost |
| Ausgrid customer 1 GG | MAE (kWh/slot) | **0.0676** | 0.0688 | XGBoost |
| Ausgrid customer 1 GG | RMSE (kWh/slot) | 0.1406 | **0.1394** | LSTM |

Both models beat the seasonal-naive and persistence baselines on validation MAE. XGBoost is preferred for the demonstration because it has the lowest MAE in both cases, trains faster in the current CPU workflow, and provides native SHAP contributions. LSTM remains valuable evidence that the platform supports a genuinely sequential deep-learning approach; it also gives the lowest Ausgrid RMSE by a small margin.

## Limitations and future options

- The LSTM uses 48 historical values and six known calendar fields, not a 168-hour history.
- Only one bounded LSTM configuration was run; the AEMO score was still improving at epoch 20.
- GRU and TCN require implementation on the same splits before a fair empirical comparison is possible.
- Validation results do not establish performance on the reserved test periods, other regions/customers, longer horizons or live data.
- The PV MAPE excludes 2,029 zero-actual targets, so MAE and RMSE are the primary PV decision metrics.

Reproduction details are in [`lstm_experiment.md`](lstm_experiment.md), [`first_ml_experiment.md`](first_ml_experiment.md) and [`validation_results_2026-10-14.json`](validation_results_2026-10-14.json).
