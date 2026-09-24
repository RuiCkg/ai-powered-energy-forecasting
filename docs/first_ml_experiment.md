# First XGBoost validation experiment

This is a reproducible **first technical experiment**, not a final forecasting platform or a claim of client approval. It uses the source cases and provisional chronological partitions documented in [the data-source inventory](data_sources.md). Team review is needed before treating the implementation as a shared, accepted increment.

## Forecast and features

- AEMO `NSW1`: predict the next actual half-hour demand value (MW) from a forecast origin one interval earlier, using fixed NEM UTC+10 timestamps.
- Ausgrid customer `1`, `GG`: predict the next **source half-hour slot within the same day** (kWh); this is not yet an absolute-time 30-minute forecast through daylight-saving transitions. Day-boundary examples are excluded.
- Each case has its **own** XGBoost model. The five features are previous actual value (`lag_1`), previous-day same-slot actual (`lag_48`), known target slot, weekday and month. No future target, weather observation or later file row is a feature.
- The latest-completed actual is assumed available at the forecast origin for this offline backtest. Its real-time availability and delivery latency must be checked before claiming a live forecasting service.
- Target-labelled examples follow the same provisional train/validation/test periods. Historical lag values may cross a partition boundary only when they precede the forecast origin. The test targets are not used to fit, select trees or report XGBoost performance at this stage.

## Fixed model setup

Python 3.13.5 with `numpy 2.5.3`, `scipy 1.18.1` and CPU-only `xgboost 3.4.1` was used locally. The repository pins these in `requirements.txt`. Each model uses squared-error regression, histogram trees, maximum depth 4, learning rate 0.05, subsample 0.8, seed 42 and two CPU threads. The maximum is 500 rounds, with MAE-based early stopping after 30 non-improving rounds on **validation only**. The best iterations were 140 for AEMO and 128 for Ausgrid. Predictions are clipped at zero as a predeclared physical non-negativity rule.

The [official XGBoost Python API](https://xgboost.readthedocs.io/en/stable/python/python_api.html) documents training with a validation set and early stopping. These parameters are a single fixed starting configuration, **not** a hyperparameter search or an optimized final design. No fitted scaler or imputer is used; raw quality checks precede model fitting.

## Validation results on identical target examples

| Case and units | Examples | Previous-day seasonal naive MAE | Previous-slot persistence MAE | XGBoost MAE | XGBoost RMSE |
| --- | ---: | ---: | ---: | ---: | ---: |
| AEMO `NSW1`, MW | 2,928 | 393.42 | 188.01 | **90.37** | 122.69 |
| Ausgrid customer `1` `GG`, kWh/half-hour | 4,230 | 0.1567 | 0.0850 | **0.0676** | 0.1406 |

XGBoost beats both simple reference forecasts on **these validation periods** and has lower MAE than the [first LSTM run](lstm_experiment.md). This does not show that it will outperform them on the reserved test periods, a different region or household, or a longer forecast horizon. The one-step task benefits substantially from access to the immediately prior actual value; the persistence comparator is therefore important and should not be omitted.

Positive-actual-only MAPE is 1.26% for AEMO and 41.71% for PV on validation. The PV metric excludes **2,029 zero-actual examples** and remains sensitive to tiny positive generation values. MAE and RMSE are the primary all-example comparisons for PV. Do not compare MW and kWh errors directly.

XGBoost's native `pred_contribs` option provides SHAP feature contributions on **validation rows only**. Mean absolute attribution for AEMO was highest for previous-slot actual (~875.5 MW), followed by target slot (~144.7 MW). For PV the previous-slot actual also dominated (~0.3543 kWh). The first local contribution sum was checked against its raw pre-clipping prediction. These explain the fitted model's output, not physical causation; if a raw prediction is negative, the clipped displayed forecast is not exactly the SHAP sum. See the [XGBoost API](https://xgboost.readthedocs.io/en/stable/python/python_api.html) for the contribution option.

## Reproduce locally

Create a Python 3.12+ virtual environment, install `requirements.txt`, keep the raw files outside Git or under ignored `data/`, and run:

```text
python -m unittest discover -s tests -v
python scripts/run_xgboost_validation.py DIRECTORY_CONTAINING_12_AEMO_MONTHLY_ZIPS PATH_TO_AUSGRID_SOLAR_HOME_DATA.zip
```

The source files, month list and acquisition caveats are in [the data-source inventory](data_sources.md). AEMO archives are rolling, and the Ausgrid original download link is no longer available; provenance must be restated for any new copy. `scripts/run_xgboost_validation.py` reports validation metrics but **does not score the test partition**. A first LSTM comparison and dashboard now exist; team confirmation, a frozen selection protocol and one-time test evaluation remain future work.
