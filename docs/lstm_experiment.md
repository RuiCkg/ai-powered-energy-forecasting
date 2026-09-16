# First LSTM validation experiment

This is a small sequence-model comparison on the same **provisional one-step validation targets** as [XGBoost](first_ml_experiment.md), not a final deployment claim. AEMO `NSW1` demand and Ausgrid customer `1` `GG` PV are trained separately, in different units. Each example contains the 48 source values before its target plus known target calendar fields; no current or future target enters the input. Ausgrid's 48 slots remain in source order without inventing daylight-saving timestamps, and PV day-boundary forecasts are omitted.

The one-layer CPU LSTM has 16 hidden units, an Adam optimizer at learning rate 0.001, seed 42, batches of 512, at most 20 epochs, and an MAE early-stop patience of three epochs. Target normalization is fitted on **train only**. The checkpoint is chosen on validation only, and predictions are clipped at zero. The [PyTorch LSTM documentation](https://docs.pytorch.org/docs/2.14/generated/torch.nn.LSTM.html) describes the recurrent layer. No hyperparameter sweep, test scoring or cross-customer experiment has been performed.

| Case | Validation targets | Seasonal-naive MAE | Persistence MAE | XGBoost MAE | LSTM MAE | LSTM RMSE | Best LSTM epoch |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| AEMO `NSW1`, MW | 2,928 | 393.42 | 188.01 | 90.37 | 127.15 | 163.28 | 20 |
| Ausgrid customer `1` `GG`, kWh/source slot | 4,230 | 0.1567 | 0.0850 | 0.0676 | 0.0692 | 0.1411 | 19 |

On these validation periods the LSTM beats both simple references, but the fixed XGBoost candidate has lower MAE. The AEMO LSTM still improved at epoch 20, so this result is a **bounded first run**, not an optimized sequence model. Positive-actual MAPE was 1.77% for demand and 42.67% for PV, with 2,029 zero PV actuals excluded. MAE and RMSE include all valid targets. This says nothing yet about the reserved test periods, longer forecast horizons, or real-time data latency.

To reproduce after obtaining the documented archives outside Git:

```text
python -m pip install -r requirements.txt -r requirements-lstm.txt
python scripts/run_lstm_validation.py DIRECTORY_CONTAINING_12_AEMO_MONTHLY_ZIPS PATH_TO_AUSGRID_SOLAR_HOME_DATA.zip
```

The run prints training progress to standard error and a compact validation JSON result to standard output. `streamlit_app.py` can also produce a validation forecast CSV and chart for either case.
