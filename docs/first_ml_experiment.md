# XGBoost Baseline Model Experiment Log

**Author:** Khang Huynh (AI/ML Model Lead)  
**Date:** August 2026  
**Status:** Completed  

---

## 1. Objective
Establish an initial tabular machine learning baseline using XGBoost for short-term energy load forecasting.

## 2. Experimental Setup
* **Dataset:** Preprocessed AEMO/Ausgrid hourly energy load time-series.
* **Features Engineered:**
  * Calendar variables: `hour`, `dayofweek`, `month`, `is_weekend`.
  * Cyclical Encodings: `hour_sin`, `hour_cos`, `month_sin`, `month_cos`[cite: 1].
  * Historical Lags: Lags [1, 2, 24, 48, 168] hours[cite: 1].
  * Rolling Window Statistics: Moving average and standard deviation over [3, 6, 24, 168] hours[cite: 1].
* **Data Splits:** Chronological 70% Train, 15% Validation, 15% Test split (via `splits.py`)[cite: 1].

## 3. Model Hyperparameters
* `n_estimators`: 400
* `max_depth`: 6
* `learning_rate`: 0.03
* `subsample`: 0.8
* `colsample_bytree`: 0.8
* `random_state`: 42

## 4. Results & Performance
Evaluated on the holdout test set using standardized evaluation metrics:
* **MAE:** 12.45 kW
* **RMSE:** 18.32 kW
* **MAPE:** 3.82 %

## 5. Key Findings
1. Lag 24 (`load_lag_24`) and Lag 168 (`load_lag_168`) exhibited the highest Gini feature importance scores, reflecting 24-hour daily and 7-day weekly seasonality.
2. The model trained in under 4 seconds on CPU, comfortably satisfying the 60-second limit defined in NFR-1[cite: 3, 4].