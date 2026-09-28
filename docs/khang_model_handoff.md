# AI/ML Model Handoff Documentation

**Author:** Khang Huynh (AI/ML Model Lead)  
**Target Audience:** Hitesh Kumar (UI & Integration Lead) & Nguyen Hoang Viet (Explainability & QA Lead)  
**Status:** Active / Sprint Deliverable  

---

## 1. Overview
This document specifies the technical handoff from the **AI/ML Model Development** stream to the **UI Presentation Layer** and **Explainability/QA Layer**. It details model outputs, serialized artifacts, prediction payload schemas, KPI verification equations, and execution instructions.

---

## 2. Generated Model Artifacts (`models/`)
Following execution of the validation scripts, serialized model files are saved to the `models/` directory for downstream explainability analysis and re-use:

| Model Artifact Path | File Format | Created By | Consumer | Purpose |
| :--- | :--- | :--- | :--- | :--- |
| `models/xgb_aemo_nsw1.joblib` | Joblib | `run_xgboost_validation.py` | Viet (Explainability) | Exact tree-based SHAP feature attribution for AEMO demand forecasting. |
| `models/xgb_ausgrid_customer1_gg.joblib` | Joblib | `run_xgboost_validation.py` | Viet (Explainability) | Tree-based SHAP feature attribution for Ausgrid PV generation. |
| `models/lstm_best.pt` | PyTorch State Dict | `run_lstm_validation.py` | Viet (QA / Extension) | PyTorch model state loading for sequence inspection and neural evaluation. |

### Loading Artifacts in Python
```python
import joblib
import torch

# 1. Load XGBoost Model for SHAP
xgb_aemo = joblib.load("models/xgb_aemo_nsw1.joblib")

# 2. Load PyTorch LSTM Model
lstm_model_state = torch.load("models/lstm_best.pt")
```

---

## 3. UI Integration Payload Contract (`results/`)
Model validation produces standardized JSON payloads exported to `results/`. **Hitesh (UI Lead)** consumes these payloads directly in Streamlit to render actual vs. predicted charts, timestamps, and metric comparisons across both forecasting cases (`AEMO_NSW1` and `Ausgrid_Customer1_GG`).

### JSON Schema Contract: `results/xgboost_aemo_nsw1_results.json`
```json
{
    "case_name": "AEMO_NSW1",
    "target_col": "load",
    "unit": "MW",
    "model_name": "XGBoost",
    "metrics": {
        "MAE": 2.5213,
        "RMSE": 3.0778,
        "MAPE": 2.8315
    },
    "timestamps": [
        "2026-08-01 00:00:00",
        "2026-08-01 00:30:00"
    ],
    "actuals": [122.0, 121.5],
    "predictions": [120.4, 122.1]
}
```

---

## 4. Key Performance Indicator (KPI) Verification
The pipeline automatically tests whether candidate ML models achieve an out-of-sample MAE at least 10% lower than the Seasonal-Naive baseline floor:

$$\text{Improvement} = \frac{\text{MAE}_{\text{Naive}} - \text{MAE}_{\text{Model}}}{\text{MAE}_{\text{Naive}}} \times 100$$

If `mae_improvement_pct >= 10.0`, `kpi_target_met` evaluates to `true` in the exported JSON payload.

---

## 5. How to Run Model Validation
To execute the complete modeling workflow and regenerate all handoff artifacts across both dataset cases, run:

```bash
# Run multi-case validation pipelines
python scripts/run_xgboost_validation.py
python scripts/run_lstm_validation.py
```
