# AI/ML Model Handoff Documentation

**Author:** Khang Huynh (AI/ML Model Lead)  
**Target Audience:** Hitesh Kumar (UI & Integration Lead) & Nguyen Hoang Viet (Explainability & QA Lead)  
**Status:** Active / Sprint Deliverable  

---

## 1. Overview
This document specifies the technical handoff from the **AI/ML Model Development** stream to the **UI Presentation Layer** and **Explainability/QA Layer**. It details model outputs, serialized artifacts, prediction payload schemas, and execution instructions.

---

## 2. Generated Model Artifacts (`models/`)
Following execution of the training scripts, serialized model files are saved to the `models/` directory for downstream explainability analysis and re-use:

| Model Artifact Path | File Format | Created By | Consumer | Purpose |
| :--- | :--- | :--- | :--- | :--- |
| `models/xgb_model.joblib` | Scikit-Learn / Joblib | `run_xgboost_validation.py` | Viet (Explainability) | Exact tree-based SHAP feature attribution. |
| `models/lstm_best.pt` | PyTorch State Dict | `run_lstm_validation.py` | Viet (QA / Extension) | PyTorch model state loading for sequence inspection. |

### Loading Artifacts in Python
```python
import joblib
import torch

# 1. Load XGBoost Model for SHAP
xgb_model = joblib.load("models/xgb_model.joblib")

# 2. Load PyTorch LSTM Model
lstm_model_state = torch.load("models/lstm_best.pt")
## 3. Key Performance Indicator (KPI) Verification

The pipeline automatically tests whether the candidate ML models achieve an out-of-sample $\text{MAE}$ at least 10% lower than the Seasonal-Naive baseline floor:

$$\text{Improvement \%} = \frac{\text{MAE}_{\text{Naive}} - \text{MAE}_{\text{Model}}}{\text{MAE}_{\text{Naive}}} \times 100\%$$


If `mae_improvement_pct >= 10.0`, `kpi_target_met` evaluates to `true` in the JSON payload.

---

## 4. How to Run Model Validation

To execute the complete modeling workflow and regenerate all handoff artifacts, run:

```bash
# Run multi-model validation pipeline
python scripts/run_xgboost_validation.py
python scripts/run_lstm_validation.py
