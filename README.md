# AI-Powered Energy Forecasting — Machine Learning Pipeline

**Lead Developer:** Khang Huynh (AI/ML Model Lead)  
**Project Domain:** Short-Term Energy Load & Solar PV Generation Forecasting  
**Tech Stack:** Python 3.12+, XGBoost, PyTorch (LSTM), Scikit-Learn, Pandas, NumPy  

---

## 📌 1. Overview & Scope

This module implements the core machine learning forecasting engine for the AI-Powered Energy Forecasting platform. It supports dual-case prediction across two distinct datasets and targets:

1. **AEMO NSW1 Regional Demand:** Operational load forecasting measured in Megawatts (**MW**).
2. **Ausgrid Customer 1 Gross Solar Generation:** Distributed PV generation forecasting measured in Kilowatt-hours (**kWh**).

The pipeline enforces strict chronological splitting—reserving validation sets for early stopping and hyperparameter tuning, and keeping test sets completely untouched until final out-of-sample evaluation.

---

## 📁 2. Repository Structure

```text
ai-powered-energy-forecasting/
├── docs/
│   └── khang_model_handoff.md        # Technical handoff documentation & payload contracts
├── models/                           # Serialized model artifacts (git-ignored / auto-generated)
│   ├── xgb_aemo_nsw1.joblib
│   ├── xgb_ausgrid_customer1_gg.joblib
│   └── lstm_best.pt
├── results/                          # Dynamic JSON payload outputs for UI consumption
│   ├── xgboost_aemo_nsw1_results.json
│   ├── xgboost_ausgrid_customer1_gg_results.json
│   └── lstm_results.json
├── scripts/                          # Execution and validation entrypoints
│   ├── run_xgboost_validation.py     # Multi-case XGBoost validation runner
│   └── run_lstm_validation.py        # Sequence-based PyTorch LSTM validation runner
├── src/
│   └── energy_forecasting/           # Core Python package
│       ├── __init__.py
│       ├── lstm_model.py             # Stacked LSTM architecture & training manager
│       ├── xgb_model.py              # Feature engineering & XGBoost wrapper
│       ├── sequences.py              # Sliding-window sequence loaders & scalers
│       ├── splits.py                 # Temporal train/validation/test splitters
│       └── demo_data.py              # Fallback synthetic energy data generators
├── requirements.txt                  # Full project dependencies
└── README.md                         # Module documentation
