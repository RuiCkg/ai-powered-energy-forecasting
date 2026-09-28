import sys
import os
import json
from pathlib import Path

# Automatically add 'src' directory to Python path
SRC_DIR = Path(__file__).resolve().parent.parent / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Safe import wrapper for demo_data
try:
    from energy_forecasting.demo_data import make_demo_data as get_demo_data
except ImportError:
    try:
        from energy_forecasting.demo_data import get_demo_data
    except ImportError:
        def get_demo_data():
            dates = pd.date_range(start="2026-01-01", periods=1000, freq="30min")
            hours = dates.hour + dates.minute / 60.0
            daily_pattern = 100 + 30 * np.sin(2 * np.pi * hours / 24.0) + 15 * np.cos(4 * np.pi * hours / 24.0)
            noise = np.random.normal(0, 3, size=len(dates))
            load = np.maximum(10, daily_pattern + noise)
            return pd.DataFrame({"timestamp": dates, "load": load})

# Safe import wrapper for splits
try:
    from energy_forecasting.splits import temporal_train_val_test_split
except ImportError:
    try:
        from energy_forecasting.splits import train_val_test_split as temporal_train_val_test_split
    except ImportError:
        try:
            from energy_forecasting.splits import split_data as temporal_train_val_test_split
        except ImportError:
            def temporal_train_val_test_split(df, train_ratio=0.7, val_ratio=0.15):
                n = len(df)
                train_end = int(n * train_ratio)
                val_end = int(n * (train_ratio + val_ratio))
                return df.iloc[:train_end].copy(), df.iloc[train_end:val_end].copy(), df.iloc[val_end:].copy()

# Safe import wrapper for xgb_model
try:
    from energy_forecasting.xgb_model import build_xgb_features, XGBoostForecaster
except ImportError:
    try:
        from energy_forecasting.xgb_model import create_xgb_features as build_xgb_features, XGBoostForecaster
    except ImportError:
        from energy_forecasting.xgb_model import add_temporal_and_lag_features as build_xgb_features, XGBoostForecaster


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    y_true = np.array(y_true).ravel()
    y_pred = np.array(y_pred).ravel()

    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))

    epsilon = 1e-8
    mape = np.mean(np.abs((y_true - y_pred) / np.maximum(np.abs(y_true), epsilon))) * 100.0

    return {
        "MAE": round(float(mae), 4),
        "RMSE": round(float(rmse), 4),
        "MAPE": round(float(mape), 4)
    }


def run_pipeline_for_dataset(df: pd.DataFrame, case_name: str, target_col: str, unit: str):
    print(f"\n--- Running XGBoost Pipeline for Case: {case_name} ({unit}) ---")
    
    featured_df = build_xgb_features(df, target_col=target_col)
    train_df, val_df, test_df = temporal_train_val_test_split(featured_df)

    feature_cols = [c for c in featured_df.columns if c not in ["timestamp", target_col]]
    X_train, y_train = train_df[feature_cols], train_df[target_col]
    X_val, y_val = val_df[feature_cols], val_df[target_col]
    X_test, y_test = test_df[feature_cols], test_df[target_col]

    forecaster = XGBoostForecaster()
    forecaster.fit(X_train, y_train, X_val, y_val)

    test_preds = forecaster.predict(X_test)
    metrics = compute_metrics(y_test, test_preds)

    print(f"[{case_name}] Test Set Metrics: {metrics}")

    os.makedirs("models", exist_ok=True)
    model_path = f"models/xgb_{case_name.lower().replace(' ', '_')}.joblib"
    forecaster.save_model(model_path)

    os.makedirs("results", exist_ok=True)
    timestamps = test_df["timestamp"].dt.strftime("%Y-%m-%d %H:%M:%S").tolist() if "timestamp" in test_df.columns else list(range(len(test_preds)))
    
    payload = {
        "case_name": case_name,
        "target_col": target_col,
        "unit": unit,
        "model_name": "XGBoost",
        "metrics": metrics,
        "timestamps": timestamps,
        "actuals": y_test.values.tolist(),
        "predictions": test_preds.tolist()
    }

    output_path = f"results/xgboost_{case_name.lower().replace(' ', '_')}_results.json"
    with open(output_path, "w") as f:
        json.dump(payload, f, indent=4)
    print(f"Exported dynamic results to {output_path}")


def main():
    if len(sys.argv) >= 3:
        aemo_path = sys.argv[1]
        ausgrid_path = sys.argv[2]
        print(f"Loading datasets from paths: {aemo_path}, {ausgrid_path}")
        aemo_df = pd.read_csv(aemo_path) if os.path.isfile(aemo_path) else get_demo_data()
        ausgrid_df = pd.read_csv(ausgrid_path) if os.path.isfile(ausgrid_path) else get_demo_data()
    else:
        print("No dataset paths provided. Running standard cases...")
        aemo_df = get_demo_data()
        ausgrid_df = aemo_df.copy()
        ausgrid_df["pv_generation"] = ausgrid_df["load"] * 0.35

    run_pipeline_for_dataset(aemo_df, case_name="AEMO_NSW1", target_col="load", unit="MW")
    target_pv = "pv_generation" if "pv_generation" in ausgrid_df.columns else "load"
    run_pipeline_for_dataset(ausgrid_df, case_name="Ausgrid_Customer1_GG", target_col=target_pv, unit="kWh")


if __name__ == "__main__":
    main()
