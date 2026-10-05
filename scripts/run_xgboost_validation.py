import os
import json
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error

from energy_forecasting.demo_data import get_demo_data
from energy_forecasting.splits import temporal_train_val_test_split
from energy_forecasting.xgb_model import build_xgb_features, XGBoostForecaster


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
    aemo_df = get_demo_data()
    run_pipeline_for_dataset(aemo_df, case_name="AEMO_NSW1", target_col="load", unit="MW")

    ausgrid_df = aemo_df.copy()
    ausgrid_df["pv_generation"] = ausgrid_df["load"] * 0.35
    run_pipeline_for_dataset(ausgrid_df, case_name="Ausgrid_Customer1_GG", target_col="pv_generation", unit="kWh")


if __name__ == "__main__":
    main()
