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


def main():
    print("Running XGBoost Validation Execution...")
    raw_df = get_demo_data()
    featured_df = build_xgb_features(raw_df, target_col="load")

    train_df, val_df, test_df = temporal_train_val_test_split(featured_df)

    feature_cols = [c for c in featured_df.columns if c not in ["timestamp", "load"]]
    X_train, y_train = train_df[feature_cols], train_df["load"]
    X_val, y_val = val_df[feature_cols], val_df["load"]
    X_test, y_test = test_df[feature_cols], test_df["load"]

    forecaster = XGBoostForecaster()
    forecaster.fit(X_train, y_train, X_val, y_val)

    preds = forecaster.predict(X_test)
    metrics = compute_metrics(y_test, preds)

    print(f"XGBoost Test Metrics: {metrics}")

    # Export serialized model artifact for Viet (Explainability/QA)
    os.makedirs("models", exist_ok=True)
    forecaster.save_model("models/xgb_model.joblib")
    print("Saved model artifact to models/xgb_model.joblib")

    # Export JSON prediction payload for Hitesh (UI)
    os.makedirs("results", exist_ok=True)
    timestamps = test_df["timestamp"].dt.strftime("%Y-%m-%d %H:%M:%S").tolist() if "timestamp" in test_df.columns else list(range(len(preds)))
    
    payload = {
        "model_name": "XGBoost",
        "metrics": metrics,
        "timestamps": timestamps,
        "actuals": y_test.values.tolist(),
        "predictions": preds.tolist()
    }

    with open("results/xgboost_results.json", "w") as f:
        json.dump(payload, f, indent=4)
    print("Saved JSON payload to results/xgboost_results.json")


if __name__ == "__main__":
    main()