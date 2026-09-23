import os
import json
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error

from energy_forecasting.demo_data import get_demo_data
from energy_forecasting.splits import temporal_train_val_test_split
from energy_forecasting.sequences import build_sequence_loaders
from energy_forecasting.lstm_model import PyTorchLSTMForecaster


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
    print("Running LSTM Validation Execution...")
    raw_df = get_demo_data()
    train_df, val_df, test_df = temporal_train_val_test_split(raw_df)

    feature_cols = [c for c in raw_df.columns if c not in ["timestamp", "load"]]
    if not feature_cols:
        feature_cols = ["load"]

    seq_length = 24
    train_loader, val_loader, test_loader, feat_scaler, target_scaler = build_sequence_loaders(
        train_df, val_df, test_df, feature_cols=feature_cols, target_col="load", seq_length=seq_length
    )

    input_dim = len(feature_cols)
    forecaster = PyTorchLSTMForecaster(input_dim=input_dim, hidden_dim=64, num_layers=2)

    forecaster.train_with_early_stopping(
        train_loader, val_loader, epochs=30, patience=5, checkpoint_path="models/lstm_best.pt"
    )

    preds = forecaster.predict(test_loader, target_scaler=target_scaler)

    actuals = test_df["load"].values[seq_length:]
    metrics = compute_metrics(actuals, preds)

    print(f"LSTM Test Metrics: {metrics}")

    # Export JSON prediction payload for Hitesh (UI)
    os.makedirs("results", exist_ok=True)
    test_timestamps = test_df["timestamp"].iloc[seq_length:].dt.strftime("%Y-%m-%d %H:%M:%S").tolist() if "timestamp" in test_df.columns else list(range(len(preds)))

    payload = {
        "model_name": "LSTM",
        "metrics": metrics,
        "timestamps": test_timestamps,
        "actuals": actuals.tolist(),
        "predictions": preds.ravel().tolist()
    }

    with open("results/lstm_results.json", "w") as f:
        json.dump(payload, f, indent=4)
    print("Saved JSON payload to results/lstm_results.json")


if __name__ == "__main__":
    main()