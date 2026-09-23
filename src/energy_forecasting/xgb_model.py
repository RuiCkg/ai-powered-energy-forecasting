import os
import json
import joblib
import numpy as np
import pandas as pd
import xgboost as xgb
from typing import List, Dict, Tuple, Any, Optional
from sklearn.metrics import mean_absolute_error, mean_squared_error


def build_xgb_features(
    df: pd.DataFrame,
    target_col: str = "load",
    lags: List[int] = [1, 2, 24, 48, 168],
    rolling_windows: List[int] = [3, 6, 24, 168]
) -> pd.DataFrame:
    """
    Transforms clean time-series data from Rui's pipeline into feature-rich tabular data.
    Engineers calendar encodings, historical lag observations, and rolling statistics.
    """
    data = df.copy()

    if "timestamp" in data.columns:
        data["timestamp"] = pd.to_datetime(data["timestamp"])
        data = data.sort_values("timestamp").reset_index(drop=True)

        data["hour"] = data["timestamp"].dt.hour
        data["dayofweek"] = data["timestamp"].dt.dayofweek
        data["month"] = data["timestamp"].dt.month
        data["is_weekend"] = data["dayofweek"].isin([5, 6]).astype(int)

        # Cyclical transformations for time periodicities
        data["hour_sin"] = np.sin(2 * np.pi * data["hour"] / 24.0)
        data["hour_cos"] = np.cos(2 * np.pi * data["hour"] / 24.0)
        data["month_sin"] = np.sin(2 * np.pi * data["month"] / 12.0)
        data["month_cos"] = np.cos(2 * np.pi * data["month"] / 12.0)

    # Historical lag features
    for lag in lags:
        if target_col in data.columns:
            data[f"{target_col}_lag_{lag}"] = data[target_col].shift(lag)

    # Moving window aggregations
    for window in rolling_windows:
        if target_col in data.columns:
            data[f"{target_col}_rolling_mean_{window}"] = (
                data[target_col].shift(1).rolling(window=window).mean()
            )
            data[f"{target_col}_rolling_std_{window}"] = (
                data[target_col].shift(1).rolling(window=window).std()
            )
            data[f"{target_col}_rolling_max_{window}"] = (
                data[target_col].shift(1).rolling(window=window).max()
            )
            data[f"{target_col}_rolling_min_{window}"] = (
                data[target_col].shift(1).rolling(window=window).min()
            )

    return data.dropna().reset_index(drop=True)


class XGBoostForecaster:
    """
    XGBoost Baseline and Tuned Regressor for Energy Load and PV Forecasting.
    Includes feature importance extractors and model serialization methods.
    """
    def __init__(
        self,
        n_estimators: int = 400,
        max_depth: int = 6,
        learning_rate: float = 0.03,
        subsample: float = 0.8,
        colsample_bytree: float = 0.8,
        random_state: int = 42
    ):
        self.params = {
            "n_estimators": n_estimators,
            "max_depth": max_depth,
            "learning_rate": learning_rate,
            "subsample": subsample,
            "colsample_bytree": colsample_bytree,
            "random_state": random_state,
            "objective": "reg:squarederror",
            "n_jobs": -1
        }
        self.model = xgb.XGBRegressor(**self.params)
        self.feature_names: List[str] = []

    def fit(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: Optional[pd.DataFrame] = None,
        y_val: Optional[pd.Series] = None
    ):
        self.feature_names = list(X_train.columns)
        eval_set = [(X_train, y_train)]
        if X_val is not None and y_val is not None:
            eval_set.append((X_val, y_val))

        self.model.fit(
            X_train,
            y_train,
            eval_set=eval_set,
            verbose=False
        )

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return self.model.predict(X[self.feature_names])

    def get_feature_importance(self) -> pd.DataFrame:
        importances = self.model.feature_importances_
        return pd.DataFrame({
            "feature": self.feature_names,
            "importance": importances
        }).sort_values("importance", ascending=False).reset_index(drop=True)

    def save_model(self, filepath: str):
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        joblib.dump(self.model, filepath)

    def load_model(self, filepath: str):
        self.model = joblib.load(filepath)