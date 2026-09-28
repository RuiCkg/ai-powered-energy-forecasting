import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from typing import Tuple, List, Optional


class TimeSeriesSequenceDataset(Dataset):
    """
    PyTorch Dataset for sliding-window time-series sequences.
    """
    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32).unsqueeze(-1)

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.X[idx], self.y[idx]


def create_sliding_windows(
    data: np.ndarray,
    target: np.ndarray,
    seq_length: int = 24
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Converts 2D tabular array into 3D sliding window sequences:
    X: (samples, seq_length, features), y: (samples,)
    """
    X_seq, y_seq = [], []
    for i in range(len(data) - seq_length):
        X_seq.append(data[i : i + seq_length])
        y_seq.append(target[i + seq_length])
    return np.array(X_seq), np.array(y_seq)


def build_sequence_loaders(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    feature_cols: List[str],
    target_col: str = "load",
    seq_length: int = 24,
    batch_size: int = 32
) -> Tuple[DataLoader, DataLoader, DataLoader, StandardScaler, StandardScaler]:
    """
    Scales features/targets and constructs PyTorch DataLoaders for train, validation, and test splits.
    """
    feat_scaler = StandardScaler()
    target_scaler = StandardScaler()

    # Fit scalers ONLY on training set to prevent data leakage
    X_train_scaled = feat_scaler.fit_transform(train_df[feature_cols])
    y_train_scaled = target_scaler.fit_transform(train_df[[target_col]]).ravel()

    X_val_scaled = feat_scaler.transform(val_df[feature_cols])
    y_val_scaled = target_scaler.transform(val_df[[target_col]]).ravel()

    X_test_scaled = feat_scaler.transform(test_df[feature_cols])
    y_test_scaled = target_scaler.transform(test_df[[target_col]]).ravel()

    # Generate sliding window sequences
    X_tr, y_tr = create_sliding_windows(X_train_scaled, y_train_scaled, seq_length)
    X_va, y_va = create_sliding_windows(X_val_scaled, y_val_scaled, seq_length)
    X_te, y_te = create_sliding_windows(X_test_scaled, y_test_scaled, seq_length)

    train_loader = DataLoader(TimeSeriesSequenceDataset(X_tr, y_tr), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(TimeSeriesSequenceDataset(X_va, y_va), batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(TimeSeriesSequenceDataset(X_te, y_te), batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader, feat_scaler, target_scaler


# Function aliases for backwards compatibility
create_sequence_loaders = build_sequence_loaders
create_sequences = build_sequence_loaders
make_sequence_loaders = build_sequence_loaders
