import numpy as np
import torch
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from typing import Tuple, List, Optional


class TimeSeriesSequenceDataset(Dataset):
    """
    Transforms 2D tabular arrays into 3D sliding window tensors:
    Shape: (num_samples, sequence_length, num_features)
    """
    def __init__(self, features: np.ndarray, targets: np.ndarray, sequence_length: int = 24):
        self.sequence_length = sequence_length
        self.X, self.y = [], []

        for i in range(len(features) - sequence_length):
            self.X.append(features[i : i + sequence_length])
            self.y.append(targets[i + sequence_length])

        self.X = torch.tensor(np.array(self.X), dtype=torch.float32)
        self.y = torch.tensor(np.array(self.y), dtype=torch.float32).unsqueeze(1)

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.X[idx], self.y[idx]


def build_sequence_loaders(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    feature_cols: List[str],
    target_col: str = "load",
    seq_length: int = 24,
    batch_size: int = 64
) -> Tuple[DataLoader, DataLoader, DataLoader, StandardScaler, StandardScaler]:
    """
    Scales features and generates PyTorch DataLoaders. Fits scalers strictly on training data
    to prevent data leakage across validation and test partitions.
    """
    feat_scaler = StandardScaler()
    target_scaler = StandardScaler()

    train_feats = feat_scaler.fit_transform(train_df[feature_cols])
    train_targets = target_scaler.fit_transform(train_df[[target_col]]).ravel()

    val_feats = feat_scaler.transform(val_df[feature_cols])
    val_targets = target_scaler.transform(val_df[[target_col]]).ravel()

    test_feats = feat_scaler.transform(test_df[feature_cols])
    test_targets = target_scaler.transform(test_df[[target_col]]).ravel()

    train_ds = TimeSeriesSequenceDataset(train_feats, train_targets, seq_length)
    val_ds = TimeSeriesSequenceDataset(val_feats, val_targets, seq_length)
    test_ds = TimeSeriesSequenceDataset(test_feats, test_targets, seq_length)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=False)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader, feat_scaler, target_scaler