"""Small CPU LSTM with train-only normalization and validation early stopping."""

from __future__ import annotations

import copy
from math import sqrt

from .baseline import evaluate_predictions, preview_predictions
from .xgb_model import feature_row


def fit_and_validate_lstm(parts: dict[str, list[dict]], case: str, progress=None, include_preview: bool = False) -> dict:
    """Train on source history, select one checkpoint on validation, omit test."""

    try:
        import torch
        from torch import nn
        from torch.utils.data import DataLoader, TensorDataset
    except ImportError as error:
        raise RuntimeError("Install requirements-lstm.txt before training the LSTM") from error

    train, validation = parts["train"], parts["validation"]
    if not train or not validation or not parts["test"]:
        raise ValueError("Chronological train/validation/test sequence examples are required")
    if any(len(example.get("history_48", ())) != 48 for example in train + validation):
        raise ValueError("Every LSTM example needs exactly 48 past source values")
    torch.manual_seed(42)
    torch.set_num_threads(2)
    target_mean = sum(example["target"] for example in train) / len(train)
    target_std = sqrt(sum((example["target"] - target_mean) ** 2 for example in train) / len(train))
    if target_std < 1e-9:
        target_std = 1.0

    def tensors(examples):
        histories = torch.tensor(
            [[(value - target_mean) / target_std for value in example["history_48"]] for example in examples],
            dtype=torch.float32,
        )
        calendars = torch.tensor(
            [[feature_row(example, case)[2] / 48, feature_row(example, case)[3] / 6,
              feature_row(example, case)[4] / 12] for example in examples],
            dtype=torch.float32,
        )
        targets = torch.tensor([(example["target"] - target_mean) / target_std for example in examples], dtype=torch.float32)
        return TensorDataset(histories, calendars, targets)

    train_loader = DataLoader(tensors(train), batch_size=512, shuffle=True, generator=torch.Generator().manual_seed(42), num_workers=0)
    validation_loader = DataLoader(tensors(validation), batch_size=1024, shuffle=False, num_workers=0)

    class TinyLSTM(nn.Module):
        def __init__(self):
            super().__init__()
            self.recurrent = nn.LSTM(input_size=1, hidden_size=16, batch_first=True)
            self.head = nn.Linear(19, 1)

        def forward(self, history, calendar):
            _, (hidden, _) = self.recurrent(history.unsqueeze(-1))
            return self.head(torch.cat((hidden[-1], calendar), dim=1)).squeeze(-1)

    model = TinyLSTM()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    def predict_validation():
        model.eval()
        predictions = []
        with torch.no_grad():
            for histories, calendars, _ in validation_loader:
                outputs = model(histories, calendars).tolist()
                predictions.extend(max(0.0, float(output) * target_std + target_mean) for output in outputs)
        return predictions

    best_mae = float("inf")
    best_epoch = 0
    best_state = None
    non_improving = 0
    epochs_run = 0
    for epoch in range(1, 21):
        model.train()
        for histories, calendars, targets in train_loader:
            optimizer.zero_grad()
            loss = nn.functional.mse_loss(model(histories, calendars), targets)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        predictions = predict_validation()
        validation_mae = evaluate_predictions(validation, predictions)["MAE"]
        epochs_run = epoch
        if progress is not None:
            progress(f"{case} epoch {epoch}: validation MAE {validation_mae:.5f}")
        if validation_mae + 1e-6 < best_mae:
            best_mae = validation_mae
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            non_improving = 0
        else:
            non_improving += 1
            if non_improving >= 3:
                break
    model.load_state_dict(best_state)
    validation_predictions = predict_validation()
    result = {
        "model": "one-layer CPU LSTM, 48 prior source values and known calendar",
        "torch_version": torch.__version__,
        "hidden_size": 16,
        "optimizer": "Adam, learning rate 0.001",
        "max_epochs": 20,
        "early_stop_patience_epochs": 3,
        "epochs_run": epochs_run,
        "best_epoch_selected_on_validation": best_epoch,
        "train_only_target_mean": target_mean,
        "train_only_target_std": target_std,
        "nonnegative_predictions": True,
        "validation_metrics": evaluate_predictions(validation, validation_predictions),
        "validation_seasonal_naive_metrics": evaluate_predictions(validation, [example["lag_48"] for example in validation]),
        "validation_persistence_metrics": evaluate_predictions(validation, [example["lag_1"] for example in validation]),
        "test_status": "reserved; not used for fitting, normalization, checkpoint selection or scoring",
    }
    if include_preview:
        result["validation_prediction_rows"] = preview_predictions(validation, validation_predictions, len(validation))
    return result
