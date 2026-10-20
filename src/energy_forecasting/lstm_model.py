"""Small one-layer CPU LSTM over the 48 past values plus known target calendar fields."""

from __future__ import annotations

import math
from datetime import datetime

from .baseline import evaluate_predictions, preview_predictions

HIDDEN_UNITS = 16
LEARNING_RATE = 0.001
BATCH_SIZE = 512
MAX_EPOCHS = 20
PATIENCE = 3
SEED = 42

FIXED_CONFIG = {
    "hidden_units": HIDDEN_UNITS,
    "learning_rate": LEARNING_RATE,
    "batch_size": BATCH_SIZE,
    "max_epochs": MAX_EPOCHS,
    "patience": PATIENCE,
    "seed": SEED,
}



def calendar_fields(example: dict, case: str) -> list[float]:
    """Cyclic encodings of the known target slot, weekday and month (no target value)."""

    key = example["key"]
    if case in {"aemo", "custom"}:
        if not isinstance(key, datetime):
            raise ValueError(f"{case} example key must be a timestamp")
        slot = example.get("target_slot", key.hour * 2 + key.minute // 30)
        day = key.date()
    elif case == "ausgrid":
        day, slot = key
        slot -= 1
    else:
        raise ValueError(f"Unknown case: {case}")
    slot_period = int(example.get("slots_per_season", 48))
    fields = []
    for value, period in ((slot, slot_period), (day.weekday(), 7), (day.month - 1, 12)):
        angle = 2 * math.pi * value / period
        fields.extend([math.sin(angle), math.cos(angle)])
    if case == "custom":
        fields.extend(
            float(value)
            for _, value in sorted(example.get("custom_covariates", {}).items())
        )
    return fields


def fit_and_validate_lstm(parts: dict, case: str, include_preview: bool = False) -> dict:
    """Train on ``train`` (scaler fitted on train only); choose the checkpoint by validation MAE."""

    try:
        import numpy as np
        import torch
        from torch import nn
    except ImportError as error:
        raise RuntimeError(f"LSTM dependencies are not installed: {error}") from error

    torch.manual_seed(SEED)
    np.random.seed(SEED)
    torch.set_num_threads(2)

    def arrays(examples):
        history = np.array([example["history_48"] for example in examples], dtype=np.float32)
        calendar = np.array([calendar_fields(example, case) for example in examples], dtype=np.float32)
        target = np.array([example["target"] for example in examples], dtype=np.float32)
        return history, calendar, target

    h_train, c_train, y_train = arrays(parts["train"])
    h_val, c_val, y_val = arrays(parts["validation"])
    mean, std = float(h_train.mean()), float(h_train.std()) or 1.0  # train-only normalisation

    if case == "custom":
        calendar_mean = c_train.mean(axis=0)
        calendar_std = c_train.std(axis=0)
        calendar_std[calendar_std == 0] = 1.0
        c_train = (c_train - calendar_mean) / calendar_std
        c_val = (c_val - calendar_mean) / calendar_std

    def scale(history):
        return torch.tensor((history - mean) / std).unsqueeze(-1)

    x_train, x_val = scale(h_train), scale(h_val)
    c_train_t, c_val_t = torch.tensor(c_train), torch.tensor(c_val)
    t_train = torch.tensor((y_train - mean) / std)

    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.lstm = nn.LSTM(input_size=1, hidden_size=HIDDEN_UNITS, num_layers=1, batch_first=True)
            self.head = nn.Linear(HIDDEN_UNITS + c_train.shape[1], 1)

        def forward(self, sequence, calendar):
            output, _ = self.lstm(sequence)
            return self.head(torch.cat([output[:, -1, :], calendar], dim=1)).squeeze(-1)

    def predict(model, sequence, calendar):
        model.eval()
        with torch.no_grad():
            scaled = torch.cat([
                model(sequence[i:i + BATCH_SIZE], calendar[i:i + BATCH_SIZE])
                for i in range(0, len(sequence), BATCH_SIZE)
            ]).numpy()
        return np.maximum(0.0, scaled * std + mean)  # inverse scaling, then non-negative clip

    model = Net()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    loss_fn = nn.MSELoss()
    generator = torch.Generator().manual_seed(SEED)
    best_mae, best_epoch, best_state, stale, history_log = float("inf"), 0, None, 0, []

    for epoch in range(1, MAX_EPOCHS + 1):
        model.train()
        order = torch.randperm(len(x_train), generator=generator)
        for start in range(0, len(order), BATCH_SIZE):
            batch = order[start:start + BATCH_SIZE]
            optimizer.zero_grad()
            loss = loss_fn(model(x_train[batch], c_train_t[batch]), t_train[batch])
            loss.backward()
            optimizer.step()
        val_mae = float(np.abs(predict(model, x_val, c_val_t) - y_val).mean())
        history_log.append(val_mae)
        if val_mae < best_mae:
            best_mae, best_epoch, stale = val_mae, epoch, 0
            best_state = {key: value.clone() for key, value in model.state_dict().items()}
        else:
            stale += 1
            if stale >= PATIENCE:
                break

    model.load_state_dict(best_state)
    predictions = [float(value) for value in predict(model, x_val, c_val_t)]
    result = {
        "model": "LSTM",
        "validation_metrics": evaluate_predictions(parts["validation"], predictions),
        "best_epoch": best_epoch,
        "epochs_run": len(history_log),
        "validation_MAE_by_epoch": history_log,
        "parameters": {
            "hidden_units": HIDDEN_UNITS, "layers": 1, "learning_rate": LEARNING_RATE,
            "batch_size": BATCH_SIZE, "max_epochs": MAX_EPOCHS, "patience": PATIENCE, "seed": SEED,
        },
        "feature_names": [
            "history_window",
            "slot_sin", "slot_cos", "weekday_sin", "weekday_cos", "month_sin", "month_cos",
            *(
                sorted(parts["train"][0].get("custom_covariates", {}))
                if case == "custom" else []
            ),
        ],
    }
    result["validation_prediction_rows"] = (
        preview_predictions(parts["validation"], predictions, limit=len(predictions)) if include_preview else []
    )
    return result


def fit_and_evaluate_test_lstm(
    parts: dict,
    case: str,
    epochs: int,
    include_preview: bool = False,
) -> dict:
    """Retrain for the validation-selected epoch count and score reserved test."""

    try:
        import numpy as np
        import torch
        from torch import nn
    except ImportError as error:
        raise RuntimeError(f"LSTM dependencies are not installed: {error}") from error
    if not isinstance(epochs, int) or epochs < 1:
        raise ValueError("Final LSTM epoch count must be a positive integer")

    torch.manual_seed(SEED)
    np.random.seed(SEED)
    torch.set_num_threads(2)
    development = list(parts["train"]) + list(parts["validation"])

    def arrays(examples):
        history = np.array([example["history_48"] for example in examples], dtype=np.float32)
        calendar = np.array([calendar_fields(example, case) for example in examples], dtype=np.float32)
        target = np.array([example["target"] for example in examples], dtype=np.float32)
        return history, calendar, target

    h_development, c_development, y_development = arrays(development)
    h_test, c_test, y_test = arrays(parts["test"])
    mean = float(h_development.mean())
    std = float(h_development.std()) or 1.0
    if case == "custom":
        calendar_mean = c_development.mean(axis=0)
        calendar_std = c_development.std(axis=0)
        calendar_std[calendar_std == 0] = 1.0
        c_development = (c_development - calendar_mean) / calendar_std
        c_test = (c_test - calendar_mean) / calendar_std

    def scale(history):
        return torch.tensor((history - mean) / std).unsqueeze(-1)

    x_development, x_test = scale(h_development), scale(h_test)
    c_development_t, c_test_t = torch.tensor(c_development), torch.tensor(c_test)
    t_development = torch.tensor((y_development - mean) / std)

    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.lstm = nn.LSTM(input_size=1, hidden_size=HIDDEN_UNITS, num_layers=1, batch_first=True)
            self.head = nn.Linear(HIDDEN_UNITS + c_development.shape[1], 1)

        def forward(self, sequence, calendar):
            output, _ = self.lstm(sequence)
            return self.head(torch.cat([output[:, -1, :], calendar], dim=1)).squeeze(-1)

    model = Net()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    loss_fn = nn.MSELoss()
    generator = torch.Generator().manual_seed(SEED)
    for _ in range(epochs):
        model.train()
        order = torch.randperm(len(x_development), generator=generator)
        for start in range(0, len(order), BATCH_SIZE):
            batch = order[start:start + BATCH_SIZE]
            optimizer.zero_grad()
            loss = loss_fn(model(x_development[batch], c_development_t[batch]), t_development[batch])
            loss.backward()
            optimizer.step()

    model.eval()
    with torch.no_grad():
        scaled = torch.cat([
            model(x_test[index:index + BATCH_SIZE], c_test_t[index:index + BATCH_SIZE])
            for index in range(0, len(x_test), BATCH_SIZE)
        ]).numpy()
    predictions = [float(value) for value in np.maximum(0.0, scaled * std + mean)]
    result = {
        "model": "LSTM",
        "test_metrics": evaluate_predictions(parts["test"], predictions),
        "epochs_selected_on_validation": epochs,
        "training_partition": "train plus validation",
        "evaluation_partition": "reserved test",
    }
    result["test_prediction_rows"] = (
        preview_predictions(parts["test"], predictions, limit=len(predictions))
        if include_preview else []
    )
    return result
