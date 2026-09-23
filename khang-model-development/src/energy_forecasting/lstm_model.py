import os
import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Any, Optional
from sklearn.preprocessing import StandardScaler


class StackedLSTMNetwork(nn.Module):
    """
    Stacked LSTM Network for Energy Forecasting.
    Captures temporal dependencies across sliding window sequences.
    """
    def __init__(self, input_dim: int, hidden_dim: int = 64, num_layers: int = 2, dropout: float = 0.2):
        super(StackedLSTMNetwork, self).__init__()
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0
        )
        self.fc_1 = nn.Linear(hidden_dim, 32)
        self.relu = nn.ReLU()
        self.fc_out = nn.Linear(32, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        lstm_out, _ = self.lstm(x)
        last_step = lstm_out[:, -1, :]
        out = self.relu(self.fc_1(last_step))
        return self.fc_out(out)


class PyTorchLSTMForecaster:
    """
    LSTM Execution Manager: Controls GPU/CPU hardware acceleration, Early Stopping,
    model serialization, and inverse prediction scaling.
    """
    def __init__(self, input_dim: int, hidden_dim: int = 64, num_layers: int = 2, lr: float = 0.001):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = StackedLSTMNetwork(input_dim, hidden_dim, num_layers).to(self.device)
        self.criterion = nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)

    def train_with_early_stopping(
        self,
        train_loader,
        val_loader=None,
        epochs: int = 50,
        patience: int = 8,
        checkpoint_path: str = "models/lstm_best.pt"
    ) -> Dict[str, list]:
        best_loss = float("inf")
        patience_counter = 0
        history = {"train_loss": [], "val_loss": []}

        os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)

        for epoch in range(epochs):
            self.model.train()
            running_train_loss = 0.0
            for X_b, y_b in train_loader:
                X_b, y_b = X_b.to(self.device), y_b.to(self.device)

                self.optimizer.zero_grad()
                preds = self.model(X_b)
                loss = self.criterion(preds, y_b)
                loss.backward()
                self.optimizer.step()

                running_train_loss += loss.item() * X_b.size(0)

            epoch_train_loss = running_train_loss / len(train_loader.dataset)
            history["train_loss"].append(epoch_train_loss)

            if val_loader:
                self.model.eval()
                running_val_loss = 0.0
                with torch.no_grad():
                    for X_v, y_v in val_loader:
                        X_v, y_v = X_v.to(self.device), y_v.to(self.device)
                        preds_v = self.model(X_v)
                        val_loss += self.criterion(preds_v, y_v).item() * X_v.size(0)

                epoch_val_loss = running_val_loss / len(val_loader.dataset)
                history["val_loss"].append(epoch_val_loss)

                if epoch_val_loss < best_loss:
                    best_loss = epoch_val_loss
                    patience_counter = 0
                    torch.save(self.model.state_dict(), checkpoint_path)
                else:
                    patience_counter += 1
                    if patience_counter >= patience:
                        break

        if os.path.exists(checkpoint_path):
            self.model.load_state_dict(torch.load(checkpoint_path, map_location=self.device))

        return history

    def predict(self, data_loader, target_scaler: Optional[StandardScaler] = None) -> np.ndarray:
        self.model.eval()
        preds = []
        with torch.no_grad():
            for X_b, _ in data_loader:
                X_b = X_b.to(self.device)
                out = self.model(X_b)
                preds.append(out.cpu().numpy())

        unscaled = np.vstack(preds)
        if target_scaler:
            unscaled = target_scaler.inverse_transform(unscaled)
        return unscaled