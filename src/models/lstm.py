"""
PyTorch LSTM Architecture Module for Solar Flare Forecasting.
"""

import os
import torch
import torch.nn as nn
import numpy as np

class SolarLSTMModel(nn.Module):
    def __init__(
        self,
        input_size: int,
        hidden_size: int = 128,
        num_layers: int = 2,
        dropout: float = 0.2
    ):
        super(SolarLSTMModel, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0
        )
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: [batch_size, sequence_length, input_size]
        out, (hn, cn) = self.lstm(x)
        # Take the hidden output of the last timestep
        last_step_out = out[:, -1, :]  # shape: [batch_size, hidden_size]
        logits = self.fc(last_step_out).squeeze(-1)  # shape: [batch_size]
        return logits

class SolarLSTMWrapper:
    """Wrapper class providing sklearn-like API (predict_proba, predict, save, load)."""
    def __init__(
        self,
        input_size: int = None,
        hidden_size: int = 128,
        num_layers: int = 2,
        dropout: float = 0.2,
        device: torch.device = None
    ):
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device
            
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        
        if input_size is not None:
            self.model = SolarLSTMModel(
                input_size=input_size,
                hidden_size=hidden_size,
                num_layers=num_layers,
                dropout=dropout
            ).to(self.device)
        else:
            self.model = None

    def predict_proba(self, X_seq: np.ndarray) -> np.ndarray:
        """
        Input X_seq: 3D numpy array [samples, seq_length, num_features]
        Returns 1D numpy array of probabilities [samples]
        """
        self.model.eval()
        X_tensor = torch.tensor(X_seq, dtype=torch.float32).to(self.device)
        with torch.no_grad():
            logits = self.model(X_tensor)
            probas = torch.sigmoid(logits).cpu().numpy()
        return probas

    def predict(self, X_seq: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        probas = self.predict_proba(X_seq)
        return (probas >= threshold).astype(int)

    def save(self, filepath: str = "models/lstm.pt"):
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        checkpoint = {
            "state_dict": self.model.state_dict(),
            "input_size": self.input_size,
            "hidden_size": self.hidden_size,
            "num_layers": self.num_layers,
            "dropout": self.dropout
        }
        torch.save(checkpoint, filepath)
        print(f"[LSTM] PyTorch model state saved to: {filepath}")

    @classmethod
    def load(cls, filepath: str = "models/lstm.pt", device: torch.device = None):
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        checkpoint = torch.load(filepath, map_location=device)
        wrapper = cls(
            input_size=checkpoint["input_size"],
            hidden_size=checkpoint["hidden_size"],
            num_layers=checkpoint["num_layers"],
            dropout=checkpoint["dropout"],
            device=device
        )
        wrapper.model.load_state_dict(checkpoint["state_dict"])
        wrapper.model.eval()
        return wrapper
