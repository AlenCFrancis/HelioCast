"""
1D Convolutional Neural Network Architecture Module for Solar Flare Forecasting.
Temporal CNN operating on the same (N, seq_length, num_features) sequences as the LSTM.
"""

import os
import torch
import torch.nn as nn
import numpy as np


class SolarCNN1D(nn.Module):
    """
    Temporal 1D CNN for binary solar flare classification.

    Input shape:  (batch_size, sequence_length, num_features)  — same as LSTM
    Internally permuted to (batch_size, num_features, sequence_length) for Conv1d.
    Output shape: (batch_size,)  — raw logits
    """

    def __init__(
        self,
        input_size: int,
        channels: list = None,
        kernel_size: int = 3,
        dropout: float = 0.2
    ):
        super(SolarCNN1D, self).__init__()
        self.input_size = input_size
        self.channels = channels if channels is not None else [32, 64]
        self.kernel_size = kernel_size
        self.dropout_rate = dropout

        # Conv block 1: input_size channels -> channels[0]
        self.conv1 = nn.Conv1d(
            in_channels=input_size,
            out_channels=self.channels[0],
            kernel_size=kernel_size,
            padding=kernel_size // 2
        )
        self.relu1 = nn.ReLU()

        # Conv block 2: channels[0] -> channels[1]
        self.conv2 = nn.Conv1d(
            in_channels=self.channels[0],
            out_channels=self.channels[1],
            kernel_size=kernel_size,
            padding=kernel_size // 2
        )
        self.relu2 = nn.ReLU()

        # Global pooling -> flatten -> classify
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(self.channels[1], 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (batch_size, sequence_length, input_size)
        # Permute to (batch_size, input_size, sequence_length) for Conv1d
        x = x.permute(0, 2, 1)

        x = self.relu1(self.conv1(x))   # (batch, channels[0], seq_len)
        x = self.relu2(self.conv2(x))   # (batch, channels[1], seq_len)
        x = self.pool(x).squeeze(-1)    # (batch, channels[1])
        x = self.dropout(x)
        logits = self.fc(x).squeeze(-1) # (batch,)
        return logits


class SolarCNN1DWrapper:
    """Wrapper class providing sklearn-like API (predict_proba, predict, save, load)."""

    def __init__(
        self,
        input_size: int = None,
        channels: list = None,
        kernel_size: int = 3,
        dropout: float = 0.2,
        device: torch.device = None
    ):
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device

        self.input_size = input_size
        self.channels = channels if channels is not None else [32, 64]
        self.kernel_size = kernel_size
        self.dropout = dropout

        if input_size is not None:
            self.model = SolarCNN1D(
                input_size=input_size,
                channels=self.channels,
                kernel_size=kernel_size,
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

    def save(self, filepath: str = "models/cnn_1d.pt"):
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        checkpoint = {
            "state_dict": self.model.state_dict(),
            "input_size": self.input_size,
            "channels": self.channels,
            "kernel_size": self.kernel_size,
            "dropout": self.dropout
        }
        torch.save(checkpoint, filepath)
        print(f"[CNN 1D] PyTorch model state saved to: {filepath}")

    @classmethod
    def load(cls, filepath: str = "models/cnn_1d.pt", device: torch.device = None):
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        checkpoint = torch.load(filepath, map_location=device)
        wrapper = cls(
            input_size=checkpoint["input_size"],
            channels=checkpoint["channels"],
            kernel_size=checkpoint["kernel_size"],
            dropout=checkpoint["dropout"],
            device=device
        )
        wrapper.model.load_state_dict(checkpoint["state_dict"])
        wrapper.model.eval()
        return wrapper
