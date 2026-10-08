"""
Sequence Builder Module for LSTM Model.
Generates 3D tensors [samples, sequence_length, features] from scaled 2D arrays.
Enforces sequence boundaries strictly inside each train/val/test split to prevent leakage.
"""

import numpy as np
from typing import Tuple

def create_sequences(
    X_scaled: np.ndarray,
    y: np.ndarray,
    seq_length: int = 24
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Construct sliding window temporal sequences for deep learning models (LSTM).

    Parameters
    ----------
    X_scaled : np.ndarray of shape (N, num_features)
        Scaled 2D feature matrix for a SINGLE split (train, val, or test).
    y : np.ndarray of shape (N,)
        1D target vector corresponding to each observation timestamp.
    seq_length : int
        Number of historical observation timesteps to include in each sequence (default: 24).

    Returns
    -------
    X_seq : np.ndarray of shape (num_samples, seq_length, num_features)
    y_seq : np.ndarray of shape (num_samples,)
        Target label corresponding to the final timestep of each sequence window.
    """
    num_timestamps, num_features = X_scaled.shape

    if num_timestamps <= seq_length:
        raise ValueError(
            f"Dataset length ({num_timestamps}) must be strictly greater than sequence length ({seq_length})."
        )

    X_seq_list = []
    y_seq_list = []

    for i in range(seq_length, num_timestamps + 1):
        X_window = X_scaled[i - seq_length : i, :]  # shape: (seq_length, num_features)
        y_target = y[i - 1]                        # target for prediction horizon at end of window

        X_seq_list.append(X_window)
        y_seq_list.append(y_target)

    X_seq = np.array(X_seq_list, dtype=np.float32)
    y_seq = np.array(y_seq_list, dtype=np.float32)

    return X_seq, y_seq
