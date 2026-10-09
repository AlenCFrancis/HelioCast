"""
Sequence Service Module for Solar Flare Forecasting.
Provides 24-hour sliding window construction, temporal sequencing, and shape formatting
for recurrent (LSTM) and classical ML models without lookahead leakage.
"""

from typing import List, Tuple, Dict, Any, Optional
import numpy as np
import pandas as pd


class SequenceService:
    """Handles temporal sequence windowing with strict zero-leakage boundaries."""

    def __init__(self, sequence_length: int = 24):
        self.sequence_length = sequence_length

    def extract_window_df(
        self,
        df: pd.DataFrame,
        target_index: int
    ) -> pd.DataFrame:
        """
        Extract a continuous historical window of exactly sequence_length rows
        ending at target_index (inclusive).
        
        Indices: [target_index - sequence_length + 1 : target_index + 1]
        """
        if target_index < self.sequence_length - 1:
            raise ValueError(
                f"Target index {target_index} does not have sufficient history. "
                f"Minimum required index is {self.sequence_length - 1}."
            )
        
        start_idx = target_index - self.sequence_length + 1
        end_idx = target_index + 1
        window_df = df.iloc[start_idx:end_idx].copy().reset_index(drop=True)
        return window_df

    def format_inputs(
        self,
        X_scaled_window: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Format a 24-step scaled 2D array of shape (24, num_features) into:
        1. 3D sequence array: (1, 24, num_features) for LSTM / deep learning.
        2. 2D snapshot array: (1, num_features) for classical models using T0 observation.
        """
        if X_scaled_window.shape[0] != self.sequence_length:
            raise ValueError(
                f"Window length must be exactly {self.sequence_length}, "
                f"got {X_scaled_window.shape[0]}."
            )

        # 3D tensor for LSTM: [batch=1, seq_len=24, features]
        X_seq_3d = np.expand_dims(X_scaled_window, axis=0).astype(np.float32)

        # 2D vector for tabular models: latest observation at T0 (last row of window)
        X_snapshot_2d = X_scaled_window[-1:, :].astype(np.float32)

        return X_seq_3d, X_snapshot_2d

    @staticmethod
    def create_sliding_sequences(
        X_scaled: np.ndarray,
        y: np.ndarray,
        seq_length: int = 24
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Construct all sliding window temporal sequences from a single split array.
        Enforces sequence boundaries strictly inside the split.
        """
        num_timestamps, num_features = X_scaled.shape
        if num_timestamps <= seq_length:
            raise ValueError(
                f"Data length ({num_timestamps}) must be greater than sequence length ({seq_length})."
            )

        X_seq_list = []
        y_seq_list = []

        for i in range(seq_length, num_timestamps + 1):
            X_seq_list.append(X_scaled[i - seq_length : i, :])
            y_seq_list.append(y[i - 1])

        return np.array(X_seq_list, dtype=np.float32), np.array(y_seq_list, dtype=np.float32)
