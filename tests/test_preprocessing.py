"""
Unit tests for data preprocessing, chronological splitting, and sequence generation.
"""

import pytest
import os
import numpy as np
import pandas as pd
from src.data.generate_dataset import generate_solar_dataset
from src.data.preprocessing import SolarDataPreprocessor
from src.data.sequence_builder import create_sequences

@pytest.fixture
def sample_df(tmp_path):
    csv_path = os.path.join(tmp_path, "sample_solar.csv")
    generate_solar_dataset(num_hours=300, random_seed=42, output_path=csv_path)
    df = pd.read_csv(csv_path)
    return df

def test_chronological_split_no_leakage(sample_df):
    preprocessor = SolarDataPreprocessor()
    train_df, val_df, test_df = preprocessor.split_chronologically(sample_df)

    n_total = len(sample_df)
    assert len(train_df) + len(val_df) + len(test_df) == n_total

    # Verify timestamp monotonicity across splits
    train_max_time = pd.to_datetime(train_df["timestamp"]).max()
    val_min_time = pd.to_datetime(val_df["timestamp"]).min()
    val_max_time = pd.to_datetime(val_df["timestamp"]).max()
    test_min_time = pd.to_datetime(test_df["timestamp"]).min()

    assert train_max_time < val_min_time
    assert val_max_time < test_min_time

def test_preprocessor_fit_transform(sample_df):
    preprocessor = SolarDataPreprocessor()
    X_dict, y_dict, _ = preprocessor.fit_transform(sample_df)

    X_train = X_dict["train"]
    X_val = X_dict["val"]
    X_test = X_dict["test"]

    # Verify scaling: train mean ~ 0, std ~ 1
    train_means = np.mean(X_train, axis=0)
    train_stds = np.std(X_train, axis=0)

    np.testing.assert_allclose(train_means, 0, atol=1e-5)
    np.testing.assert_allclose(train_stds, 1, atol=1e-5)

    assert X_train.shape[1] == len(preprocessor.feature_names)
    assert len(y_dict["train"]) == len(X_train)

def test_create_sequences():
    N, F = 100, 5
    X_scaled = np.random.randn(N, F).astype(np.float32)
    y = np.random.randint(0, 2, size=N).astype(np.float32)

    seq_len = 24
    X_seq, y_seq = create_sequences(X_scaled, y, seq_length=seq_len)

    expected_samples = N - seq_len + 1
    assert X_seq.shape == (expected_samples, seq_len, F)
    assert y_seq.shape == (expected_samples,)
    # Verify alignment of targets
    assert y_seq[0] == y[seq_len - 1]
