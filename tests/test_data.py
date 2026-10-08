"""
Unit tests for data loading, validation, and generation.
"""

import pytest
import os
import pandas as pd
from src.data.loader import load_config, load_solar_data
from src.data.generate_dataset import generate_solar_dataset
from src.data.validator import validate_solar_dataset

@pytest.fixture
def sample_dataset_path(tmp_path):
    csv_path = os.path.join(tmp_path, "sample_solar.csv")
    generate_solar_dataset(num_hours=200, random_seed=42, output_path=csv_path)
    return csv_path

def test_generate_solar_dataset(sample_dataset_path):
    assert os.path.exists(sample_dataset_path)
    df = pd.read_csv(sample_dataset_path)
    assert len(df) > 0
    assert "timestamp" in df.columns
    assert "target_m_plus_24h" in df.columns
    assert df["target_m_plus_24h"].isin([0, 1]).all()

def test_load_solar_data(sample_dataset_path):
    df = load_solar_data(data_path=sample_dataset_path)
    assert pd.api.types.is_datetime64_any_dtype(df["timestamp"])
    # Ensure chronological order
    assert (df["timestamp"].diff().dropna() >= pd.Timedelta(0)).all()

def test_validate_solar_dataset(sample_dataset_path):
    df = load_solar_data(data_path=sample_dataset_path)
    report = validate_solar_dataset(df=df)
    assert report["dataset_shape"][0] > 0
    assert report["duplicate_timestamps"] == 0
    assert "xrsb_flux" in report["solar_features"]
    assert report["target_positive_ratio"] >= 0.0
