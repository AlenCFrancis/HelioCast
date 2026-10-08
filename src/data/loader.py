"""
Data loader module for Solar Flare Prediction.
Loads data based on YAML configuration.
"""

import os
import yaml
import pandas as pd
from typing import Dict, Any, Tuple

def load_config(config_path: str = "configs/config.yaml") -> Dict[str, Any]:
    """Load configuration YAML file."""
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found at: {config_path}")
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config

def load_solar_data(data_path: str = None, config: Dict[str, Any] = None) -> pd.DataFrame:
    """
    Load raw or processed solar observation CSV dataset.
    Parses timestamp column and ensures chronological sorting.
    """
    if config is None:
        config = load_config()
    
    if data_path is None:
        data_path = config["data"]["processed_path"]

    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Solar dataset CSV not found at: {data_path}")

    df = pd.read_csv(data_path)
    timestamp_col = config["data"].get("timestamp_col", "timestamp")
    
    if timestamp_col not in df.columns:
        raise KeyError(f"Timestamp column '{timestamp_col}' not found in dataset columns: {list(df.columns)}")

    df[timestamp_col] = pd.to_datetime(df[timestamp_col])
    df = df.sort_values(by=timestamp_col).reset_index(drop=True)
    return df
