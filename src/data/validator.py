"""
Dataset Validator Module for Solar Flare Prediction.
Validates solar dataset schema, timestamps, missing values, duplicates, class distribution,
and observation cadence.
"""

import sys
import pandas as pd
import numpy as np
from typing import Dict, Any
from src.data.loader import load_config, load_solar_data

def validate_solar_dataset(df: pd.DataFrame = None, config: Dict[str, Any] = None) -> Dict[str, Any]:
    """
    Perform dataset inspection and validation checks.
    """
    if config is None:
        config = load_config()
    
    if df is None:
        df = load_solar_data(config=config)

    timestamp_col = config["data"].get("timestamp_col", "timestamp")
    target_col = config["data"].get("target_col", "target_m_plus_24h")

    # 1. Dataset Shape
    shape = df.shape

    # 2. Column names & data types
    dtypes = {col: str(dtype) for col, dtype in df.dtypes.items()}
    solar_features = [col for col in df.columns if col not in [timestamp_col, target_col]]

    # 3. Timestamp inspection & Cadence
    time_series = df[timestamp_col]
    min_time = time_series.min()
    max_time = time_series.max()

    time_diffs = time_series.diff().dropna()
    mode_cadence = time_diffs.mode()[0] if len(time_diffs) > 0 else None
    
    # Estimate cadence in minutes/hours
    cadence_hours = mode_cadence.total_seconds() / 3600.0 if mode_cadence is not None else 1.0

    # 4. Missing values
    missing_counts = df.isnull().sum().to_dict()

    # 5. Duplicate timestamps
    duplicate_timestamps = int(time_series.duplicated().sum())

    # 6. Target distribution
    if target_col in df.columns:
        target_counts = df[target_col].value_counts().to_dict()
        target_ratio = float(df[target_col].mean())
    else:
        target_counts = {}
        target_ratio = 0.0

    # Summary statistics for features
    feature_stats = df[solar_features].describe().to_dict()

    report = {
        "dataset_shape": shape,
        "columns": list(df.columns),
        "solar_features": solar_features,
        "timestamp_col": timestamp_col,
        "target_col": target_col,
        "data_types": dtypes,
        "min_timestamp": str(min_time),
        "max_timestamp": str(max_time),
        "observation_cadence": str(mode_cadence),
        "cadence_hours": cadence_hours,
        "missing_values": missing_counts,
        "duplicate_timestamps": duplicate_timestamps,
        "target_counts": target_counts,
        "target_positive_ratio": target_ratio,
        "feature_stats": feature_stats
    }

    return report

def print_validation_report(report: Dict[str, Any]):
    print("=" * 60)
    print(" SOLAR FLARE DATASET VALIDATION REPORT")
    print("=" * 60)
    print(f"Dataset Shape        : {report['dataset_shape']}")
    print(f"Timestamp Column     : {report['timestamp_col']}")
    print(f"Target Column        : {report['target_col']}")
    print(f"Solar Features ({len(report['solar_features'])}) : {report['solar_features']}")
    print(f"Timestamp Range      : {report['min_timestamp']} to {report['max_timestamp']}")
    print(f"Observation Cadence  : {report['observation_cadence']} (~{report['cadence_hours']:.2f} hours)")
    print(f"Duplicate Timestamps : {report['duplicate_timestamps']}")
    print(f"Missing Values Total : {sum(report['missing_values'].values())}")
    for col, count in report['missing_values'].items():
        if count > 0:
            print(f"  - {col}: {count} missing")
    print("-" * 60)
    print("TARGET CLASS DISTRIBUTION (M+ 24h Flare):")
    for k, v in report['target_counts'].items():
        print(f"  Class {k}: {v} samples ({v / report['dataset_shape'][0] * 100:.2f}%)")
    print(f"Positive Ratio       : {report['target_positive_ratio'] * 100:.2f}%")
    print("=" * 60)

if __name__ == "__main__":
    try:
        report = validate_solar_dataset()
        print_validation_report(report)
    except Exception as e:
        print(f"Validation failed: {e}", file=sys.stderr)
        sys.exit(1)
