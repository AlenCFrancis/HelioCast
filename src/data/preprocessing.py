"""
Preprocessing Module for Solar Flare Prediction.
Handles chronological splitting, strict no-leakage scaling, missing value imputation,
and preprocessing artifact persistence.
"""

import os
import joblib
import pandas as pd
import numpy as np
from typing import Dict, Any, Tuple
from sklearn.preprocessing import StandardScaler
from src.data.loader import load_config, load_solar_data

class SolarDataPreprocessor:
    def __init__(self, config: Dict[str, Any] = None):
        if config is None:
            config = load_config()
        self.config = config
        self.timestamp_col = config["data"].get("timestamp_col", "timestamp")
        self.target_col = config["data"].get("target_col", "target_m_plus_24h")
        self.train_ratio = config["data"]["splits"].get("train_ratio", 0.70)
        self.val_ratio = config["data"]["splits"].get("val_ratio", 0.15)
        self.test_ratio = config["data"]["splits"].get("test_ratio", 0.15)
        self.scaler = StandardScaler()
        self.feature_names = []
        self.impute_values = {}
        self.is_fitted = False

    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Parse timestamps, sort chronologically, drop duplicate timestamps."""
        df = df.copy()
        df[self.timestamp_col] = pd.to_datetime(df[self.timestamp_col])
        df = df.sort_values(by=self.timestamp_col).reset_index(drop=True)
        # Drop duplicates based on timestamp
        df = df.drop_duplicates(subset=[self.timestamp_col], keep="first").reset_index(drop=True)
        return df

    def split_chronologically(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Chronological split without random shuffling.
        Train -> Earlier period
        Val   -> Middle period
        Test  -> Most recent period
        """
        n = len(df)
        train_end = int(n * self.train_ratio)
        val_end = int(n * (self.train_ratio + self.val_ratio))

        train_df = df.iloc[:train_end].copy().reset_index(drop=True)
        val_df = df.iloc[train_end:val_end].copy().reset_index(drop=True)
        test_df = df.iloc[val_end:].copy().reset_index(drop=True)

        return train_df, val_df, test_df

    def fit_transform(
        self, df: pd.DataFrame
    ) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray], Dict[str, pd.DataFrame]]:
        """
        Clean data, split chronologically, fit scaler and imputations ONLY on train,
        and transform train, val, and test.
        """
        cleaned_df = self.clean_data(df)
        train_df, val_df, test_df = self.split_chronologically(cleaned_df)

        # Identify feature columns
        self.feature_names = [
            c for c in cleaned_df.columns if c not in [self.timestamp_col, self.target_col]
        ]

        # Calculate imputation values (median) from TRAIN set only
        for col in self.feature_names:
            median_val = train_df[col].median()
            if pd.isna(median_val):
                median_val = 0.0
            self.impute_values[col] = median_val
            train_df[col] = train_df[col].fillna(median_val)
            val_df[col] = val_df[col].fillna(median_val)
            test_df[col] = test_df[col].fillna(median_val)

        # Fit Scaler strictly on Train feature data
        X_train_raw = train_df[self.feature_names].values
        self.scaler.fit(X_train_raw)
        self.is_fitted = True

        # Transform features
        X_train = self.scaler.transform(X_train_raw)
        X_val = self.scaler.transform(val_df[self.feature_names].values)
        X_test = self.scaler.transform(test_df[self.feature_names].values)

        y_train = train_df[self.target_col].values
        y_val = val_df[self.target_col].values
        y_test = test_df[self.target_col].values

        X_dict = {"train": X_train, "val": X_val, "test": X_test}
        y_dict = {"train": y_train, "val": y_val, "test": y_test}
        df_dict = {"train": train_df, "val": val_df, "test": test_df}

        return X_dict, y_dict, df_dict

    def transform_df(self, df: pd.DataFrame) -> np.ndarray:
        """Transform single dataframe using fitted scaler and imputations."""
        if not self.is_fitted:
            raise ValueError("SolarDataPreprocessor must be fitted before calling transform_df.")
        df = df.copy()
        for col in self.feature_names:
            if col in df.columns:
                df[col] = df[col].fillna(self.impute_values.get(col, 0.0))
            else:
                df[col] = self.impute_values.get(col, 0.0)
        X_raw = df[self.feature_names].values
        return self.scaler.transform(X_raw)

    def save(self, model_dir: str = "models"):
        """Save scaler and preprocessor metadata."""
        os.makedirs(model_dir, exist_ok=True)
        scaler_path = os.path.join(model_dir, "scaler.joblib")
        metadata = {
            "scaler": self.scaler,
            "feature_names": self.feature_names,
            "impute_values": self.impute_values,
            "timestamp_col": self.timestamp_col,
            "target_col": self.target_col
        }
        joblib.dump(metadata, scaler_path)
        print(f"[Preprocessor] Scaler and metadata saved to: {scaler_path}")

    @classmethod
    def load(cls, scaler_path: str = "models/scaler.joblib", config: Dict[str, Any] = None):
        """Load fitted preprocessor metadata."""
        preprocessor = cls(config=config)
        metadata = joblib.load(scaler_path)
        preprocessor.scaler = metadata["scaler"]
        preprocessor.feature_names = metadata["feature_names"]
        preprocessor.impute_values = metadata["impute_values"]
        preprocessor.timestamp_col = metadata["timestamp_col"]
        preprocessor.target_col = metadata["target_col"]
        preprocessor.is_fitted = True
        return preprocessor
