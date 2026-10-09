"""
Inference Service Module for Solar Flare Forecasting.
Handles model loading, preprocessor artifact management, zero-leakage scaling,
frozen validation thresholding, and real-time prediction execution.
"""

import os
import sys
import math
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import joblib
import numpy as np
import pandas as pd
import torch

from app.core.config import settings
from app.schemas.prediction import (
    PredictionResponse,
    TelemetryWindowResponse,
    AvailableTimestampsResponse,
    ModelMetrics,
    MetricsSummaryResponse,
    DatasetSummaryResponse,
    ConfusionMatrix
)
from app.services.sequence_service import SequenceService

# Ensure project root is in sys.path so src modules can be imported
if str(settings.PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(settings.PROJECT_ROOT))

logger = logging.getLogger(__name__)

MODEL_DISPLAY_NAMES = {
    "logistic_regression": "Logistic Regression",
    "random_forest": "Random Forest",
    "xgboost": "XGBoost",
    "lstm": "LSTM (Recurrent Neural Network)",
    "cnn_1d": "1D Temporal CNN"
}


class InferenceService:
    """Production inference engine managing pre-trained models and scaler execution."""

    def __init__(self):
        self.sequence_service = SequenceService(sequence_length=settings.HISTORY_HOURS)
        self.df: Optional[pd.DataFrame] = None
        self.preprocessor: Any = None
        self.feature_names: List[str] = []
        self.impute_values: Dict[str, float] = {}
        self.scaler: Any = None
        self.timestamp_col: str = "timestamp"
        self.target_col: str = "target_m_plus_24h"
        self.models: Dict[str, Any] = {}
        self.metrics_summary: Dict[str, Dict[str, Any]] = {}
        self.is_loaded: bool = False

    def load_artifacts(self) -> None:
        """Load scaler, dataset, models, and evaluation metrics from disk."""
        if self.is_loaded:
            return

        logger.info("Initializing inference service artifacts from %s", settings.PROJECT_ROOT)

        # 1. Load Preprocessor and Scaler
        scaler_path = settings.MODELS_DIR / "scaler.joblib"
        if not scaler_path.exists():
            raise FileNotFoundError(f"Scaler artifact not found at {scaler_path}")

        metadata = joblib.load(scaler_path)
        self.scaler = metadata["scaler"]
        self.feature_names = metadata["feature_names"]
        self.impute_values = metadata.get("impute_values", {})
        self.timestamp_col = metadata.get("timestamp_col", "timestamp")
        self.target_col = metadata.get("target_col", "target_m_plus_24h")

        # 2. Load Processed Solar Observations Dataset
        if not settings.DATA_PATH.exists():
            raise FileNotFoundError(f"Dataset not found at {settings.DATA_PATH}")

        df = pd.read_csv(settings.DATA_PATH)
        df[self.timestamp_col] = pd.to_datetime(df[self.timestamp_col])
        df = df.sort_values(by=self.timestamp_col).reset_index(drop=True)
        self.df = df

        # 3. Load Models
        self.models = {}

        # Logistic Regression
        lr_path = settings.MODELS_DIR / "logistic_regression.joblib"
        if lr_path.exists():
            try:
                from src.models.logistic_regression import SolarLogisticRegression
                self.models["logistic_regression"] = SolarLogisticRegression.load(str(lr_path))
            except Exception as e:
                logger.warning("Falling back to direct joblib for Logistic Regression: %s", e)
                lr_obj = joblib.load(lr_path)
                self.models["logistic_regression"] = lr_obj

        # Random Forest
        rf_path = settings.MODELS_DIR / "random_forest.joblib"
        if rf_path.exists():
            try:
                from src.models.random_forest import SolarRandomForest
                self.models["random_forest"] = SolarRandomForest.load(str(rf_path))
            except Exception as e:
                logger.warning("Falling back to direct joblib for Random Forest: %s", e)
                self.models["random_forest"] = joblib.load(rf_path)

        # XGBoost
        xgb_path = settings.MODELS_DIR / "xgboost.joblib"
        if xgb_path.exists():
            try:
                from src.models.xgboost_model import SolarXGBoost
                self.models["xgboost"] = SolarXGBoost.load(str(xgb_path))
            except Exception as e:
                logger.warning("Falling back to direct joblib for XGBoost: %s", e)
                self.models["xgboost"] = joblib.load(xgb_path)

        # LSTM (PyTorch)
        lstm_path = settings.MODELS_DIR / "lstm.pt"
        if lstm_path.exists():
            try:
                from src.models.lstm import SolarLSTMWrapper
                self.models["lstm"] = SolarLSTMWrapper.load(str(lstm_path), device=torch.device("cpu"))
            except Exception as e:
                logger.error("Failed to load LSTM model: %s", e)

        # 1D CNN (if present)
        cnn_path = settings.MODELS_DIR / "cnn_1d.pt"
        if cnn_path.exists():
            try:
                from src.models.cnn_1d import SolarCNN1DWrapper
                self.models["cnn_1d"] = SolarCNN1DWrapper.load(str(cnn_path), device=torch.device("cpu"))
            except Exception as e:
                logger.info("1D CNN artifact present but wrapper not initialized: %s", e)

        # 4. Load Metrics Reports
        self.metrics_summary = {}
        if settings.METRICS_DIR.exists():
            for f in os.listdir(settings.METRICS_DIR):
                if f.endswith(".json"):
                    model_key = f.replace(".json", "")
                    with open(settings.METRICS_DIR / f, "r", encoding="utf-8") as jf:
                        self.metrics_summary[model_key] = json.load(jf)

        self.is_loaded = True
        logger.info(
            "Inference service ready: %d models loaded (%s), %d observations",
            len(self.models),
            list(self.models.keys()),
            len(self.df)
        )

    def get_available_timestamps(self) -> AvailableTimestampsResponse:
        """Return list of valid observation timestamps with 24h history."""
        self.load_artifacts()
        all_timestamps = self.df[self.timestamp_col].dt.strftime("%Y-%m-%d %H:%M:%S").tolist()
        valid_timestamps = all_timestamps[self.sequence_service.sequence_length - 1:]
        return AvailableTimestampsResponse(
            timestamps=valid_timestamps,
            total_count=len(valid_timestamps),
            earliest=valid_timestamps[0],
            latest=valid_timestamps[-1]
        )

    def get_dataset_summary(self) -> DatasetSummaryResponse:
        """Provide telemetry dataset overview and metadata."""
        self.load_artifacts()
        min_ts = self.df[self.timestamp_col].min().strftime("%Y-%m-%d %H:%M:%S")
        max_ts = self.df[self.timestamp_col].max().strftime("%Y-%m-%d %H:%M:%S")
        return DatasetSummaryResponse(
            total_observations=len(self.df),
            feature_count=len(self.feature_names),
            feature_names=self.feature_names,
            min_timestamp=min_ts,
            max_timestamp=max_ts,
            cadence_hours=1.0,
            active_models=list(self.models.keys())
        )

    def get_metrics_matrix(self) -> MetricsSummaryResponse:
        """Compile tabular comparison of offline test set metrics."""
        self.load_artifacts()
        model_list = []
        for key, metrics in self.metrics_summary.items():
            cm_data = metrics.get("confusion_matrix")
            cm_obj = None
            if cm_data:
                cm_obj = ConfusionMatrix(
                    tn=cm_data.get("tn", 0),
                    fp=cm_data.get("fp", 0),
                    fn=cm_data.get("fn", 0),
                    tp=cm_data.get("tp", 0)
                )

            model_list.append(ModelMetrics(
                model_key=key,
                model_name=MODEL_DISPLAY_NAMES.get(key, key.upper()),
                threshold=round(float(metrics.get("threshold", 0.5)), 4),
                roc_auc=round(float(metrics.get("roc_auc", 0.0)), 4),
                pr_auc=round(float(metrics.get("pr_auc", 0.0)), 4),
                precision=round(float(metrics.get("precision", 0.0)), 4),
                recall=round(float(metrics.get("recall", 0.0)), 4),
                f1_score=round(float(metrics.get("f1_score", 0.0)), 4),
                tss=round(float(metrics.get("tss", 0.0)), 4),
                far=round(float(metrics.get("far", 0.0)), 4),
                brier_score=round(float(metrics.get("brier_score", 0.0)), 4) if "brier_score" in metrics else None,
                confusion_matrix=cm_obj
            ))
        return MetricsSummaryResponse(models=model_list)

    def transform_dataframe(self, window_df: pd.DataFrame) -> np.ndarray:
        """
        Impute missing features and scale using the training-fitted StandardScaler.
        Guarantees strict zero-leakage inference.
        """
        temp_df = window_df.copy()
        for col in self.feature_names:
            if col in temp_df.columns:
                temp_df[col] = temp_df[col].fillna(self.impute_values.get(col, 0.0))
            else:
                temp_df[col] = self.impute_values.get(col, 0.0)
        X_raw = temp_df[self.feature_names].values
        return self.scaler.transform(X_raw)

    def get_telemetry_window(self, timestamp_str: Optional[str] = None) -> TelemetryWindowResponse:
        """Extract 24-hour historical telemetry sequence for charting."""
        self.load_artifacts()
        
        if not timestamp_str:
            target_idx = len(self.df) - 1
        else:
            match = self.df[self.df[self.timestamp_col].dt.strftime("%Y-%m-%d %H:%M:%S") == timestamp_str]
            if match.empty:
                raise ValueError(f"Requested timestamp '{timestamp_str}' not found in observation history.")
            target_idx = match.index[0]

        window_df = self.sequence_service.extract_window_df(self.df, target_idx)
        window_timestamps = window_df[self.timestamp_col].dt.strftime("%Y-%m-%d %H:%M:%S").tolist()

        features_data = {}
        for col in self.feature_names:
            features_data[col] = [round(float(val), 8) for val in window_df[col].tolist()]

        ground_truth = None
        if self.target_col in window_df.columns:
            ground_truth = int(window_df[self.target_col].iloc[-1])

        return TelemetryWindowResponse(
            target_timestamp=window_timestamps[-1],
            timestamps=window_timestamps,
            features=features_data,
            ground_truth=ground_truth
        )

    def predict(
        self,
        model_name: str,
        timestamp_str: Optional[str] = None,
        custom_window: Optional[List[Dict[str, float]]] = None
    ) -> PredictionResponse:
        """
        Run inference using the selected model architecture on a 24-hour observation window.
        Uses frozen validation threshold to determine flare classification.
        """
        self.load_artifacts()

        normalized_model_name = model_name.lower().replace("-", "_").replace(" ", "_")
        if normalized_model_name not in self.models:
            valid_keys = list(self.models.keys())
            raise ValueError(
                f"Model '{model_name}' not available. Available models: {valid_keys}"
            )

        model = self.models[normalized_model_name]
        
        # 1. Obtain 24-step DataFrame
        if custom_window is not None:
            if len(custom_window) != self.sequence_service.sequence_length:
                raise ValueError(
                    f"Custom window must contain exactly {self.sequence_service.sequence_length} observations."
                )
            window_df = pd.DataFrame(custom_window)
            target_timestamp = "CUSTOM_INJECTION_T0"
            window_timestamps = [f"T-{self.sequence_service.sequence_length - 1 - i}h" for i in range(self.sequence_service.sequence_length)]
            ground_truth = None
        else:
            if not timestamp_str:
                target_idx = len(self.df) - 1
            else:
                match = self.df[self.df[self.timestamp_col].dt.strftime("%Y-%m-%d %H:%M:%S") == timestamp_str]
                if match.empty:
                    raise ValueError(f"Requested timestamp '{timestamp_str}' not found in observation history.")
                target_idx = match.index[0]

            window_df = self.sequence_service.extract_window_df(self.df, target_idx)
            window_timestamps = window_df[self.timestamp_col].dt.strftime("%Y-%m-%d %H:%M:%S").tolist()
            target_timestamp = window_timestamps[-1]
            ground_truth = int(window_df[self.target_col].iloc[-1]) if self.target_col in window_df.columns else None

        # 2. Preprocess & Scale
        X_scaled_window = self.transform_dataframe(window_df)

        # 3. Format inputs for recurrent vs classical architecture
        X_seq_3d, X_snapshot_2d = self.sequence_service.format_inputs(X_scaled_window)

        # 4. Execute Model Inference
        if normalized_model_name in ("lstm", "cnn_1d"):
            raw_prob = float(model.predict_proba(X_seq_3d)[0])
        else:
            # Check if model object is wrapper or direct scikit-learn/xgboost estimator
            if hasattr(model, "predict_proba"):
                res = model.predict_proba(X_snapshot_2d)
                if len(res.shape) > 1 and res.shape[1] > 1:
                    raw_prob = float(res[0, 1])
                else:
                    raw_prob = float(res[0])
            else:
                raise RuntimeError(f"Loaded model '{model_name}' does not implement predict_proba.")

        # Bound probability to [0.0, 1.0]
        raw_prob = max(0.0, min(1.0, raw_prob))
        prob_percent = raw_prob * 100.0

        # High-precision formatted string
        if prob_percent == 0.0:
            formatted_prob = "0.00%"
        elif prob_percent >= 0.01:
            formatted_prob = f"{prob_percent:.2f}%"
        else:
            digits = int(-math.floor(math.log10(prob_percent))) + 1
            formatted_prob = f"{prob_percent:.{digits}f}%"

        # 5. Retrieve Frozen Validation Threshold
        model_metrics = self.metrics_summary.get(normalized_model_name, {})
        frozen_threshold = float(model_metrics.get("threshold", 0.5))

        # 6. Evaluate Classification & Risk
        is_flare = bool(raw_prob >= frozen_threshold)

        if raw_prob >= frozen_threshold:
            risk_level = "CRITICAL"
        elif raw_prob >= (frozen_threshold * 0.7):
            risk_level = "MODERATE"
        else:
            risk_level = "NORMAL"

        ground_truth_label = None
        if ground_truth is not None:
            ground_truth_label = "DETECTED" if ground_truth == 1 else "NO FLARE"

        return PredictionResponse(
            model_name=normalized_model_name,
            model_display_name=MODEL_DISPLAY_NAMES.get(normalized_model_name, normalized_model_name.upper()),
            target_timestamp=target_timestamp,
            probability=round(raw_prob, 6),
            probability_percent=round(prob_percent, 4),
            probability_formatted=formatted_prob,
            threshold=round(frozen_threshold, 4),
            is_flare=is_flare,
            risk_level=risk_level,
            ground_truth=ground_truth,
            ground_truth_label=ground_truth_label,
            sequence_length=self.sequence_service.sequence_length,
            window_timestamps=window_timestamps
        )


# Global Singleton Service
inference_service = InferenceService()
