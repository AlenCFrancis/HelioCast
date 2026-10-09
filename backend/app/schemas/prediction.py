from typing import Dict, List, Optional, Any
from pydantic import BaseModel, Field


class PredictionRequest(BaseModel):
    model_name: str = Field(
        default="logistic_regression",
        description="Model architecture identifier (e.g. logistic_regression, random_forest, xgboost, lstm, cnn_1d)"
    )
    timestamp: Optional[str] = Field(
        default=None,
        description="Target timestamp T0 for historical evaluation (e.g. '2026-09-30 00:00:00'). Defaults to latest observation if None."
    )
    custom_window: Optional[List[Dict[str, float]]] = Field(
        default=None,
        description="Optional 24-step telemetry array of magnetogram parameters for synthetic or live telemetry injection."
    )


class PredictionResponse(BaseModel):
    model_name: str
    model_display_name: str
    target_timestamp: str
    probability: float
    probability_percent: float
    probability_formatted: str
    threshold: float
    is_flare: bool
    risk_level: str
    ground_truth: Optional[int] = None
    ground_truth_label: Optional[str] = None
    sequence_length: int = 24
    window_timestamps: List[str]


class TelemetryWindowResponse(BaseModel):
    target_timestamp: str
    timestamps: List[str]
    features: Dict[str, List[float]]
    ground_truth: Optional[int] = None


class AvailableTimestampsResponse(BaseModel):
    timestamps: List[str]
    total_count: int
    earliest: str
    latest: str


class ConfusionMatrix(BaseModel):
    tn: int
    fp: int
    fn: int
    tp: int


class ModelMetrics(BaseModel):
    model_key: str
    model_name: str
    threshold: float
    roc_auc: float
    pr_auc: float
    precision: float
    recall: float
    f1_score: float
    tss: float
    far: float
    brier_score: Optional[float] = None
    confusion_matrix: Optional[ConfusionMatrix] = None


class MetricsSummaryResponse(BaseModel):
    models: List[ModelMetrics]


class DatasetSummaryResponse(BaseModel):
    total_observations: int
    feature_count: int
    feature_names: List[str]
    min_timestamp: str
    max_timestamp: str
    cadence_hours: float = 1.0
    active_models: List[str]


class HealthResponse(BaseModel):
    status: str
    version: str
    models_loaded: List[str]
    dataset_loaded: bool
