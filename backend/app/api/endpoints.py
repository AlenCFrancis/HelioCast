"""
API Endpoints Router for Solar Flare Forecasting System.
Provides prediction, telemetry sequence, dataset inspection, and metric evaluation routes.
"""

from typing import Optional, List, Dict, Any
from fastapi import APIRouter, HTTPException, Query, status

from app.schemas.prediction import (
    PredictionRequest,
    PredictionResponse,
    TelemetryWindowResponse,
    AvailableTimestampsResponse,
    MetricsSummaryResponse,
    DatasetSummaryResponse,
    HealthResponse
)
from app.services.inference import inference_service, MODEL_DISPLAY_NAMES
from app.core.config import settings

router = APIRouter()


@router.get("/health", response_model=HealthResponse, tags=["Health"])
async def check_health():
    """System health and artifact readiness check."""
    try:
        inference_service.load_artifacts()
        return HealthResponse(
            status="healthy",
            version=settings.VERSION,
            models_loaded=list(inference_service.models.keys()),
            dataset_loaded=inference_service.df is not None
        )
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Inference engine health check failed: {str(exc)}"
        )


@router.get("/summary", response_model=DatasetSummaryResponse, tags=["Telemetry"])
async def get_summary():
    """Return high-level telemetry dataset metadata and operational parameters."""
    try:
        return inference_service.get_dataset_summary()
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve dataset summary: {str(exc)}"
        )


@router.get("/models", tags=["Models"])
async def get_models():
    """List available pre-trained model architectures and classification modes."""
    try:
        inference_service.load_artifacts()
        models_info = []
        for key in inference_service.models.keys():
            arch_type = "Recurrent Deep Learning (24-Step Sequence)" if key in ("lstm", "cnn_1d") else "Classical Tabular Classifier (T0 Snapshot)"
            threshold = inference_service.metrics_summary.get(key, {}).get("threshold", 0.5)
            models_info.append({
                "id": key,
                "display_name": MODEL_DISPLAY_NAMES.get(key, key.upper()),
                "architecture_type": arch_type,
                "frozen_threshold": round(float(threshold), 4)
            })
        return {"models": models_info}
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve model list: {str(exc)}"
        )


@router.get("/metrics", response_model=MetricsSummaryResponse, tags=["Evaluation"])
async def get_metrics():
    """Retrieve frozen validation and test-set performance benchmarks across models."""
    try:
        return inference_service.get_metrics_matrix()
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to compile metrics matrix: {str(exc)}"
        )


@router.get("/timestamps", response_model=AvailableTimestampsResponse, tags=["Telemetry"])
async def get_timestamps():
    """Retrieve chronologically sorted observation timestamps with full 24h history."""
    try:
        return inference_service.get_available_timestamps()
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve timestamps: {str(exc)}"
        )


@router.get("/telemetry", response_model=TelemetryWindowResponse, tags=["Telemetry"])
async def get_telemetry_window(
    timestamp: Optional[str] = Query(
        default=None,
        description="Target observation timestamp T0 (UTC format: 'YYYY-MM-DD HH:MM:SS'). Defaults to latest available."
    )
):
    """Retrieve continuous 24-hour sequence of SHARP magnetic and flare telemetry parameters."""
    try:
        return inference_service.get_telemetry_window(timestamp_str=timestamp)
    except ValueError as val_err:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(val_err)
        )
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to fetch telemetry window: {str(exc)}"
        )


@router.post("/predict", response_model=PredictionResponse, tags=["Prediction"])
async def run_prediction(request: PredictionRequest):
    """
    Execute real-time M+ solar flare forecast using specified model and temporal window.
    Applies strict leakage-free scaling and frozen validation threshold.
    """
    try:
        return inference_service.predict(
            model_name=request.model_name,
            timestamp_str=request.timestamp,
            custom_window=request.custom_window
        )
    except ValueError as val_err:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(val_err)
        )
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Inference execution failed: {str(exc)}"
        )
