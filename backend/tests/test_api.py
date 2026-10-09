"""
Unit and integration tests for FastAPI backend endpoints and ML inference.
"""

import sys
from pathlib import Path
import pytest
from fastapi.testclient import TestClient

# Ensure backend directory is in sys.path
BACKEND_DIR = Path(__file__).resolve().parent.parent
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from app.main import app
from app.services.inference import inference_service


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as test_client:
        yield test_client


def test_root_endpoint(client):
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "OPERATIONAL"
    assert "version" in data


def test_health_endpoint(client):
    response = client.get("/api/v1/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert "models_loaded" in data
    assert len(data["models_loaded"]) > 0


def test_models_endpoint(client):
    response = client.get("/api/v1/models")
    assert response.status_code == 200
    data = response.json()
    assert "models" in data
    model_ids = [m["id"] for m in data["models"]]
    assert "logistic_regression" in model_ids
    assert "random_forest" in model_ids
    assert "xgboost" in model_ids
    assert "lstm" in model_ids


def test_summary_endpoint(client):
    response = client.get("/api/v1/summary")
    assert response.status_code == 200
    data = response.json()
    assert data["total_observations"] > 0
    assert data["feature_count"] > 0
    assert len(data["feature_names"]) == data["feature_count"]
    assert "cadence_hours" in data


def test_metrics_endpoint(client):
    response = client.get("/api/v1/metrics")
    assert response.status_code == 200
    data = response.json()
    assert "models" in data
    assert len(data["models"]) >= 4

    for m in data["models"]:
        assert "pr_auc" in m
        assert "tss" in m
        assert "far" in m
        assert "threshold" in m
        assert 0.0 <= m["threshold"] <= 1.0


def test_timestamps_endpoint(client):
    response = client.get("/api/v1/timestamps")
    assert response.status_code == 200
    data = response.json()
    assert data["total_count"] > 0
    assert len(data["timestamps"]) == data["total_count"]
    assert data["earliest"] is not None
    assert data["latest"] is not None


def test_telemetry_endpoint_latest(client):
    response = client.get("/api/v1/telemetry")
    assert response.status_code == 200
    data = response.json()
    assert len(data["timestamps"]) == 24
    assert len(data["features"]) > 0
    for feat_vals in data["features"].values():
        assert len(feat_vals) == 24


def test_telemetry_endpoint_specific_timestamp(client):
    ts_response = client.get("/api/v1/timestamps")
    valid_ts = ts_response.json()["timestamps"][50]

    response = client.get(f"/api/v1/telemetry?timestamp={valid_ts}")
    assert response.status_code == 200
    data = response.json()
    assert data["target_timestamp"] == valid_ts
    assert len(data["timestamps"]) == 24


def test_predict_classical_models(client):
    for model_name in ["logistic_regression", "random_forest", "xgboost"]:
        payload = {"model_name": model_name}
        response = client.post("/api/v1/predict", json=payload)
        assert response.status_code == 200
        data = response.json()
        assert data["model_name"] == model_name
        assert 0.0 <= data["probability"] <= 1.0
        assert data["probability_percent"] == pytest.approx(data["probability"] * 100.0, rel=1e-3)
        assert isinstance(data["is_flare"], bool)
        assert data["risk_level"] in ("NORMAL", "MODERATE", "CRITICAL")
        assert len(data["window_timestamps"]) == 24


def test_predict_lstm_model(client):
    payload = {"model_name": "lstm"}
    response = client.post("/api/v1/predict", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["model_name"] == "lstm"
    assert 0.0 <= data["probability"] <= 1.0
    assert isinstance(data["is_flare"], bool)
    assert data["risk_level"] in ("NORMAL", "MODERATE", "CRITICAL")


def test_predict_invalid_model(client):
    payload = {"model_name": "non_existent_neural_net"}
    response = client.post("/api/v1/predict", json=payload)
    assert response.status_code == 400
