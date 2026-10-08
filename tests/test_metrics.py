"""
Unit tests for metric evaluation routines and threshold optimization.
"""

import pytest
import numpy as np
from src.evaluation.metrics import (
    compute_all_metrics,
    find_optimal_threshold
)

def test_compute_all_metrics():
    y_true = np.array([0, 0, 0, 1, 1, 1, 1, 0, 0, 1])
    y_prob = np.array([0.1, 0.2, 0.8, 0.9, 0.7, 0.6, 0.85, 0.15, 0.3, 0.95])

    metrics = compute_all_metrics(y_true, y_prob, threshold=0.5)

    assert "roc_auc" in metrics
    assert "pr_auc" in metrics
    assert "precision" in metrics
    assert "recall" in metrics
    assert "f1_score" in metrics
    assert "tss" in metrics
    assert "far" in metrics
    assert "brier_score" in metrics
    assert "confusion_matrix" in metrics

    assert 0.0 <= metrics["roc_auc"] <= 1.0
    assert 0.0 <= metrics["pr_auc"] <= 1.0
    assert 0.0 <= metrics["brier_score"] <= 1.0

def test_find_optimal_threshold():
    y_true = np.array([0, 0, 0, 0, 1, 1, 1, 1])
    y_prob = np.array([0.1, 0.2, 0.3, 0.6, 0.4, 0.7, 0.8, 0.9])

    best_thresh = find_optimal_threshold(y_true, y_prob, metric_to_maximize="f1")
    assert 0.05 <= best_thresh <= 0.95
