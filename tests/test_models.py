"""
Unit tests for machine learning and deep learning models (fit, predict, save, load).
"""

import pytest
import os
import numpy as np
from src.models.logistic_regression import SolarLogisticRegression
from src.models.random_forest import SolarRandomForest
from src.models.xgboost_model import SolarXGBoost
from src.models.lstm import SolarLSTMWrapper

@pytest.fixture
def synthetic_2d_data():
    np.random.seed(42)
    X = np.random.randn(100, 10).astype(np.float32)
    y = np.random.randint(0, 2, size=100).astype(int)
    return X, y

@pytest.fixture
def synthetic_3d_data():
    np.random.seed(42)
    X = np.random.randn(50, 24, 10).astype(np.float32)
    y = np.random.randint(0, 2, size=50).astype(np.float32)
    return X, y

def test_logistic_regression(synthetic_2d_data, tmp_path):
    X, y = synthetic_2d_data
    model = SolarLogisticRegression(random_state=42)
    model.fit(X, y)
    
    probs = model.predict_proba(X)
    preds = model.predict(X, threshold=0.5)

    assert probs.shape == (100,)
    assert (probs >= 0.0).all() and (probs <= 1.0).all()
    assert preds.shape == (100,)

    # Save & Load
    save_file = os.path.join(tmp_path, "lr.joblib")
    model.save(save_file)
    loaded_model = SolarLogisticRegression.load(save_file)
    loaded_probs = loaded_model.predict_proba(X)
    np.testing.assert_array_almost_equal(probs, loaded_probs)

def test_random_forest(synthetic_2d_data, tmp_path):
    X, y = synthetic_2d_data
    model = SolarRandomForest(n_estimators=10, random_state=42)
    model.fit(X, y)

    probs = model.predict_proba(X)
    assert probs.shape == (100,)

    save_file = os.path.join(tmp_path, "rf.joblib")
    model.save(save_file)
    loaded_model = SolarRandomForest.load(save_file)
    np.testing.assert_array_almost_equal(probs, loaded_model.predict_proba(X))

def test_xgboost(synthetic_2d_data, tmp_path):
    X, y = synthetic_2d_data
    model = SolarXGBoost(n_estimators=10, random_state=42)
    model.fit(X, y)

    probs = model.predict_proba(X)
    assert probs.shape == (100,)

    save_file = os.path.join(tmp_path, "xgb.joblib")
    model.save(save_file)
    loaded_model = SolarXGBoost.load(save_file)
    np.testing.assert_array_almost_equal(probs, loaded_model.predict_proba(X))

def test_lstm_model(synthetic_3d_data, tmp_path):
    X_seq, y_seq = synthetic_3d_data
    wrapper = SolarLSTMWrapper(input_size=10, hidden_size=32, num_layers=1, dropout=0.0)
    
    probs = wrapper.predict_proba(X_seq)
    preds = wrapper.predict(X_seq, threshold=0.5)

    assert probs.shape == (50,)
    assert (probs >= 0.0).all() and (probs <= 1.0).all()
    assert preds.shape == (50,)

    save_file = os.path.join(tmp_path, "lstm.pt")
    wrapper.save(save_file)
    loaded_wrapper = SolarLSTMWrapper.load(save_file)
    loaded_probs = loaded_wrapper.predict_proba(X_seq)
    np.testing.assert_array_almost_equal(probs, loaded_probs, decimal=5)
