"""
Unit tests for 1D CNN model (instantiation, forward pass, save, load).
"""

import pytest
import os
import numpy as np
from src.models.cnn_1d import SolarCNN1DWrapper


@pytest.fixture
def synthetic_3d_data():
    np.random.seed(42)
    X = np.random.randn(50, 24, 10).astype(np.float32)
    y = np.random.randint(0, 2, size=50).astype(np.float32)
    return X, y


def test_cnn_1d_forward(synthetic_3d_data):
    X_seq, y_seq = synthetic_3d_data
    wrapper = SolarCNN1DWrapper(input_size=10, channels=[16, 32], kernel_size=3, dropout=0.0)

    probs = wrapper.predict_proba(X_seq)
    preds = wrapper.predict(X_seq, threshold=0.5)

    assert probs.shape == (50,)
    assert (probs >= 0.0).all() and (probs <= 1.0).all()
    assert preds.shape == (50,)


def test_cnn_1d_probabilities(synthetic_3d_data):
    X_seq, _ = synthetic_3d_data
    wrapper = SolarCNN1DWrapper(input_size=10, channels=[16, 32], kernel_size=3, dropout=0.0)

    probs = wrapper.predict_proba(X_seq)
    # Probabilities must be valid
    assert np.all(np.isfinite(probs))
    assert probs.min() >= 0.0
    assert probs.max() <= 1.0


def test_cnn_1d_save_load(synthetic_3d_data, tmp_path):
    X_seq, y_seq = synthetic_3d_data
    wrapper = SolarCNN1DWrapper(input_size=10, channels=[16, 32], kernel_size=3, dropout=0.0)

    probs = wrapper.predict_proba(X_seq)

    save_file = os.path.join(tmp_path, "cnn_1d.pt")
    wrapper.save(save_file)
    loaded_wrapper = SolarCNN1DWrapper.load(save_file)
    loaded_probs = loaded_wrapper.predict_proba(X_seq)
    np.testing.assert_array_almost_equal(probs, loaded_probs, decimal=5)
