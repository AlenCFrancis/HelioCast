"""
Logistic Regression Model Wrapper for Solar Flare Prediction.
"""

import os
import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression

class SolarLogisticRegression:
    def __init__(self, max_iter: int = 1000, class_weight: str = "balanced", random_state: int = 42):
        self.model = LogisticRegression(
            max_iter=max_iter,
            class_weight=class_weight,
            random_state=random_state
        )

    def fit(self, X_train: np.ndarray, y_train: np.ndarray):
        self.model.fit(X_train, y_train)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Returns probability of positive class (M+ flare)."""
        return self.model.predict_proba(X)[:, 1]

    def predict(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        probas = self.predict_proba(X)
        return (probas >= threshold).astype(int)

    def save(self, filepath: str = "models/logistic_regression.joblib"):
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        joblib.dump(self.model, filepath)
        print(f"[Logistic Regression] Model saved to {filepath}")

    @classmethod
    def load(cls, filepath: str = "models/logistic_regression.joblib"):
        instance = cls()
        instance.model = joblib.load(filepath)
        return instance
