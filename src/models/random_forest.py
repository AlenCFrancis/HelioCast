"""
Random Forest Classifier Wrapper for Solar Flare Prediction.
"""

import os
import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier

class SolarRandomForest:
    def __init__(
        self,
        n_estimators: int = 300,
        max_depth: int = 15,
        class_weight: str = "balanced",
        random_state: int = 42,
        n_jobs: int = -1
    ):
        self.model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            class_weight=class_weight,
            random_state=random_state,
            n_jobs=n_jobs
        )

    def fit(self, X_train: np.ndarray, y_train: np.ndarray):
        self.model.fit(X_train, y_train)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Returns probability of positive class (M+ flare)."""
        return self.model.predict_proba(X)[:, 1]

    def predict(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        probas = self.predict_proba(X)
        return (probas >= threshold).astype(int)

    def save(self, filepath: str = "models/random_forest.joblib"):
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        joblib.dump(self.model, filepath)
        print(f"[Random Forest] Model saved to {filepath}")

    @classmethod
    def load(cls, filepath: str = "models/random_forest.joblib"):
        instance = cls()
        instance.model = joblib.load(filepath)
        return instance
