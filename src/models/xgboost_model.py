"""
XGBoost Classifier Wrapper for Solar Flare Prediction.
"""

import os
import joblib
import numpy as np
from xgboost import XGBClassifier

class SolarXGBoost:
    def __init__(
        self,
        n_estimators: int = 500,
        max_depth: int = 6,
        learning_rate: float = 0.05,
        subsample: float = 0.8,
        colsample_bytree: float = 0.8,
        eval_metric: str = "logloss",
        random_state: int = 42,
        scale_pos_weight: float = 1.0
    ):
        self.model = XGBClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            subsample=subsample,
            colsample_bytree=colsample_bytree,
            eval_metric=eval_metric,
            random_state=random_state,
            scale_pos_weight=scale_pos_weight
        )

    def fit(self, X_train: np.ndarray, y_train: np.ndarray, eval_set=None):
        num_neg = np.sum(y_train == 0)
        num_pos = np.sum(y_train == 1)
        if num_pos > 0 and self.model.scale_pos_weight == 1.0:
            scale_weight = float(num_neg) / float(num_pos)
            self.model.set_params(scale_pos_weight=scale_weight)

        self.model.fit(X_train, y_train, eval_set=eval_set, verbose=False)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Returns probability of positive class (M+ flare)."""
        return self.model.predict_proba(X)[:, 1]

    def predict(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        probas = self.predict_proba(X)
        return (probas >= threshold).astype(int)

    def save(self, filepath: str = "models/xgboost.joblib"):
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        joblib.dump(self.model, filepath)
        print(f"[XGBoost] Model saved to {filepath}")

    @classmethod
    def load(cls, filepath: str = "models/xgboost.joblib"):
        instance = cls()
        instance.model = joblib.load(filepath)
        return instance
