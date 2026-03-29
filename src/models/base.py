"""
BaseMLBModel: abstract base class for all ML prediction models.

Every model in this system implements this interface so they can be:
- Registered and trained via training/train_all.py without code changes
- Swapped out by changing the `class` entry in config/models.yaml
- Loaded and served by the API in a uniform way

The interface is intentionally minimal. Each subclass handles its own
hyperparameter config, feature selection, and serialization format.
"""
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional

import joblib
import numpy as np
import pandas as pd


class BaseMLBModel(ABC):
    """Abstract base class for all MLB prediction models."""

    model_type: str = ""  # Must be set by subclasses (matches config/models.yaml key)

    def __init__(self):
        self._model = None           # The underlying sklearn-compatible estimator
        self._feature_names: list[str] = []
        self._artifact_path: Optional[Path] = None

    # ------------------------------------------------------------------
    # Required interface
    # ------------------------------------------------------------------

    @abstractmethod
    def fit(self, X: pd.DataFrame, y: pd.Series) -> "BaseMLBModel":
        """
        Train the model on feature matrix X and targets y.
        Must set self._model and self._feature_names.
        Returns self for chaining.
        """
        ...

    @abstractmethod
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Generate predictions for feature matrix X.
        Returns a 1-D numpy array of predictions (same length as X).
        Missing features should be handled gracefully (imputed or NaN-safe).
        """
        ...

    @property
    def feature_names(self) -> list[str]:
        """Return the ordered list of feature names this model was trained on."""
        return self._feature_names

    @property
    def is_trained(self) -> bool:
        return self._model is not None

    # ------------------------------------------------------------------
    # Serialization (shared implementation)
    # ------------------------------------------------------------------

    def save(self, path: Path) -> None:
        """Save model artifact to disk using joblib."""
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({"model": self._model, "feature_names": self._feature_names}, path)
        self._artifact_path = path

    def load(self, path: Path) -> "BaseMLBModel":
        """Load model artifact from disk."""
        artifact = joblib.load(path)
        self._model = artifact["model"]
        self._feature_names = artifact["feature_names"]
        self._artifact_path = path
        return self

    # ------------------------------------------------------------------
    # Shared helpers
    # ------------------------------------------------------------------

    def _align_features(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Align a prediction DataFrame to the training feature set.
        Adds missing columns as NaN, drops extra columns, reorders.
        This ensures the model sees the same feature order as at training time.
        """
        missing = set(self._feature_names) - set(X.columns)
        if missing:
            for col in missing:
                X = X.copy()
                X[col] = np.nan
        return X[self._feature_names]

    def _drop_non_features(self, df: pd.DataFrame,
                            non_feature_cols: list[str]) -> pd.DataFrame:
        """Drop metadata/target columns before passing to model."""
        return df.drop(columns=[c for c in non_feature_cols if c in df.columns])
