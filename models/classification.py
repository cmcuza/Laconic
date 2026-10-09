from __future__ import annotations
import numpy as np
from aeon.classification.distance_based import ProximityForest
from aeon.classification.feature_based import TSFreshClassifier
from .base import Classifier
from typing import Any, Dict

class ProximityForestClassifier(Classifier):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.model_name = "ProximityForest"
        self.model = ProximityForest(**kwargs)
    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        self.model.fit(X, y)
    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(X)

class FeatureClassifier(Classifier):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.model_name = "TSFresh"
        self.model = TSFreshClassifier(**kwargs)
    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        self.model.fit(X, y)
    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(X)

def build_model(name: str, kwargs: Dict[str, Any]):
    n = name.lower()
    if n in ("proximity_forest","pf","random_forest","proximityforest"):
        return ProximityForestClassifier(**kwargs)
    if n in ("tsfresh","tf","feature_classifier","tsfresh"):
        return FeatureClassifier(**kwargs)
    raise ValueError(f"Unknown classification model: {name}")
