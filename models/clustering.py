from __future__ import annotations
import numpy as np
from numba import get_num_threads, set_num_threads
from sklearn.cluster import KMeans
from threadpoolctl import threadpool_limits
from aeon.clustering import KShape as AeonKShape, TimeSeriesKMedoids
from aeon.clustering.feature_based import TSFreshClusterer
from .base import Clustering
from typing import Any, Dict


class KMedoids(Clustering):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.model_name = "KMedoids"
        self.model = TimeSeriesKMedoids(**kwargs)
    def fit(self, X: np.ndarray) -> None:
        self.model.fit(X)
    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(X) + 1
    

class KShape(Clustering):
    _cache_implementation_id = "aeon.clustering.KShape"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.model_name = "KShape"
        kwargs['tol'] = float(kwargs['tol'])
        self.model = AeonKShape(**kwargs)
    def fit(self, X: np.ndarray) -> None:
        previous_numba_threads = get_num_threads()
        set_num_threads(1)
        try:
            with threadpool_limits(limits=1):
                self.model.fit(X)
        finally:
            set_num_threads(previous_numba_threads)
    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(X) + 1
    

class FeatureCluster(Clustering):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.model_name = "TSFreshCluster"
        self.model = TSFreshClusterer(
            estimator=KMeans(**kwargs["kmeans"]), **kwargs["tsfresh"]
        )
    def fit(self, X: np.ndarray) -> None:
        self.model.fit(X)
    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(X) + 1
    

def build_model(name: str, kwargs: Dict[str, Any]):
    n = name.lower()
    if n in ("tsfresh","tf","feature","tsfresh"):
        return FeatureCluster(**kwargs)
    if n in ("kmediods", "kmedoids", "kmdoids", "kmdiods"):
        return KMedoids(**kwargs)
    if n in ("kshape", "shape"):
        return KShape(**kwargs)
    raise ValueError(f"Unknown clustering model: {name}")
