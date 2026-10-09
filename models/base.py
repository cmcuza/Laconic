"""Base model wrappers with shared disk caching."""
from __future__ import annotations
import os, json, hashlib, platform
from typing import Any, Dict, Optional
import numpy as np
import joblib


class _CachedModelMixin:
    def _call_fit(self, X: np.ndarray, y: Optional[np.ndarray]) -> None:
        if y is None:
            self.fit(X)
        else:
            self.fit(X, y)

    def _get_cache_payload(self) -> Any:
        return self.model

    def _load_cache_payload(self, payload: Any) -> None:
        self.model = payload

    def fit_cached(
        self,
        X: np.ndarray,
        y: Optional[np.ndarray] = None,
        *,
        dataset: str,
        random_state: int,
        fold: int,
        splitter_name: str,
        train_indices: np.ndarray,
        out_dir: str,
    ) -> None:
        cache_dir = self._cache_dir(out_dir, dataset, self.model_name, random_state, fold)
        expected = self._expected_meta(dataset, random_state, fold, splitter_name, train_indices)

        loaded = self._try_load(cache_dir, expected)
        if loaded is not None:
            self._load_cache_payload(loaded)
            return

        self._call_fit(X, y)
        meta = dict(expected, train_size=int(train_indices.shape[0]))
        self._save(cache_dir, self._get_cache_payload(), meta)

    @staticmethod
    def _cache_dir(out_dir: str, dataset: str, model_name: str, random_state: int, fold: int) -> str:
        return os.path.join(out_dir, "_model_cache", dataset, model_name, f"rs{random_state}", f"fold_{fold}").lower()

    def _expected_meta(
        self,
        dataset: str,
        random_state: int,
        fold: int,
        splitter_name: str,
        train_indices: np.ndarray,
    ) -> Dict[str, Any]:
        import sklearn
        import aeon

        mk = json.dumps(self.kwargs or {}, sort_keys=True)
        return {
            "dataset": dataset,
            "model_name": self.model_name,
            "random_state": int(random_state),
            "fold": int(fold),
            "splitter_name": splitter_name,
            "train_idx_sha1": hashlib.sha1(np.asarray(train_indices, dtype=np.int64).tobytes()).hexdigest(),
            "model_kwargs_sha1": hashlib.sha1(mk.encode("utf-8")).hexdigest(),
            "model_implementation": getattr(self, "_cache_implementation_id", None),
            "versions": {
                "python": platform.python_version(),
                "numpy": np.__version__,
                "scikit_learn": sklearn.__version__,
                "aeon": aeon.__version__,
            },
        }

    @staticmethod
    def _save(path_dir: str, payload: Any, meta: Dict[str, Any]) -> None:
        os.makedirs(path_dir, exist_ok=True)
        tag = os.getpid()
        model_tmp = os.path.join(path_dir, f".model.joblib.{tag}")
        meta_tmp = os.path.join(path_dir, f".meta.json.{tag}")
        joblib.dump(payload, model_tmp)
        with open(meta_tmp, "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2, sort_keys=True)
        os.replace(model_tmp, os.path.join(path_dir, "model.joblib"))
        os.replace(meta_tmp, os.path.join(path_dir, "meta.json"))

    @staticmethod
    def _try_load(path_dir: str, expected: Dict[str, Any]) -> Optional[Any]:
        model_path = os.path.join(path_dir, "model.joblib")
        meta_path  = os.path.join(path_dir, "meta.json")
        if not (os.path.exists(model_path) and os.path.exists(meta_path)):
            return None

        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)

        mismatched = [k for k in expected if meta.get(k) != expected[k]]
        if mismatched:
            print(f"Model cache at {path_dir} mismatches on {mismatched}; retraining.")
            return None

        return joblib.load(model_path)


class _ScoringMixin:
    def _standardize_scores(self, s) -> np.ndarray:
        s = np.asarray(s)
        if s.ndim == 1:
            s = np.stack([-s, s], axis=1)
        return s

    def predict_both(self, X: np.ndarray, need_score: bool = False):
        """Return (y_pred, y_score or None), deriving y_pred from scores when available."""
        classes = getattr(self.model, "classes_", None)

        if need_score:
            if hasattr(self.model, "predict_proba"):
                score = np.asarray(self.model.predict_proba(X))
                y_pred = (classes[np.argmax(score, axis=1)]
                        if classes is not None else np.argmax(score, axis=1))
                return y_pred, score

            if hasattr(self.model, "decision_function"):
                raw = self.model.decision_function(X)
                score = self._standardize_scores(raw)
                y_pred = (classes[np.argmax(score, axis=1)]
                        if classes is not None else np.argmax(score, axis=1))
                return y_pred, score

        y_pred = self.model.predict(X)
        return y_pred, None

    def predict_proba(self, X: np.ndarray):
        if hasattr(self.model, "predict_proba"):
            return self.model.predict_proba(X)
        if hasattr(self.model, "decision_function"):
            return self._standardize_scores(self.model.decision_function(X))
        return None

    def get_classes(self):
        return getattr(self.model, "classes_", None)


class Classifier(_CachedModelMixin, _ScoringMixin):
    """Base classifier interface + shared caching."""
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.model_name = getattr(self, "model_name", self.__class__.__name__)
        self.model = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        raise NotImplementedError

    def predict(self, X: np.ndarray) -> np.ndarray:
        raise NotImplementedError


class Clustering(_CachedModelMixin):
    """Base cluster interface + shared caching."""
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.model_name = getattr(self, "model_name", self.__class__.__name__)
        self.model = None

    def fit(self, X: np.ndarray) -> None:
        raise NotImplementedError

    def predict(self, X: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def get_clusters(self):
        return getattr(self.model, "classes_", None)


class Forecasting(_CachedModelMixin):
    """Base forecasting interface + shared caching."""
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.model_name = getattr(self, "model_name", self.__class__.__name__)
        self.model = None

    def fit(self, X: np.ndarray) -> None:
        raise NotImplementedError

    def predict(self, X: np.ndarray) -> np.ndarray:
        raise NotImplementedError


class Regressor(_CachedModelMixin, _ScoringMixin):
    """Base regressor interface + shared caching."""
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.model_name = getattr(self, "model_name", self.__class__.__name__)
        self.model = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        raise NotImplementedError

    def predict(self, X: np.ndarray) -> np.ndarray:
        raise NotImplementedError
