# models/base.py
"""Base classes for every task's model wrapper, plus the shared disk-caching
behavior every one of them needs.

The original version of this file defined Classifier/Anomaly/Clustering/
Forecasting/Regressor as five separate classes that each duplicated the same
~150 lines of caching logic (fit_cached, _cache_dir, _build_meta,
_expected_meta, _save, _try_load) nearly verbatim. This version factors that
logic into _CachedModelMixin once; each task class is now just the thin part
that's actually task-specific (which abstract methods it declares, and
_ScoringMixin for the two tasks - Classifier/Regressor - that expose
predict_both/predict_proba). Public API (fit_cached's signature and behavior,
_cache_dir's path shape, etc.) is unchanged, so subclasses and callers written
against the old five classes work unmodified.
"""
from __future__ import annotations
import os, json, hashlib, platform
from typing import Any, Dict, Optional
import numpy as np
import joblib


class _CachedModelMixin:
    """fit_cached(...) and its disk-caching internals, shared by every task.

    Subclasses set self.kwargs/self.model_name/self.model in __init__ and
    implement fit(...)/predict(...). fit_cached transparently handles both
    calling conventions in use across runners: fit_cached(X, y, ...) for
    supervised tasks (classification/regression) and fit_cached(X, ...) for
    unsupervised ones (clustering/forecasting) - see _call_fit below.
    """

    def _call_fit(self, X: np.ndarray, y: Optional[np.ndarray]) -> None:
        """Dispatches to self.fit(X, y) or self.fit(X) depending on whether a
        target was given. Every concrete model in this repo needs exactly one
        of these two shapes, so subclasses don't need to override this."""
        if y is None:
            self.fit(X)
        else:
            self.fit(X, y)

    def _get_cache_payload(self) -> Any:
        """The object joblib-dumps to disk. Default: the fitted model itself.
        Override together with _load_cache_payload when a subclass needs to
        persist more than one object - e.g. RocketRegressor (models/regression.py)
        saves {"kernels": ..., "regressor": ...} since its numba-generated
        kernels live alongside the sklearn RidgeCV, not inside one attribute."""
        return self.model

    def _load_cache_payload(self, payload: Any) -> None:
        self.model = payload

    # --- caching API (runners call fit_cached instead of fit) ---
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

        # train fresh
        self._call_fit(X, y)
        meta = dict(expected, train_size=int(train_indices.shape[0]))
        self._save(cache_dir, self._get_cache_payload(), meta)

    # --- helpers ---
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
        """The full cache contract: identity of the training data and of the
        libraries the pickled model was fitted with. Every field is verified on
        every load - a cached model that doesn't match retrains rather than
        being silently reused for a different split/kwargs/library version."""
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
            # Subclasses can opt into invalidation when an upstream package
            # replaces an estimator without changing its package version.
            # Missing legacy values compare equal to None for other models.
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
        """Write the cache entry atomically.

        Several processes can miss the same entry at once and fit concurrently
        (the grid sweep shards one fold across workers). Writing in place would
        let one process read a half-written model.joblib; a temp file plus
        os.replace means a reader sees either the old entry or the new one.
        The pid keeps two concurrent writers off the same temp path.
        """
        os.makedirs(path_dir, exist_ok=True)
        tag = os.getpid()
        model_tmp = os.path.join(path_dir, f".model.joblib.{tag}")
        meta_tmp = os.path.join(path_dir, f".meta.json.{tag}")
        joblib.dump(payload, model_tmp)
        with open(meta_tmp, "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2, sort_keys=True)
        # Model first: meta.json is what _try_load checks, so it must not
        # appear before the payload it describes.
        os.replace(model_tmp, os.path.join(path_dir, "model.joblib"))
        os.replace(meta_tmp, os.path.join(path_dir, "meta.json"))

    @staticmethod
    def _try_load(path_dir: str, expected: Dict[str, Any]) -> Optional[Any]:
        """None = cache miss (absent, or contract mismatch -> retrain). A cache
        entry that exists but can't be read is corruption, not a miss - raise."""
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
    """predict_both/predict_proba/get_classes, shared by Classifier and
    Regressor (both originally duplicated this identically)."""

    def _standardize_scores(self, s) -> np.ndarray:
        """Return 2D scores: (n_samples, n_classes). Handles binary 1D decision_function."""
        s = np.asarray(s)
        if s.ndim == 1:  # binary decision_function -> make 2 columns (-s, s)
            s = np.stack([-s, s], axis=1)
        return s  # multiclass is already (n, C)

    def predict_both(self, X: np.ndarray, need_score: bool = False):
        """
        Returns (y_pred, y_score_or_None).
        - If need_score=True, tries predict_proba, else decision_function.
        - If scores available, derive y_pred via argmax over scores (no second model call).
        - If no scores available or not needed, fall back to .predict().
        """
        classes = getattr(self.model, "classes_", None)

        if need_score:
            # Prefer probabilities (some estimators compute them as part of trees' vote counts)
            if hasattr(self.model, "predict_proba"):
                score = np.asarray(self.model.predict_proba(X))   # shape (n, C)
                y_pred = (classes[np.argmax(score, axis=1)]
                        if classes is not None else np.argmax(score, axis=1))
                return y_pred, score

            # Fall back to decision_function (works for many linear/SVM-like models)
            if hasattr(self.model, "decision_function"):
                raw = self.model.decision_function(X)  # (n,) binary or (n,C) multiclass
                score = self._standardize_scores(raw)
                y_pred = (classes[np.argmax(score, axis=1)]
                        if classes is not None else np.argmax(score, axis=1))
                return y_pred, score

        # No score needed or unavailable -> just predict
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
    """Base classifier interface + shared caching.
       Subclasses must implement: fit(X, y), predict(X).
       Subclasses should set: self.model (the aeon/sklearn estimator), self.model_name (str).
    """
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.model_name = getattr(self, "model_name", self.__class__.__name__)
        self.model = None  # set by subclass

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        raise NotImplementedError

    def predict(self, X: np.ndarray) -> np.ndarray:
        raise NotImplementedError


class Clustering(_CachedModelMixin):
    """Base cluster interface + shared caching.
       Subclasses must implement: fit(X), predict(X).
       Subclasses should set: self.model (the aeon/sklearn estimator), self.model_name (str).
    """
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
    """Base forecasting interface + shared caching.
       Subclasses must implement: fit(X), predict(X) (single-series API - see
       models/forecasting.py's DLinear/XGBoostModel, both fit on one long
       series rather than a batch of (X, y) pairs).
       Subclasses should set: self.model, self.model_name (str).
    """
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.model_name = getattr(self, "model_name", self.__class__.__name__)
        self.model = None

    def fit(self, X: np.ndarray) -> None:
        raise NotImplementedError

    def predict(self, X: np.ndarray) -> np.ndarray:
        raise NotImplementedError


class Regressor(_CachedModelMixin, _ScoringMixin):
    """Base regressor interface + shared caching.
       Subclasses must implement: fit(X, y), predict(X).
       Subclasses should set: self.model (the aeon/sklearn estimator), self.model_name (str).
    """
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.model_name = getattr(self, "model_name", self.__class__.__name__)
        self.model = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        raise NotImplementedError

    def predict(self, X: np.ndarray) -> np.ndarray:
        raise NotImplementedError
