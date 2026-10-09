from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple
import numpy as np
from evals.metrics import MetricSpec, combine_fitness
from compression.utils import compress_and_decompress_batch
from profiling import stage_clock


@dataclass
class ClassificationObjective:
    model: Any
    X_val: np.ndarray
    y_val: np.ndarray
    backend: Any
    alpha: float
    spec: MetricSpec
    classes: Optional[np.ndarray] = None

    def _compute_components(self, parameters_vector: Dict[str, float], eval_indices: Optional[np.ndarray] = None) -> Tuple[float, float]:
        X_val = self.X_val if eval_indices is None else self.X_val[eval_indices]
        y_val = self.y_val if eval_indices is None else self.y_val[eval_indices]

        params = self.backend.params_from_vector(parameters_vector)
        with stage_clock.stage("compress"):
            Xv_rec, avg_cr = compress_and_decompress_batch(X_val, self.backend, params)

        need_score = self.spec.needs_score
        with stage_clock.stage("inference"):
            if need_score and hasattr(self.model, "predict_both"):
                y_pred, y_score = self.model.predict_both(Xv_rec, need_score=True)
            else:
                y_pred, y_score = self.model.predict(Xv_rec), None

        with stage_clock.stage("metric"):
            metric_raw = self.spec.fn(y_true=y_val, y_pred=y_pred, y_score=y_score, classes=self.classes)
            metric_norm = self.spec.normalized(float(metric_raw))
        return metric_norm, avg_cr

    def __call__(self, parameters_vector: Dict[str, float], eval_indices: Optional[np.ndarray] = None) -> float:
        metric_norm, avg_cr = self._compute_components(parameters_vector, eval_indices)
        return combine_fitness(metric_norm, avg_cr, self.alpha)

@dataclass
class RegressionObjective:
    model: Any
    X_val: np.ndarray
    y_val: np.ndarray
    backend: Any
    alpha: float
    spec: MetricSpec
    classes: Optional[np.ndarray] = None

    def _compute_components(self, parameters_vector: Dict[str, float], eval_indices: Optional[np.ndarray] = None) -> Tuple[float, float]:
        X_val = self.X_val if eval_indices is None else self.X_val[eval_indices]
        y_val = self.y_val if eval_indices is None else self.y_val[eval_indices]

        params = self.backend.params_from_vector(parameters_vector)
        Xv_rec, avg_cr = np.empty_like(X_val), []
        for d in range(X_val.shape[2]):
            with stage_clock.stage("compress"):
                Xv_rec_dim, avg_cr_dim = compress_and_decompress_batch(X_val[..., d], self.backend, params)
            Xv_rec[..., d] = Xv_rec_dim
            avg_cr.append(avg_cr_dim)
        avg_cr = np.mean(avg_cr)

        with stage_clock.stage("inference"):
            y_pred = self.model.predict(Xv_rec)

        with stage_clock.stage("metric"):
            metric = self.spec.fn(y_true=y_val, y_pred=y_pred)
        return metric, avg_cr

    def __call__(self, parameters_vector: Dict[str, float], eval_indices: Optional[np.ndarray] = None) -> float:
        metric, avg_cr = self._compute_components(parameters_vector, eval_indices)
        return combine_fitness(metric, avg_cr, self.alpha)


@dataclass
class ClusteringObjective:
    model: Any
    X_val: np.ndarray
    y_val: np.ndarray
    backend: Any
    alpha: float
    spec: MetricSpec
    classes: Optional[np.ndarray] = None

    def _compute_components(self, parameters_vector: Dict[str, float], eval_indices: Optional[np.ndarray] = None) -> Tuple[float, float]:
        X_val = self.X_val if eval_indices is None else self.X_val[eval_indices]
        y_val = self.y_val if eval_indices is None else self.y_val[eval_indices]

        params = self.backend.params_from_vector(parameters_vector)
        with stage_clock.stage("compress"):
            Xv_rec, avg_cr = compress_and_decompress_batch(X_val, self.backend, params)

        need_score = self.spec.needs_score
        with stage_clock.stage("inference"):
            if need_score and hasattr(self.model, "predict_both"):
                y_pred, y_score = self.model.predict_both(Xv_rec, need_score=True)
            else:
                y_pred, y_score = self.model.predict(Xv_rec), None

        with stage_clock.stage("metric"):
            metric_raw = self.spec.fn(y_true=y_val, y_pred=y_pred, y_score=y_score, classes=self.classes)
            metric_norm = self.spec.normalized(float(metric_raw))
        return metric_norm, avg_cr

    def __call__(self, parameters_vector: Dict[str, float], eval_indices: Optional[np.ndarray] = None) -> float:
        metric_norm, avg_cr = self._compute_components(parameters_vector, eval_indices)
        return combine_fitness(metric_norm, avg_cr, self.alpha)


@dataclass
class ForecastingObjective:
    model: Any
    X_val: np.ndarray
    y_val: np.ndarray
    backend: Any
    alpha: float
    spec: MetricSpec

    def _compute_components(self, parameters_vector: Dict[str, float]) -> Tuple[float, float]:
        params = self.backend.params_from_vector(parameters_vector)
        with stage_clock.stage("compress"):
            Xv_rec, avg_cr = compress_and_decompress_batch(self.X_val[np.newaxis, :], self.backend, params)

        with stage_clock.stage("inference"):
            _, y_pred = self.model.predict(Xv_rec.squeeze())

        with stage_clock.stage("metric"):
            metric = self.spec.fn(y_true=self.y_val, y_pred=y_pred)
        return metric, avg_cr

    def __call__(self, parameters_vector: Dict[str, float]) -> float:
        metric, avg_cr = self._compute_components(parameters_vector)
        return combine_fitness(metric, avg_cr, self.alpha)
