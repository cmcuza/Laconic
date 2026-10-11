import sys
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple
import numpy as np
from evals.metrics import MetricSpec, combine_fitness
from evals.anomaly_metrics import NonFiniteScores, anomaly_q
from compression.utils import compress_and_decompress_batch, reconstruct_array
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


@dataclass
class AnomalyObjective:
    """J(p) = alpha * Q_val(p) + (1 - alpha) * (1 - 1/CR(p)) for one series.

    Every candidate pipeline compresses and reconstructs the series. The detector
    scores the whole reconstruction, and Q is computed on the validation range
    [0, val_end) only. If the detector returns non-finite scores for the
    reconstruction, Q is 0 and the candidate keeps its storage term. Any other
    error propagates.
    """
    x_clean: np.ndarray
    y: np.ndarray
    detector: Any
    backend: Any
    alpha: float
    val_end: int
    q_measure: str = "AUC-PR"
    sliding_window: Optional[int] = None

    def _compute_components(self, parameters_vector: Dict[str, float]) -> Tuple[float, float]:
        params = self.backend.params_from_vector(parameters_vector)
        with stage_clock.stage("compress"):
            x_rec, avg_cr = reconstruct_array(self.x_clean, self.backend, params)

        try:
            with stage_clock.stage("inference"):
                scores = self.detector.score(x_rec)
        except NonFiniteScores as err:
            print(f"[anomaly] Q set to 0: {err}", file=sys.stderr)
            return 0.0, avg_cr
        with stage_clock.stage("metric"):
            # Q on the validation range only.
            q_val = anomaly_q(self.y[:self.val_end], scores[:self.val_end],
                              self.q_measure, self.sliding_window)
        if not np.isfinite(q_val):
            raise ValueError(f"non-finite {self.q_measure}: {q_val}")
        return q_val, avg_cr

    def __call__(self, parameters_vector: Dict[str, float], eval_indices: Optional[np.ndarray] = None) -> float:
        if eval_indices is not None:
            raise NotImplementedError("eval_indices is not supported: the objective scores one series")
        q_val, avg_cr = self._compute_components(parameters_vector)
        return combine_fitness(q_val, avg_cr, self.alpha)
