# experiments/objective_fn.py
import inspect
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
from evals.metrics import MetricSpec, combine_fitness
from compression.utils import compress_and_decompress_batch
from profiling import stage_clock

# ---------------------------------------------------------------------
# objective (tunes on validation; model is trained on raw train)
#
# Trimmed to the four active tasks (classification, clustering, regression,
# forecasting). AnomalyObjective is parked, not deleted, alongside the rest of
# the anomaly task - see project CLAUDE.md.
# ---------------------------------------------------------------------


@dataclass
class ClassificationObjective:
    model: Any
    X_val: np.ndarray
    y_val: np.ndarray
    backend: Any
    alpha: float
    spec: MetricSpec                 # <- pass the spec, not a lambda
    classes: Optional[np.ndarray] = None

    def _compute_components(self, parameters_vector: Dict[str, float], eval_indices: Optional[np.ndarray] = None) -> Tuple[float, float]:
        # ``eval_indices`` (optional) restricts the evaluation to a subset of the
        # validation series. Because both the task metric (e.g. accuracy) and
        # ``avg_cr`` are means over the per-series validation set, a subset is an
        # unbiased estimator of the full-set fitness — this is what multi-fidelity
        # optimizers (e.g. mfrfei) exploit to screen many configs cheaply.
        X_val = self.X_val if eval_indices is None else self.X_val[eval_indices]
        y_val = self.y_val if eval_indices is None else self.y_val[eval_indices]

        params = self.backend.params_from_vector(parameters_vector)
        with stage_clock.stage("compress"):
            Xv_rec, avg_cr = compress_and_decompress_batch(X_val, self.backend, params)

        # one model call -> both outputs. This is import for anomaly detection.
        need_score = self.spec.needs_score
        with stage_clock.stage("inference"):
            if need_score and hasattr(self.model, "predict_both"):
                y_pred, y_score = self.model.predict_both(Xv_rec, need_score=True)
            else:
                y_pred, y_score = self.model.predict(Xv_rec), None

        # raw metric -> normalized (0..1 if needed)
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
    spec: MetricSpec                 # <- pass the spec, not a lambda
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

        # one model call -> both outputs
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
    spec: MetricSpec                 # <- pass the spec, not a lambda
    classes: Optional[np.ndarray] = None

    def _compute_components(self, parameters_vector: Dict[str, float], eval_indices: Optional[np.ndarray] = None) -> Tuple[float, float]:
        X_val = self.X_val if eval_indices is None else self.X_val[eval_indices]
        y_val = self.y_val if eval_indices is None else self.y_val[eval_indices]

        params = self.backend.params_from_vector(parameters_vector)
        with stage_clock.stage("compress"):
            Xv_rec, avg_cr = compress_and_decompress_batch(X_val, self.backend, params)

        # one model call -> both outputs
        need_score = self.spec.needs_score
        with stage_clock.stage("inference"):
            if need_score and hasattr(self.model, "predict_both"):
                y_pred, y_score = self.model.predict_both(Xv_rec, need_score=True)
            else:
                y_pred, y_score = self.model.predict(Xv_rec), None

        # raw metric -> normalized (0..1 if needed)
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
class PreferenceObjective:
    """Adapts any *Objective above to the `objective(candidate) ->
    (task_metric, avg_cr)` contract that optimizer/preference.py and
    subset-scoring callers use, by delegating to the wrapped
    objective's `_compute_components` and skipping the final
    `combine_fitness` scalarization those classes do for every other
    optimizer (see optimizer/preference.py's module docstring).

    The subset helpers exist for ballot-style scoring, which needs to score the
    same candidate on several resamples of the validation set to build its
    voters. They live here rather than in the optimizer because this
    is the class that owns `X_val`/`y_val` and the `eval_indices` contract -
    an optimizer reaching through `.inner` for them would couple every
    optimizer to every objective's field names."""

    inner: Any  # one of Classification/Clustering/Regression/ForecastingObjective

    def __call__(
        self,
        parameters_vector: Dict[str, float],
        eval_indices: Optional[np.ndarray] = None,
    ) -> Tuple[float, float]:
        if eval_indices is None:
            return self.inner._compute_components(parameters_vector)
        if not self.supports_subsets():
            raise ValueError(
                f"{type(self.inner).__name__} does not accept eval_indices; "
                "its validation set is one contiguous window of a single series, "
                "so a subset is not an unbiased estimator of the full-set score. "
                "Run this optimizer with n_voters=1 on this task."
            )
        return self.inner._compute_components(parameters_vector, eval_indices)

    def supports_subsets(self) -> bool:
        """Whether the wrapped objective can be scored on an index subset.
        False for forecasting - see the error message above."""
        return "eval_indices" in inspect.signature(
            self.inner._compute_components
        ).parameters

    def voter_indices(
        self, n_voters: int, rng: np.random.Generator, scheme: str = "bootstrap"
    ) -> List[np.ndarray]:
        """`n_voters` index sets over the validation series, one per voter.

        Both schemes stratify by `y_val` for the two tasks whose labels are
        discrete, so no voter can end up missing a class - which would make its
        task metric incomparable with its siblings' rather than merely noisier.

        - ``partition``: disjoint slices covering the set. Independent voters,
          but each sees only `n/n_voters` series, so it needs a validation set
          several times `n_voters` wide. **Most of this repo's datasets are not
          that wide** - measured, the ucr_small group's validation splits are
          Coffee 6, Wine 12, Lightning2 12, ECG200 20, Plane 21 (and only
          Adiac/CricketX 78, Wafer 200).
        - ``bootstrap``: resamples of the FULL width, drawn with replacement
          (stratified: within each class, preserving its count). Every voter
          sees `n` series regardless of how small `n` is, which is why this is
          the default. The trade is that voters overlap (~63% expected shared
          series), so they carry less independent information than disjoint
          slices - the aggregation measures resample stability rather than
          independent-split stability."""
        n = len(self.inner.X_val)
        if n_voters < 1:
            raise ValueError(f"n_voters must be >= 1, got {n_voters}")
        discrete = isinstance(self.inner, (ClassificationObjective, ClusteringObjective))
        groups = (
            [np.arange(n)]
            if not discrete
            else [np.flatnonzero(np.asarray(self.inner.y_val) == value)
                  for value in np.unique(np.asarray(self.inner.y_val))]
        )
        if scheme == "partition":
            parts: List[List[int]] = [[] for _ in range(n_voters)]
            for members in groups:
                for offset, index in enumerate(rng.permutation(members)):
                    parts[offset % n_voters].append(int(index))
            return [np.sort(np.asarray(part, dtype=int)) for part in parts]
        if scheme == "bootstrap":
            return [
                np.sort(np.concatenate([
                    rng.choice(members, size=len(members), replace=True) for members in groups
                ]))
                for _ in range(n_voters)
            ]
        raise ValueError(f"Unknown voter scheme '{scheme}'. Expected 'partition' or 'bootstrap'.")
