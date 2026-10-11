"""TSB-AD detectors, dispatched by name through TSB_AD.model_wrapper."""
from __future__ import annotations
import inspect
from typing import Any, Dict
import numpy as np

import TSB_AD.model_wrapper as model_wrapper
from TSB_AD.HP_list import Optimal_Uni_algo_HP_dict

from evals.anomaly_metrics import NonFiniteScores


# run_experiments.py adds random_state to every model's kwargs. No TSB-AD runner takes it.
_FRAMEWORK_KWARGS = frozenset({"random_state"})

# Seed of NumPy's global RNG during a detector call, so stochastic detectors are repeatable.
DETECTOR_SEED = 0


def _accepted_kwargs(runner: Any, name: str, kwargs: Dict[str, Any]) -> Dict[str, Any]:
    """Drop framework kwargs and raise on any kwarg the TSB-AD runner does not take."""
    parameters = inspect.signature(runner).parameters
    kwargs = {k: v for k, v in kwargs.items() if k not in _FRAMEWORK_KWARGS}
    if any(p.kind is p.VAR_KEYWORD for p in parameters.values()):
        return kwargs
    unknown = sorted(k for k in kwargs if k not in parameters)
    if unknown:
        raise TypeError(f"run_{name}() does not accept {unknown}")
    return kwargs


class TSBADDetector:
    """One unsupervised TSB-AD detector with the hyperparameters tuned by TSB-AD.

    The hyperparameters come from TSB_AD.HP_list.Optimal_Uni_algo_HP_dict and are the
    same for the raw and the compressed series. A detector without an entry runs with
    its runner's defaults. `overrides` replace single hyperparameters.
    """

    def __init__(self, name: str, **overrides: Any):
        if name not in model_wrapper.Unsupervise_AD_Pool:
            raise ValueError(f"Unknown unsupervised TSB-AD detector '{name}'")
        self.model_name = name
        runner = getattr(model_wrapper, f"run_{name}")
        tuned = dict(Optimal_Uni_algo_HP_dict.get(name) or {})
        self.hyperparameters = _accepted_kwargs(runner, name, {**tuned, **overrides})

    def score(self, data: np.ndarray) -> np.ndarray:
        """Anomaly score for every point of `data`, which has shape (T, 1).

        No slidingWindow is passed: detectors that need one derive it from `data`.
        """
        state = np.random.get_state()
        try:
            np.random.seed(DETECTOR_SEED)
            scores = model_wrapper.run_Unsupervise_AD(self.model_name, data, **self.hyperparameters)
        finally:
            np.random.set_state(state)

        # model_wrapper returns the error message as a string when a detector raises.
        if not isinstance(scores, np.ndarray):
            raise RuntimeError(f"{self.model_name} failed: {scores!r}")
        if len(scores) != len(data):
            raise RuntimeError(f"{self.model_name} returned {len(scores)} scores for {len(data)} points")
        n_bad = int(np.count_nonzero(~np.isfinite(scores)))
        if n_bad:
            raise NonFiniteScores(f"{self.model_name} returned {n_bad}/{len(scores)} non-finite scores")
        return scores


def build_model(name: str, kwargs: Dict[str, Any]) -> TSBADDetector:
    return TSBADDetector(name, **kwargs)
