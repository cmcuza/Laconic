"""BOMAB: Bandit-over-model-based optimization for TerseTS CASH.

This file implements :class:`BOMABOptimizer`, an optimizer tailored to the
TerseTS “CASH” (combined algorithm selection + hyperparameter optimization)
search space used in this repository.

High-level idea
---------------
The configuration is split into two parts:

1) **Discrete method selection** (treated as a multi-armed bandit):
     - ``logical_method_index``
     - ``coefficient_method_index``
     - ``indices_method_index``

2) **Continuous error bounds** (optimized with Bayesian optimization per arm):
     - ``logical_method_error``
     - ``coefficient_method_error``

Each *arm* corresponds to one discrete triple
``(logical_method_index, coefficient_method_index, indices_method_index)``.
For a chosen arm, the optimizer runs a small 2D Bayesian optimization loop over
the two continuous error hyperparameters.

Execution cycle (what ``maximize`` does)
---------------------------------------
Within the total evaluation budget ``init_points + n_iter``:

1) **Enumerate pipeline_arms** from the index bounds in ``search_space``.
2) **Warm start**: sample up to one random point per arm (until budget runs out).
3) **Adaptive loop**:
     - Score each arm using a simple UCB policy:
         ``UCB = best_y_arm + beta * sqrt(log(t+1)/(n_arm+1))``.
     - Pick the arm with the highest UCB.
     - Propose the next 2D point for that arm:
             - If the arm has <2 observations: sample uniformly at random.
             - Else: fit a local GP and maximize Log Expected Improvement.
             - If GP fitting/optimization fails: fall back to random.
4) Track and return the best observed configuration across all pipeline_arms.

Search-space expectations
-------------------------
``search_space`` is expected to be a mapping from parameter name to bounds.
Index parameters must be present as 2-tuples ``(lo, hi)`` and are treated as
inclusive integer ranges after rounding.

``logical_method_error`` and ``coefficient_method_error`` bounds are a single
``(lo, hi)`` tuple/list each, covering every method. The per-method conditional
dict form this optimizer once accepted was removed: only its ``"default"`` entry
was ever read, so a per-method range was silently discarded (see
``optimizer.space.parse_bounds_pair``).

Scaling
-------
If ``space_definition`` specifies ``scale: log`` for an error dimension, the GP
operates in log-space. Sampling is uniform in log-space and values are
exponentiated before being passed to the objective.

Notes
-----
- The objective is assumed to be **maximized** (larger is better).
- ``SuccessiveHalvingOptimizer`` and ``AdaptiveHalvingOptimizer`` subclass this
  class and reuse ``_sample``/``_arm_*_bounds``/``_evaluate_arm_candidate`` and
  the whole logging half, so those signatures are a contract, not local detail.
  Both define their own ``_propose_candidate_for_arm``; BOMAB's own proposal
  step is ``_propose_for_arm`` and is deliberately named differently.
"""

from __future__ import annotations

import math
import os
import time
import traceback
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Tuple

import numpy as np

from optimizer.space import parse_bounds_pair
from optimizer.arm_trace import (
    EVALUATION_CSV_FIELDS,
    ArmTraceLogging,
    _arm_components,
    _json_safe,
    _now_utc_iso,
    _sanitize_for_path,
)


def _run_id(log_run: Dict[str, Any] | None) -> str | None:
    """Run id for a trace row; ``None`` when logging is disabled."""
    return None if log_run is None else log_run["run_id"]


def _require_botorch():
    """Import optional BoTorch/GPyTorch dependencies.

    This optimizer is optional-dependency heavy. Importing inside the function
    keeps the rest of the repository usable without ``torch``/``botorch``.

    Returns
    -------
    dict
        A small dependency bundle with the required symbols.

    Raises
    ------
    ImportError
        If one of the optional dependencies is missing.
    """
    try:
        import torch
        from botorch.acquisition import LogExpectedImprovement
        from botorch.fit import fit_gpytorch_mll
        from botorch.models import SingleTaskGP
        from botorch.models.transforms import Normalize, Standardize
        from botorch.optim import optimize_acqf
        from gpytorch.mlls import ExactMarginalLogLikelihood

        return {
            "torch": torch,
            "LogExpectedImprovement": LogExpectedImprovement,
            "fit_gpytorch_mll": fit_gpytorch_mll,
            "SingleTaskGP": SingleTaskGP,
            "Normalize": Normalize,
            "Standardize": Standardize,
            "optimize_acqf": optimize_acqf,
            "ExactMarginalLogLikelihood": ExactMarginalLogLikelihood,
        }
    except Exception as exc:
        raise ImportError(
            "botorch_cash requires optional dependencies: torch, gpytorch, botorch. "
            "Install them to use this optimizer."
        ) from exc


def _resolve_torch_device(torch, requested_device: str):
    """Resolve the execution device for BoTorch tensors and models."""
    normalized = str(requested_device).lower()
    if normalized == "auto":
        normalized = "cuda" if torch.cuda.is_available() else "cpu"
    if normalized == "cuda" and not torch.cuda.is_available():
        print("Warning: CUDA requested for bomab, but no GPU is available. Falling back to CPU.")
        normalized = "cpu"
    if normalized not in {"cpu", "cuda"}:
        raise ValueError(
            f"Unsupported device '{requested_device}' for BOMABOptimizer. Expected one of: auto, cpu, cuda."
        )
    return torch.device(normalized)





UCB_CSV_FIELDS = [
    "run_id",
    "evaluation",
    "timestamp_utc",
    "rank",
    "selected",
    "arm_label",
    "logical_method_index",
    "coefficient_method_index",
    "indices_method_index",
    "logical_method_name",
    "coefficient_method_name",
    "indices_method_name",
    "num_observations",
    "best_reward",
    "exploit",
    "explore",
    "ucb_score",
]



class BOMABOptimizer(ArmTraceLogging):
    """Bandit allocation over per-arm Bayesian optimization.

    Parameters
    ----------
    init_points:
        Warm-start budget. The implementation attempts to evaluate up to one
        random point per arm first, stopping early if the budget is exhausted.
    n_iter:
        Additional adaptive iterations after the warm start.
    beta:
        Exploration strength for the UCB arm-selection policy.
    num_restarts, raw_samples:
        BoTorch acquisition optimizer settings.
    verbose:
        If > 0, prints whenever a new global best is found.
    alpha:
        Reserved for future variants; currently unused (kept for config
        compatibility).
    random_state:
        Run identifier kept for config compatibility and log metadata. Random
        number generators are expected to be seeded by the experiment runner.
    log_dir:
        Root directory for structured optimization logs. Set to ``None`` or
        disable ``log_process`` to turn file logging off.
    """

    def __init__(
        self,
        init_points: int = 20,
        n_iter: int = 60,
        beta: float = 0.75,
        num_restarts: int = 5,
        raw_samples: int = 32,
        verbose: int = 1,
        alpha: float = 0.75,
        random_state: int = 32,
        device: str = "auto",
        log_dir: str | None = ".logs",
        log_subdir: str = "_mab_opt",
        log_process: bool = True,
        log_ucb_scores: bool = True,
        log_flush_every: int = 64,
    ):
        self.init_points = int(init_points)
        self.n_iter = int(n_iter)
        self.total_budget = self.init_points + self.n_iter
        self.beta = float(beta)
        self.num_restarts = int(num_restarts)
        self.raw_samples = int(raw_samples)
        self.verbose = int(verbose)
        self.alpha = float(alpha)
        self.random_state = int(random_state)
        self.device = str(device)
        self.log_dir = log_dir
        self.log_subdir = str(log_subdir)
        self.log_process = bool(log_process)
        self.log_ucb_scores = bool(log_ucb_scores)
        self.log_flush_every = max(1, int(log_flush_every))
        self.rng = np.random.default_rng(self.random_state)

    def _start_log_run(
        self,
        *,
        log_dir: str | None,
        run_metadata: Dict[str, Any] | None,
        search_space: Dict[str, Any],
        space_definition: Dict[str, Any] | None,
        methods: List[str],
        pipeline_arms: List[Tuple[int, int, int]],
        total_budget: int,
        logical_scale: str,
        coeff_scale: str,
        device: Any,
    ) -> Dict[str, Any] | None:
        if not self.log_process:
            return None

        selected_log_dir = self.log_dir if log_dir is None else log_dir
        if selected_log_dir is None or str(selected_log_dir).strip() == "":
            return None

        if not isinstance(run_metadata, dict):
            raise ValueError("run_metadata is required when optimization logging is enabled.")
        metadata = _json_safe(run_metadata)
        dataset_name = _sanitize_for_path(metadata["dataset"])
        model_name = _sanitize_for_path(metadata["model"])
        alpha_name = f"alpha_{self.alpha:g}"
        fold_value = metadata["fold"]
        fold_name = f"fold_{_sanitize_for_path(fold_value)}"
        timestamp_for_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
        run_id = f"{dataset_name}_{model_name}_{alpha_name}_{fold_name}_rs{self.random_state}_{timestamp_for_id}"

        root_dir = str(selected_log_dir)
        subdir = self.log_subdir if self.log_subdir else "_mab_opt"
        # Model is part of the path (not just metadata.json) so two models
        # sharing (dataset, optimizer, budget, fold) - e.g. classification's
        # proximity_forest + tsfresh - can never clobber each other's trace.
        # SuccessiveHalvingOptimizer subclasses BOMABOptimizer and reuses this
        # method unchanged, so the fix applies to both optimizers at once.
        # alpha is also a path segment (like budget) so re-running the same
        # config at a different task-vs-compression tradeoff lands in a
        # sibling directory instead of overwriting the previous trace.
        run_dir = os.path.join(root_dir, subdir, dataset_name, model_name, alpha_name, f"budget_{total_budget}", fold_name)
        os.makedirs(run_dir, exist_ok=True)
        log_run = {
            "run_id": run_id,
            "run_dir": run_dir,
            "started_at": _now_utc_iso(),
            "start_monotonic": time.perf_counter(),
            "flush_every": self.log_flush_every,
            "buffers": {
                "jsonl": {},
                "csv": {},
            },
            "csv_headers_written": {},
            "paths": {
                "metadata": os.path.join(run_dir, "metadata.json"),
                "events": os.path.join(run_dir, "events.jsonl"),
                "evaluations_jsonl": os.path.join(run_dir, "evaluations.jsonl"),
                "evaluations_csv": os.path.join(run_dir, "evaluations.csv"),
                "ucb_jsonl": os.path.join(run_dir, "ucb_scores.jsonl"),
                "ucb_csv": os.path.join(run_dir, "ucb_scores.csv"),
                "arms_csv": os.path.join(run_dir, "arms.csv"),
                "arm_summary_csv": os.path.join(run_dir, "arm_summary.csv"),
                "summary": os.path.join(run_dir, "summary.json"),
            },
        }
        for path in log_run["paths"].values():
            if os.path.exists(path):
                os.remove(path)
        self._write_json(
            log_run,
            "metadata",
            {
                "run_id": run_id,
                "started_at": log_run["started_at"],
                "cache": {
                    "path_schema": "_mab_opt/dataset_name/model_name/alpha_x/budget_x/fold_x",
                    "dataset": dataset_name,
                    "fold": fold_value,
                    "cache_key": {
                        "task": metadata["task"],
                        "dataset": metadata["dataset"],
                        "fold": metadata["fold"],
                        "model": metadata["model"],
                        "compressor": metadata["compressor"],
                        "optimizer": metadata["optimizer"],
                        "random_state": self.random_state,
                    },
                },
                "run_metadata": metadata,
                "optimizer": {
                    "name": "bomab",
                    "init_points": self.init_points,
                    "n_iter": self.n_iter,
                    "total_budget": total_budget,
                    "beta": self.beta,
                    "num_restarts": self.num_restarts,
                    "raw_samples": self.raw_samples,
                    "alpha": self.alpha,
                    "random_state": self.random_state,
                    "device": str(device),
                    "log_ucb_scores": self.log_ucb_scores,
                    "log_flush_every": self.log_flush_every,
                },
                "search_space": search_space,
                "space_definition": space_definition,
                "scales": {
                    "logical_method_error": logical_scale,
                    "coefficient_method_error": coeff_scale,
                },
                "methods": methods,
                "num_pipeline_arms": len(pipeline_arms),
                "pipeline_arms": pipeline_arms,
                "files": log_run["paths"],
            },
        )
        self._log_event(
            log_run,
            {
                "event": "optimization_started",
                "total_budget": total_budget,
                "num_pipeline_arms": len(pipeline_arms),
            },
        )
        if self.verbose > 0:
            print(f"[bomab] logging optimization trace to {run_dir}")
        return log_run
    
    def _log_ucb_scores(
        self,
        log_run: Dict[str, Any] | None,
        *,
        evaluation: int,
        methods: List[str],
        selected_arm: Tuple[int, int, int],
        ranked_rows: List[Dict[str, Any]],
    ) -> None:
        if not self.log_ucb_scores:
            return
        timestamp_utc = _now_utc_iso()
        json_rows = []
        for rank, row in enumerate(ranked_rows, start=1):
            arm = row["arm"]
            csv_row = {
                "run_id": _run_id(log_run),
                "evaluation": evaluation,
                "timestamp_utc": timestamp_utc,
                "rank": rank,
                "selected": arm == selected_arm,
                "num_observations": row["num_observations"],
                "best_reward": row["best_reward"],
                "exploit": row["exploit"],
                "explore": row["explore"],
                "ucb_score": row["ucb_score"],
            }
            csv_row.update(_arm_components(arm, methods))
            self._append_csv(log_run, "ucb_csv", csv_row, UCB_CSV_FIELDS)
            json_rows.append(csv_row)
        self._append_jsonl(
            log_run,
            "ucb_jsonl",
            {
                "run_id": _run_id(log_run),
                "evaluation": evaluation,
                "timestamp_utc": timestamp_utc,
                "selected_arm": list(selected_arm),
                "scores": json_rows,
            },
        )

    def _write_final_summary(
        self,
        log_run: Dict[str, Any] | None,
        *,
        state: Dict[str, Any],
        total_budget: int,
        status: str,
        error: str | None = None,
    ) -> None:
        self._flush_log_buffers(log_run, force=True)
        self._write_json(
            log_run,
            "summary",
            {
                "run_id": _run_id(log_run),
                "status": status,
                "completed_at": _now_utc_iso(),
                "elapsed_sec": None
                if log_run is None
                else round(time.perf_counter() - log_run["start_monotonic"], 6),
                "evaluations": state["evaluations"],
                "total_budget": total_budget,
                "best_reward": state["best_y"],
                "best_params": state["best_x"],
                "error": error,
            },
        )

    def _sample(self, rng: Any, lower_bound: float, upper_bound: float, scale: str) -> float:
        """Sample a value in *compression* space from the provided bounds.
        This is uniform in log-space if scale=="log", else uniform in linear space.
        For example, if lower_bound=0.01 and upper_bound=1.0, and scale is log, 
        the function uniformly samples between -4.605 and 0 in log-space and returns the exponentiated value.
        Otherwise, it samples uniformly between 0.01 and 1.0 in linear space.
        """
        if scale == "log":
            return float(math.exp(rng.uniform(math.log(lower_bound), math.log(upper_bound))))
        return float(rng.uniform(lower_bound, upper_bound))

    def _to_model_space(self, value: float, scale: str) -> float:
        """Map a compression-space value to model space (log if requested)."""
        return float(math.log(value)) if scale == "log" else float(value)

    def _to_compression_space(self, value: float, scale: str) -> float:
        """Map a model-space value back to compression space (exp if requested)."""
        return float(math.exp(value)) if scale == "log" else float(value)

    def _arm_error_bounds(
        self,
        _arm: Tuple[int, int, int],
        _methods: List[str],
        search_space: Dict[str, Any],
    ) -> Dict[str, Tuple[float, float]]:
        """Return raw-space error bounds for the given arm.

        ``_arm``/``_methods`` are unused and kept only to preserve the arm-aware
        call convention shared with successive_halving/adaptive_halving/random
        (all eleven call sites pass them positionally). The error ranges are no
        longer conditional on which method the arm selected — that dict form
        silently discarded its per-method entries and was removed; see
        ``optimizer.space.parse_bounds_pair``.
        """
        logical_lo, logical_hi = parse_bounds_pair(
            search_space["logical_method_error"], "logical_method_error"
        )
        coefficient_lo, coefficient_hi = parse_bounds_pair(
            search_space["coefficient_method_error"], "coefficient_method_error"
        )
        return {
            "logical_method_error": (logical_lo, logical_hi),
            "coefficient_method_error": (coefficient_lo, coefficient_hi),
        }

    def _arm_model_bounds(
        self,
        arm_raw_bounds: Dict[str, Tuple[float, float]],
        logical_scale: str,
        coeff_scale: str,
    ) -> Dict[str, Tuple[float, float]]:
        """Return per-arm bounds in model space."""
        logical_lo, logical_hi = arm_raw_bounds["logical_method_error"]
        coeff_lo, coeff_hi = arm_raw_bounds["coefficient_method_error"]
        return {
            "logical_method_error": (
                self._to_model_space(logical_lo, logical_scale),
                self._to_model_space(logical_hi, logical_scale),
            ),
            "coefficient_method_error": (
                self._to_model_space(coeff_lo, coeff_scale),
                self._to_model_space(coeff_hi, coeff_scale),
            ),
        }

    def _random_arm_candidate(
        self,
        arm_raw_bounds: Dict[str, Tuple[float, float]],
        logical_scale: str,
        coeff_scale: str,
    ) -> Dict[str, float]:
        """Sample both error dimensions uniformly (scale-aware) inside the arm's bounds."""
        return {
            "logical_method_error": self._sample(
                self.rng, *arm_raw_bounds["logical_method_error"], logical_scale
            ),
            "coefficient_method_error": self._sample(
                self.rng, *arm_raw_bounds["coefficient_method_error"], coeff_scale
            ),
        }

    def _ucb_scores(
        self,
        pipeline_arms: List[Tuple[int, int, int]],
        history: Dict[Tuple[int, int, int], Dict[str, Any]],
        t: int,
    ) -> List[Dict[str, Any]]:
        """Score every arm: ``best_y_arm + beta * sqrt(log(t+1) / (n_arm+1))``.

        Scoring all arms each iteration (rather than tracking a heap) keeps the
        UCB trace complete - ``ucb_scores.csv`` gets a row per arm per
        evaluation, which is what the arm-allocation plots read.
        """
        rows = []
        for arm in pipeline_arms:
            num_observations = len(history[arm]["y"])
            exploit = history[arm]["best"]
            explore = self.beta * math.sqrt(math.log(t + 1.0) / (num_observations + 1.0))
            rows.append(
                {
                    "arm": arm,
                    "num_observations": num_observations,
                    "best_reward": exploit,
                    "exploit": exploit,
                    "explore": explore,
                    "ucb_score": exploit + explore,
                }
            )
        return rows

    def _propose_for_arm(
        self,
        arm: Tuple[int, int, int],
        arm_state: Dict[str, Any],
        arm_raw_bounds: Dict[str, Tuple[float, float]],
        logical_scale: str,
        coeff_scale: str,
    ) -> Tuple[Dict[str, float], str, str | None]:
        """Propose the next 2D point for ``arm``: GP + LogEI, else random.

        Returns ``(candidate, proposal_source, fallback_reason)``. This is
        BOMAB's own proposal step and deliberately *not* named
        ``_propose_candidate_for_arm`` - successive_halving and
        adaptive_halving each define a method by that name returning a
        2-tuple, and entangling their contract with the ``fallback_reason``
        BOMAB logs would make one of the three wrong.

        Random is used in two distinct situations, kept distinguishable in the
        trace: fewer than two observations (a GP cannot be fit at all), and a
        genuine numerical failure of the fit/acquisition step.
        """
        random_candidate = self._random_arm_candidate(arm_raw_bounds, logical_scale, coeff_scale)
        if len(arm_state["y"]) < 2:
            return random_candidate, "random_insufficient_arm_data", None

        gp = self._gp_context
        torch = gp["torch"]
        dtype = gp["dtype"]
        device = gp["device"]
        arm_model_bounds = self._arm_model_bounds(arm_raw_bounds, logical_scale, coeff_scale)
        model_bounds_t = torch.tensor(
            [
                [
                    arm_model_bounds["logical_method_error"][0],
                    arm_model_bounds["coefficient_method_error"][0],
                ],
                [
                    arm_model_bounds["logical_method_error"][1],
                    arm_model_bounds["coefficient_method_error"][1],
                ],
            ],
            dtype=dtype,
            device=device,
        )
        # Train data lives in model space so that log scaling behaves well.
        train_X = torch.tensor(np.asarray(arm_state["X"]), dtype=dtype, device=device)
        train_Y = torch.tensor(np.asarray(arm_state["y"]).reshape(-1, 1), dtype=dtype, device=device)
        if torch.isclose(train_Y.var(), torch.zeros(1, dtype=dtype, device=device)).item():
            # Standardize(m=1) divides by this variance - a GP is not merely
            # inaccurate here, it is undefined.
            return random_candidate, "random_gp_fallback", "Arm observations are constant."

        try:
            model = gp["SingleTaskGP"](
                train_X,
                train_Y,
                # Explicit search-space bounds, not data-inferred: SingleTaskGP's
                # default lengthscale prior assumes inputs in [0,1]^d.
                input_transform=gp["Normalize"](d=2, bounds=model_bounds_t),
                outcome_transform=gp["Standardize"](m=1),
            )
            mll = gp["ExactMarginalLogLikelihood"](model.likelihood, model)
            gp["fit_gpytorch_mll"](mll)

            # Acquisition is computed against this arm's best observed value.
            acq = gp["LogExpectedImprovement"](model=model, best_f=float(train_Y.max().item()))
            cand, _ = gp["optimize_acqf"](
                acq_function=acq,
                bounds=model_bounds_t,
                q=1,
                num_restarts=self.num_restarts,
                raw_samples=self.raw_samples,
            )
        except Exception as exc:
            # A GP fit is a numerical procedure that can legitimately fail on
            # ill-conditioned data; falling back to random keeps the budget
            # spent on real evaluations. This is not a swallowed config error -
            # the reason is logged and lands in the trace.
            print(f"Warning: GP fit/acquisition failed for arm {arm}. Falling back to random sampling.")
            return random_candidate, "random_gp_fallback", f"{type(exc).__name__}: {exc}"

        candidate = {
            "logical_method_error": self._to_compression_space(float(cand[0, 0].item()), logical_scale),
            "coefficient_method_error": self._to_compression_space(float(cand[0, 1].item()), coeff_scale),
        }
        return candidate, "log_expected_improvement", None

    def _evaluate_arm_candidate(
        self,
        *,
        objective: Callable[[Dict[str, float]], float],
        arm: Tuple[int, int, int],
        candidate_parameter: Dict[str, float],
        methods: List[str],
        logical_scale: str,
        coeff_scale: str,
        history: Dict[Tuple[int, int, int], Dict[str, Any]],
        state: Dict[str, Any],
        log_run: Dict[str, Any] | None = None,
        phase: str = "unknown",
        proposal_source: str = "unknown",
        selection_context: Dict[str, Any] | None = None,
        fallback_reason: str | None = None,
    ) -> float:
        """Evaluate a model-space candidate, update per-arm history and global best."""
        candidate_compression_space = {
            "logical_method_index": int(arm[0]),
            "coefficient_method_index": int(arm[1]),
            "indices_method_index": int(arm[2]),
            "logical_method_error": candidate_parameter["logical_method_error"],
            "coefficient_method_error": candidate_parameter["coefficient_method_error"],
        }

        arm_state = history[arm]
        evaluation = state["evaluations"] + 1
        arm_observations_before = len(arm_state["y"])
        arm_best_before = arm_state["best"]
        global_best_before = state["best_y"]
        candidate_model_space = {
            "logical_method_error": self._to_model_space(
                candidate_parameter["logical_method_error"],
                logical_scale,
            ),
            "coefficient_method_error": self._to_model_space(
                candidate_parameter["coefficient_method_error"],
                coeff_scale,
            ),
        }
        selection_context = selection_context or {}

        start = time.perf_counter()
        self._log_event(
            log_run,
            {
                "event": "evaluation_started",
                "evaluation": evaluation,
                "phase": phase,
                "proposal_source": proposal_source,
                "arm": _arm_components(arm, methods),
                "candidate": candidate_compression_space,
                "candidate_model_space": candidate_model_space,
                "selection_context": selection_context,
                "fallback_reason": fallback_reason,
            },
        )
        try:
            score = float(objective(candidate_compression_space))
        except Exception as exc:
            elapsed_sec = time.perf_counter() - start
            self._log_event(
                log_run,
                {
                    "event": "evaluation_failed",
                    "evaluation": evaluation,
                    "phase": phase,
                    "proposal_source": proposal_source,
                    "arm": _arm_components(arm, methods),
                    "candidate": candidate_compression_space,
                    "candidate_model_space": candidate_model_space,
                    "objective_elapsed_sec": round(elapsed_sec, 6),
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                    "traceback": traceback.format_exc(),
                    "selection_context": selection_context,
                    "fallback_reason": fallback_reason,
                },
            )
            raise
        elapsed_sec = time.perf_counter() - start

        # Per-arm BO remains 2D: only lossy error bounds are optimized.
        arm_state["X"].append([
            candidate_model_space["logical_method_error"],
            candidate_model_space["coefficient_method_error"],
        ])
        arm_state["y"].append(score)
        arm_state["best"] = max(arm_state["best"], score)

        state["evaluations"] += 1
        is_new_global_best = score > state["best_y"]
        if is_new_global_best:
            state["best_y"] = score
            state["best_x"] = candidate_compression_space
            if self.verbose > 0:
                print(f"[bomab] new best={state['best_y']:.5f} arm={arm}")

        evaluation_row = {
            "run_id": _run_id(log_run),
            "evaluation": evaluation,
            "timestamp_utc": _now_utc_iso(),
            "phase": phase,
            "proposal_source": proposal_source,
            "logical_method_error": candidate_compression_space["logical_method_error"],
            "coefficient_method_error": candidate_compression_space["coefficient_method_error"],
            "logical_method_error_model_space": candidate_model_space["logical_method_error"],
            "coefficient_method_error_model_space": candidate_model_space["coefficient_method_error"],
            "reward": score,
            "objective_elapsed_sec": round(elapsed_sec, 6),
            "arm_observations_before": arm_observations_before,
            "arm_observations_after": len(arm_state["y"]),
            "arm_best_before": arm_best_before,
            "arm_best_after": arm_state["best"],
            "global_best_before": global_best_before,
            "global_best_after": state["best_y"],
            "is_new_global_best": is_new_global_best,
            "ucb_score": selection_context.get("ucb_score"),
            "ucb_exploit": selection_context.get("exploit"),
            "ucb_explore": selection_context.get("explore"),
            "ucb_rank": selection_context.get("rank"),
            "fallback_reason": fallback_reason,
        }
        evaluation_row.update(_arm_components(arm, methods))
        self._append_jsonl(log_run, "evaluations_jsonl", evaluation_row)
        self._append_csv(log_run, "evaluations_csv", evaluation_row, EVALUATION_CSV_FIELDS)
        self._log_event(
            log_run,
            {
                "event": "evaluation_completed",
                "evaluation": evaluation,
                "phase": phase,
                "proposal_source": proposal_source,
                "reward": score,
                "objective_elapsed_sec": round(elapsed_sec, 6),
                "is_new_global_best": is_new_global_best,
                "global_best_after": state["best_y"],
                "arm": _arm_components(arm, methods),
                "candidate": candidate_compression_space,
                "selection_context": selection_context,
                "fallback_reason": fallback_reason,
            },
        )

        return score

    def maximize(
        self,
        objective: Callable[[Dict[str, float]], float],
        search_space: Dict[str, Any],
        space_definition: Dict[str, Any],
        log_dir: str | None,
        run_metadata: Dict[str, Any] | None,
        **_,
    ) -> Dict[str, float]:
        """Run the BOMAB optimization loop and return the best configuration.

        Parameters
        ----------
        objective:
            Callable that accepts a configuration dict and returns a scalar to
            **maximize**. This is expected to be the usual objective wrapper used
            in this repository; BOMAB additionally reads ``objective.backend._methods``
            to resolve conditional bounds.
        search_space:
            Parameter bounds. See the module docstring for the supported shapes.
            Required keys are:

            - ``logical_method_index``
            - ``coefficient_method_index``
            - ``indices_method_index``
            - ``logical_method_error``
            - ``coefficient_method_error``

        space_definition:
            Optional metadata for parameters. Only ``scale`` is used here for
            ``*_method_error`` (``linear`` or ``log``).

        Returns
        -------
        dict
            Best observed configuration in *raw* space.
        """
        deps = _require_botorch()
        torch = deps["torch"]
        device = _resolve_torch_device(torch, self.device)
        # Bundle handed to _propose_for_arm, mirroring adaptive_halving.
        self._gp_context = dict(deps, device=device, dtype=torch.double)

        methods: List[str] = objective.backend._methods

        # Arms are the cross-product of the three index ranges (inclusive
        # integer ranges after rounding).
        index_ranges = [
            range(int(round(lo)), int(round(hi)) + 1)
            for lo, hi in (
                search_space["logical_method_index"],
                search_space["coefficient_method_index"],
                search_space["indices_method_index"],
            )
        ]
        pipeline_arms = [(i, j, k) for i in index_ranges[0] for j in index_ranges[1] for k in index_ranges[2]]

        # Continuous dimensions may be optimized in linear space or log space.
        logical_scale = str(space_definition["logical_method_error"]["scale"])
        coeff_scale = str(space_definition["coefficient_method_error"]["scale"])
        if logical_scale not in ("linear", "log") or coeff_scale not in ("linear", "log"):
            raise ValueError("Only linear/log scales are supported for *_method_error in BOMABOptimizer.")

        # Total number of objective calls (warm start + adaptive iterations).
        total_budget = self.total_budget

        log_run = self._start_log_run(
            log_dir=log_dir,
            run_metadata=run_metadata,
            search_space=search_space,
            space_definition=space_definition,
            methods=methods,
            pipeline_arms=pipeline_arms,
            total_budget=total_budget,
            logical_scale=logical_scale,
            coeff_scale=coeff_scale,
            device=device,
        )
        self.last_run_dir = log_run["run_dir"] if log_run else None
        self._log_arm_inventory(
            log_run,
            pipeline_arms,
            methods,
            search_space,
            logical_scale,
            coeff_scale,
        )

        # Per-arm history:
        # - X is always 2D in *model space* (log-transformed if configured)
        # - y is objective values (higher is better)
        # - best is the best y seen for that arm
        history: Dict[Tuple[int, int, int], Dict[str, Any]] = {
            arm: {"X": [], "y": [], "best": float("-inf")} for arm in pipeline_arms
        }

        # Global incumbent across all pipeline_arms (mutated by _evaluate_arm_candidate).
        state: Dict[str, Any] = {
            "best_x": None,
            "best_y": float("-inf"),
            "evaluations": 0,
        }

        # Warm start: one random sample per arm, until the budget runs out.
        # With more arms than budget this consumes the whole run, so the shuffle
        # matters: without it, which arms get sampled at all would be decided by
        # method-list order rather than by the seed.
        warm_pipeline_arms = list(pipeline_arms)
        self.rng.shuffle(warm_pipeline_arms)
        for arm in warm_pipeline_arms:
            if state["evaluations"] >= total_budget:
                break
            arm_raw_bounds = self._arm_error_bounds(arm, methods, search_space)
            self._evaluate_arm_candidate(
                objective=objective,
                arm=arm,
                candidate_parameter=self._random_arm_candidate(
                    arm_raw_bounds, logical_scale, coeff_scale
                ),
                methods=methods,
                logical_scale=logical_scale,
                coeff_scale=coeff_scale,
                history=history,
                state=state,
                log_run=log_run,
                phase="warm_start",
                proposal_source="random_warm_start",
            )

        # Adaptive allocation: pick an arm by UCB, propose one point inside it.
        while state["evaluations"] < total_budget:
            print(f"[bomab] evaluation {state['evaluations']}/{total_budget} - scoring arms...")
            ucb_rows = self._ucb_scores(pipeline_arms, history, state["evaluations"] + 1)
            selected_index = int(np.argmax([row["ucb_score"] for row in ucb_rows]))
            arm = pipeline_arms[selected_index]

            ranked_ucb_rows = sorted(ucb_rows, key=lambda row: row["ucb_score"], reverse=True)
            selection_context = dict(ucb_rows[selected_index])
            selection_context["rank"] = 1 + ranked_ucb_rows.index(ucb_rows[selected_index])
            self._log_ucb_scores(
                log_run,
                evaluation=state["evaluations"] + 1,
                methods=methods,
                selected_arm=arm,
                ranked_rows=ranked_ucb_rows,
            )

            arm_raw_bounds = self._arm_error_bounds(arm, methods, search_space)
            candidate_parameter, proposal_source, fallback_reason = self._propose_for_arm(
                arm, history[arm], arm_raw_bounds, logical_scale, coeff_scale
            )
            if fallback_reason is not None:
                self._log_event(
                    log_run,
                    {
                        "event": "proposal_fallback",
                        "evaluation": state["evaluations"] + 1,
                        "arm": _arm_components(arm, methods),
                        "selection_context": selection_context,
                        "fallback_reason": fallback_reason,
                    },
                )

            self._evaluate_arm_candidate(
                objective=objective,
                arm=arm,
                candidate_parameter=candidate_parameter,
                methods=methods,
                logical_scale=logical_scale,
                coeff_scale=coeff_scale,
                history=history,
                state=state,
                log_run=log_run,
                phase="adaptive",
                proposal_source=proposal_source,
                selection_context=selection_context,
                fallback_reason=fallback_reason,
            )

        # A failed objective evaluation raises out of _evaluate_arm_candidate,
        # so reaching this line means best_x is set.
        self._write_arm_summary(
            log_run,
            history=history,
            pipeline_arms=pipeline_arms,
            methods=methods,
            search_space=search_space,
            logical_scale=logical_scale,
            coeff_scale=coeff_scale,
        )
        self._write_final_summary(
            log_run,
            state=state,
            total_budget=total_budget,
            status="completed",
        )
        self._log_event(
            log_run,
            {
                "event": "optimization_completed",
                "evaluations": state["evaluations"],
                "best_reward": state["best_y"],
                "best_params": state["best_x"],
            },
        )
        self._flush_log_buffers(log_run, force=True)
        return state["best_x"]
