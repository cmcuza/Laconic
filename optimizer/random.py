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

# Random search allocates no bandit budget and has no acquisition to fall back
# from, so the UCB and fallback columns of the shared arm schema would be blank
# in every row it ever writes. It declares the subset it actually fills; the
# analysis loader picks columns up by name, so a narrower header is fine.
RANDOM_EVALUATION_CSV_FIELDS = [
    field
    for field in EVALUATION_CSV_FIELDS
    if field
    not in {"ucb_score", "ucb_exploit", "ucb_explore", "ucb_rank", "fallback_reason"}
]





class RandomOptimizer(ArmTraceLogging):
    def __init__(
        self,
        n_iter: int,
        verbose: int,
        alpha: float,
        random_state: int,
        log_dir: str | None = ".logs",
        log_subdir: str = "_random_opt",
        log_process: bool = True,
        log_flush_every: int = 64,
    ):
        self.n_iter = n_iter
        self.total_budget = self.n_iter
        self.verbose = verbose
        self.alpha = alpha
        self.random_state = random_state
        self.log_dir = log_dir
        self.log_subdir = str(log_subdir)
        self.log_process = bool(log_process)
        self.log_flush_every = max(1, int(log_flush_every))

    # ------------------------------------------------------------------
    # Logging helpers (adapted from BOMABOptimizer)
    # ------------------------------------------------------------------

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
        subdir = self.log_subdir if self.log_subdir else "_random_opt"
        # Model is part of the path (not just metadata.json) so two models
        # sharing (dataset, optimizer, budget, fold) - e.g. classification's
        # proximity_forest + tsfresh - can never clobber each other's trace.
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
                    "path_schema": "_random_opt/dataset_name/model_name/alpha_x/budget_x/fold_x",
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
                    "name": "random",
                    "n_iter": self.n_iter,
                    "total_budget": total_budget,
                    "alpha": self.alpha,
                    "random_state": self.random_state,
                    "device": str(device),
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
            print(f"[random] logging optimization trace to {run_dir}")
        return log_run

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
                "run_id": None if log_run is None else log_run["run_id"],
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

    # ------------------------------------------------------------------
    # Search-space helpers
    # ------------------------------------------------------------------

    def _sample(self, rng: Any, lower_bound: float, upper_bound: float, scale: str) -> float:
        if scale == "log":
            return float(math.exp(rng.uniform(math.log(lower_bound), math.log(upper_bound))))
        return float(rng.uniform(lower_bound, upper_bound))

    def _to_model_space(self, value: float, scale: str) -> float:
        return float(math.log(value)) if scale == "log" else float(value)

    def _arm_error_bounds(
        self,
        _arm: Tuple[int, int, int],
        _methods: List[str],
        search_space: Dict[str, Any],
    ) -> Dict[str, Tuple[float, float]]:
        # `_arm`/`_methods` are unused and kept only to preserve the arm-aware
        # call convention shared with the halving optimizers; the error ranges
        # are no longer conditional on the arm's method (see parse_bounds_pair).
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
    ) -> float:
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
                candidate_parameter["logical_method_error"], logical_scale
            ),
            "coefficient_method_error": self._to_model_space(
                candidate_parameter["coefficient_method_error"], coeff_scale
            ),
        }

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
                },
            )
            raise
        elapsed_sec = time.perf_counter() - start

        arm_state["y"].append(score)
        arm_state["best"] = max(arm_state["best"], score)

        state["evaluations"] += 1
        is_new_global_best = score > state["best_y"]
        if score > state["best_y"]:
            state["best_y"] = score
            state["best_x"] = candidate_compression_space
            if self.verbose > 0:
                print(f"[random] new best={state['best_y']:.5f} arm={arm}")

        evaluation_row = {
            "run_id": None if log_run is None else log_run["run_id"],
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
        }
        evaluation_row.update(_arm_components(arm, methods))
        self._append_jsonl(log_run, "evaluations_jsonl", evaluation_row)
        self._append_csv(log_run, "evaluations_csv", evaluation_row, RANDOM_EVALUATION_CSV_FIELDS)
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
        """
        search_space: dict param→(lo,hi)
        space_definition: dict param→{type,scale[,significant_digits]}
        """
        methods: List[str] = objective.backend._methods

        # Discrete method-selection bounds (inclusive integer ranges after rounding).

        logical_index_lo, logical_index_hi = search_space["logical_method_index"]
        coefficient_index_lo, coefficient_index_hi = search_space["coefficient_method_index"]
        indices_index_lo, indices_index_hi = search_space["indices_method_index"]

        logical_index_range = range(int(round(logical_index_lo)), int(round(logical_index_hi)) + 1)
        coefficient_index_range = range(int(round(coefficient_index_lo)), int(round(coefficient_index_hi)) + 1)
        indices_index_range = range(int(round(indices_index_lo)), int(round(indices_index_hi)) + 1)

        # Create all pipeline_arms as the cross-product of the logical/coefficient/indices method index ranges.
        pipeline_arms = [(i, j, k) for i in logical_index_range for j in coefficient_index_range for k in indices_index_range]

        logical_scale = str(space_definition["logical_method_error"]['scale'])
        coeff_scale = str(space_definition["coefficient_method_error"]['scale'])
        if logical_scale not in ("linear", "log") or coeff_scale not in ("linear", "log"):
            raise ValueError("Only linear/log scales are supported for *_method_error in RandomOptimizer.")

        rng = np.random.default_rng(self.random_state)
        arm_indices = rng.choice(len(pipeline_arms), size=self.n_iter, replace=True)
        warm_pipeline_arms = [pipeline_arms[i] for i in arm_indices]

        log_run = self._start_log_run(
            log_dir=log_dir,
            run_metadata=run_metadata,
            search_space=search_space,
            space_definition=space_definition,
            methods=methods,
            pipeline_arms=pipeline_arms,
            total_budget=self.n_iter,
            logical_scale=logical_scale,
            coeff_scale=coeff_scale,
            device='cpu',
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
        history: Dict[Tuple[int, int, int], Dict[str, Any]] = {
            # "y"/"best" feed arm_summary.csv only - random search never reads
            # its own history back when choosing the next arm.
            arm: {"y": [], "best": float("-inf")} for arm in pipeline_arms
        }

        # Global incumbent across all pipeline_arms (mutated by _evaluate_arm_candidate).
        state: Dict[str, Any] = {
            "best_x": None,
            "best_y": float("-inf"),
            "evaluations": 0,
        }

        for arm in warm_pipeline_arms:
            arm_raw_bounds = self._arm_error_bounds(arm, methods, search_space)
            # Sample in raw space using the parameter scale (uniform in log-space when requested).

            logical_error = self._sample(
                rng,
                arm_raw_bounds["logical_method_error"][0],
                arm_raw_bounds["logical_method_error"][1],
                logical_scale,
            )
            coefficient_error = self._sample(
                rng,
                arm_raw_bounds["coefficient_method_error"][0],
                arm_raw_bounds["coefficient_method_error"][1],
                coeff_scale,
            )

            self._evaluate_arm_candidate(
                objective=objective,
                arm=arm,
                candidate_parameter={
                    "logical_method_error": logical_error,
                    "coefficient_method_error": coefficient_error,
                },
                methods=methods,
                logical_scale=logical_scale,
                coeff_scale=coeff_scale,
                history=history,
                state=state,
                log_run=log_run,
                phase="random_search",
                proposal_source="random",
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
            total_budget=self.n_iter,
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
