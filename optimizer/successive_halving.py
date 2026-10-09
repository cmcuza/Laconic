from __future__ import annotations

import csv
import json
import math
import os
import random as py_random
import re
import time
import traceback
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Tuple

import numpy as np
from optimizer.bomab import BOMABOptimizer

EVALUATION_CSV_FIELDS = [
    "run_id",
    "evaluation",
    "timestamp_utc",
    "phase",
    "proposal_source",
    "arm_label",
    "logical_method_index",
    "coefficient_method_index",
    "indices_method_index",
    "logical_method_name",
    "coefficient_method_name",
    "indices_method_name",
    "logical_method_error",
    "coefficient_method_error",
    "logical_method_error_model_space",
    "coefficient_method_error_model_space",
    "reward",
    "objective_elapsed_sec",
    "arm_observations_before",
    "arm_observations_after",
    "arm_best_before",
    "arm_best_after",
    "global_best_before",
    "global_best_after",
    "is_new_global_best",
    "ucb_score",
    "ucb_exploit",
    "ucb_explore",
    "ucb_rank",
    "cluster_id",
    "cluster_role",
    "fallback_reason",
]


def _arm_label(arm: Tuple[int, int, int], methods: List[str]) -> str:
    return f"L{arm[0]}:{methods[arm[0]]}|C{arm[1]}:{methods[arm[1]]}|I{arm[2]}:{methods[arm[2]]}"


def _arm_components(arm: Tuple[int, int, int], methods: List[str]) -> Dict[str, Any]:
    return {
        "arm": list(arm),
        "arm_label": _arm_label(arm, methods),
        "logical_method_index": int(arm[0]),
        "coefficient_method_index": int(arm[1]),
        "indices_method_index": int(arm[2]),
        "logical_method_name": methods[arm[0]],
        "coefficient_method_name": methods[arm[1]],
        "indices_method_name": methods[arm[2]],
    }

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

def _sample(lower_bound: float, upper_bound: float, scale: str) -> float:
    if scale == "log":
        return float(math.exp(np.random.uniform(math.log(lower_bound), math.log(upper_bound))))
    return float(np.random.uniform(lower_bound, upper_bound))

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


class SuccessiveHalvingOptimizer(BOMABOptimizer):
    """Successive Halving over per-arm Bayesian optimization.

    Parameters
    ----------
    total_budget:
        Total budget for the optimization process. 
        This is the total number of function evaluations allowed across all arms and iterations.
    n_clusters:
        This is the number of arms' clusters.
        Each cluster represents a group of arms that are evaluated together in each iteration of the successive halving process.
    verbose:
        If > 0, prints whenever a new global best is found.
    random_state:
        Seed used for NumPy and Torch.
    log_dir:
        Root directory for structured optimization logs. Set to ``None`` or
        disable ``log_process`` to turn file logging off.
    """

    def __init__(
        self,
        init_points: int = 20,
        n_iter: int = 80,
        beta: float = 0.75,
        num_restarts: int = 5,
        raw_samples: int = 32,
        verbose: int = 1,
        alpha: float = 0.75,
        random_state: int = 32,
        device: str = "auto",
        total_budget: int | None = None,
        log_dir: str | None = ".logs",
        log_subdir: str = "_successive_halving_opt",
        log_process: bool = True,
        log_ucb_scores: bool = True,
        log_flush_every: int = 64,
        physical_ucb_beta: float = 0.5,
        physical_novelty_bonus: float = 0.25,
        physical_softmax_temperature: float = 0.25,
    ):
        super().__init__(
            init_points=init_points,
            n_iter=n_iter,
            beta=beta,
            num_restarts=num_restarts,
            raw_samples=raw_samples,
            verbose=verbose,
            alpha=alpha,
            random_state=random_state,
            device=device,
            log_dir=log_dir,
            log_subdir=log_subdir,
            log_process=log_process,
            log_ucb_scores=log_ucb_scores,
            log_flush_every=log_flush_every,
        )
        computed_budget = self.init_points + self.n_iter
        self.total_budget = max(1, int(total_budget if total_budget is not None else computed_budget))
        self.physical_ucb_beta = float(physical_ucb_beta)
        self.physical_novelty_bonus = float(physical_novelty_bonus)
        self.physical_softmax_temperature = float(physical_softmax_temperature)

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
        # Reuse BOMAB logger plumbing but customize identity for this optimizer.
        original_verbose = self.verbose
        self.verbose = 0
        try:
            log_run = super()._start_log_run(
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
        finally:
            self.verbose = original_verbose

        if log_run is None:
            return None

        try:
            metadata_path = log_run["paths"]["metadata"]
            with open(metadata_path, "r", encoding="utf-8") as f:
                metadata = json.load(f)
            metadata.setdefault("optimizer", {})["name"] = "successive_halving"
            metadata["optimizer"]["total_budget"] = total_budget
            metadata["optimizer"]["init_points"] = self.init_points
            metadata["optimizer"]["n_iter"] = self.n_iter
            metadata["optimizer"]["beta"] = self.beta
            metadata["optimizer"]["alpha"] = self.alpha
            metadata["optimizer"]["num_restarts"] = self.num_restarts
            metadata["optimizer"]["raw_samples"] = self.raw_samples
            metadata["optimizer"]["physical_ucb_beta"] = self.physical_ucb_beta
            metadata["optimizer"]["physical_novelty_bonus"] = self.physical_novelty_bonus
            metadata["optimizer"]["physical_softmax_temperature"] = self.physical_softmax_temperature
            metadata["optimizer"]["random_state"] = self.random_state
            metadata["optimizer"]["device"] = str(device)
            with open(metadata_path, "w", encoding="utf-8") as f:
                json.dump(metadata, f, indent=2, sort_keys=True)
                f.write("\n")
        except Exception as exc:
            print(f"Warning: could not update SuccessiveHalving metadata identity: {exc}")

        if self.verbose > 0:
            print(f"[successive_halving] logging optimization trace to {log_run['run_dir']}")
        return log_run
    
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
        if score > state["best_y"]:
            state["best_y"] = score
            state["best_x"] = candidate_compression_space
            if self.verbose > 0:
                print(f"[botorch_cash] new best={state['best_y']:.5f} arm={arm}")

        evaluation_row = {
            "run_id": None if log_run is None else log_run["run_id"],
            "evaluation": evaluation,
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
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
            "cluster_id": selection_context.get("cluster_id"),
            "cluster_role": selection_context.get("cluster_role"),
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

    def _best_evaluated_arm_in_cluster(
        self,
        cluster: List[Tuple[int, int, int]],
        history: Dict[Tuple[int, int, int], Dict[str, Any]],
    ) -> Tuple[int, int, int] | None:
        """Return the arm with the highest observed reward in this cluster."""
        best_arm: Tuple[int, int, int] | None = None
        best_score = float("-inf")
        for arm in cluster:
            arm_state = history.get(arm)
            if arm_state is None or len(arm_state["y"]) == 0:
                continue
            score = float(arm_state["best"])
            if score > best_score:
                best_score = score
                best_arm = arm
        return best_arm

    def _select_unevaluated_arm(
        self,
        cluster: List[Tuple[int, int, int]],
        history: Dict[Tuple[int, int, int], Dict[str, Any]],
    ) -> Tuple[int, int, int] | None:
        """Pick one random arm in the cluster that has no evaluations yet."""
        unevaluated = [arm for arm in cluster if len(history[arm]["y"]) == 0]
        if not unevaluated:
            return None
        return py_random.choice(unevaluated)

    def _physical_method_stats_in_cluster(
        self,
        cluster: List[Tuple[int, int, int]],
        history: Dict[Tuple[int, int, int], Dict[str, Any]],
    ) -> Dict[int, Dict[str, Any]]:
        stats: Dict[int, Dict[str, Any]] = {}
        for arm in cluster:
            physical_method_index = int(arm[1])
            arm_rewards = history[arm]["y"]
            if physical_method_index not in stats:
                stats[physical_method_index] = {
                    "arms": [],
                    "num_observations": 0,
                    "sum_reward": 0.0,
                    "best_reward": float("-inf"),
                }
            stats[physical_method_index]["arms"].append(arm)
            stats[physical_method_index]["num_observations"] += len(arm_rewards)
            if arm_rewards:
                stats[physical_method_index]["sum_reward"] += float(np.sum(arm_rewards))
                stats[physical_method_index]["best_reward"] = max(
                    stats[physical_method_index]["best_reward"],
                    float(np.max(arm_rewards)),
                )

        for physical_method_index, physical_stats in stats.items():
            n_obs = physical_stats["num_observations"]
            if n_obs > 0:
                physical_stats["mean_reward"] = physical_stats["sum_reward"] / n_obs
            else:
                physical_stats["mean_reward"] = 0.0
        return stats

    def _sample_physical_method_in_cluster(
        self,
        cluster: List[Tuple[int, int, int]],
        history: Dict[Tuple[int, int, int], Dict[str, Any]],
        evaluation: int,
    ) -> int:
        physical_stats = self._physical_method_stats_in_cluster(cluster, history)
        physical_method_indices = list(physical_stats.keys())
        physical_scores: List[float] = []

        for physical_method_index in physical_method_indices:
            n_obs = physical_stats[physical_method_index]["num_observations"]
            mean_reward = physical_stats[physical_method_index]["mean_reward"]
            exploration = self.physical_ucb_beta * math.sqrt(math.log(evaluation + 1.0) / (n_obs + 1.0))
            novelty = self.physical_novelty_bonus if n_obs == 0 else 0.0
            physical_scores.append(mean_reward + exploration + novelty)

        physical_scores_np = np.asarray(physical_scores, dtype=float)
        logits = physical_scores_np / self.physical_softmax_temperature
        logits = logits - np.max(logits)
        weights = np.exp(logits)
        weights = weights / np.sum(weights)
        return int(py_random.choices(physical_method_indices, weights=weights.tolist(), k=1)[0])

    def _select_arm_for_physical_method(
        self,
        cluster: List[Tuple[int, int, int]],
        history: Dict[Tuple[int, int, int], Dict[str, Any]],
        physical_method_index: int,
    ) -> Tuple[int, int, int]:
        candidate_arms = [arm for arm in cluster if int(arm[1]) == int(physical_method_index)]
        min_observations = min(len(history[arm]["y"]) for arm in candidate_arms)
        least_observed_arms = [arm for arm in candidate_arms if len(history[arm]["y"]) == min_observations]
        return py_random.choice(least_observed_arms)

    def _propose_candidate_for_arm(
        self,
        arm,
        arm_raw_bounds,
        logical_scale,
        coeff_scale,
        history,
        init_points: int,
    ):
        """Propose one candidate for an arm using warm-start random or GP/EI."""
        arm_state = history[arm]
        if len(arm_state["y"]) < int(init_points):
            logical_error = _sample(
                arm_raw_bounds["logical_method_error"][0],
                arm_raw_bounds["logical_method_error"][1],
                logical_scale,
            )
            coefficient_error = _sample(
                arm_raw_bounds["coefficient_method_error"][0],
                arm_raw_bounds["coefficient_method_error"][1],
                coeff_scale,
            )
            return {
                "logical_method_error": logical_error,
                "coefficient_method_error": coefficient_error,
            }, "random_warm_start"

        gp_context = getattr(self, "_gp_context", None)
        if gp_context is None:
            logical_error = _sample(
                arm_raw_bounds["logical_method_error"][0],
                arm_raw_bounds["logical_method_error"][1],
                logical_scale,
            )
            coefficient_error = _sample(
                arm_raw_bounds["coefficient_method_error"][0],
                arm_raw_bounds["coefficient_method_error"][1],
                coeff_scale,
            )
            return {
                "logical_method_error": logical_error,
                "coefficient_method_error": coefficient_error,
            }, "random_fallback"

        torch = gp_context["torch"]
        SingleTaskGP = gp_context["SingleTaskGP"]
        Normalize = gp_context["Normalize"]
        Standardize = gp_context["Standardize"]
        ExactMarginalLogLikelihood = gp_context["ExactMarginalLogLikelihood"]
        fit_gpytorch_mll = gp_context["fit_gpytorch_mll"]
        LogExpectedImprovement = gp_context["LogExpectedImprovement"]
        optimize_acqf = gp_context["optimize_acqf"]
        device = gp_context["device"]
        dtype = gp_context["dtype"]

        arm_model_bounds = {
            "logical_method_error": (
                self._to_model_space(arm_raw_bounds["logical_method_error"][0], logical_scale),
                self._to_model_space(arm_raw_bounds["logical_method_error"][1], logical_scale),
            ),
            "coefficient_method_error": (
                self._to_model_space(arm_raw_bounds["coefficient_method_error"][0], coeff_scale),
                self._to_model_space(arm_raw_bounds["coefficient_method_error"][1], coeff_scale),
            ),
        }

        try:
            train_X = torch.tensor(np.asarray(arm_state["X"]), dtype=dtype, device=device)
            train_Y = torch.tensor(np.asarray(arm_state["y"]).reshape(-1, 1), dtype=dtype, device=device)
            if train_X.shape[0] < 2:
                raise RuntimeError("Insufficient points for GP fit.")
            if torch.isclose(train_Y.var(), torch.zeros(1, dtype=dtype, device=device)).item():
                raise RuntimeError("Arm observations are constant; skipping EI proposal.")

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
            model = SingleTaskGP(
                train_X,
                train_Y,
                input_transform=Normalize(d=2, bounds=model_bounds_t),
                outcome_transform=Standardize(m=1),
            )
            mll = ExactMarginalLogLikelihood(model.likelihood, model)
            fit_gpytorch_mll(mll)

            acq = LogExpectedImprovement(model=model, best_f=float(train_Y.max().item()))
            cand, _ = optimize_acqf(
                acq_function=acq,
                bounds=model_bounds_t,
                q=1,
                num_restarts=self.num_restarts,
                raw_samples=self.raw_samples,
            )
            return {
                "logical_method_error": self._to_compression_space(float(cand[0, 0].item()), logical_scale),
                "coefficient_method_error": self._to_compression_space(float(cand[0, 1].item()), coeff_scale),
            }, "gp_ei"
        except Exception:
            logical_error = _sample(
                arm_raw_bounds["logical_method_error"][0],
                arm_raw_bounds["logical_method_error"][1],
                logical_scale,
            )
            coefficient_error = _sample(
                arm_raw_bounds["coefficient_method_error"][0],
                arm_raw_bounds["coefficient_method_error"][1],
                coeff_scale,
            )
            return {
                "logical_method_error": logical_error,
                "coefficient_method_error": coefficient_error,
            }, "random_fallback"
    
    def cluster_pipelines(self, arms: List[Tuple[int, int, int]], based_on: str = "logical_method_index") -> List[Tuple[int, int, int]]:
        """Cluster pipelines (arms) based on a specified component index."""
        if based_on not in {"logical_method_index", "coefficient_method_index", "indices_method_index"}:
            raise ValueError(f"Invalid clustering key '{based_on}'. Must be one of: logical_method_index, coefficient_method_index, indices_method_index.")
        
        index_position = {"logical_method_index": 0, "coefficient_method_index": 1, "indices_method_index": 2}[based_on]
        clusters = []
        inverted_index = {}
        for arm in arms:
            key = arm[index_position]
            inverted_index[arm] = key
            if key >= len(clusters):
                clusters.extend([[] for _ in range(key - len(clusters) + 1)])
            clusters[key].append(arm)
        
        return clusters, inverted_index

    @staticmethod
    def _select_screening_arms(
        cluster: List[Tuple[int, int, int]],
    ) -> List[Tuple[int, int, int]]:
        """Choose the first two configured coefficient/index combinations.

        Screening intentionally starts with the lowest coefficient-method
        indices and pairs them with the lowest available indices method. For
        tersets_reduced this selects SerfQT/SerfXOR with
        DeltaFORPFOREncoding: (5, 9) and (6, 9).
        """
        if not cluster:
            return []

        coefficient_indices = sorted({int(arm[1]) for arm in cluster})
        selected_coefficients = coefficient_indices[:2]
        selected = [
            min(
                (arm for arm in cluster if int(arm[1]) == coefficient_index),
                key=lambda arm: int(arm[2]),
            )
            for coefficient_index in selected_coefficients
        ]

        # A one-value coefficient range can still screen two distinct index
        # methods when the cluster contains them.
        for arm in sorted(cluster, key=lambda item: (int(item[1]), int(item[2]))):
            if len(selected) >= min(2, len(cluster)):
                break
            if arm not in selected:
                selected.append(arm)
        return selected
    
    def compute_cluster_halving_budget(
        self,
        n_clusters: int,
        total_budget: int,
        initial_budget_per_cluster: int = 4,
        eta: int = 2,
        target_n_clusters: int = 2,
        round_budget_bonus_step: int = 10,
    ):
        budgets = []
        remaining = total_budget
        current_clusters = n_clusters
        budget_per_cluster = initial_budget_per_cluster
        round_budget_bonus = 0

        while current_clusters > target_n_clusters and remaining > 0:
            round_budget = min(
                current_clusters * budget_per_cluster + round_budget_bonus,
                remaining,
            )

            budgets.append(round_budget)
            remaining -= round_budget

            current_clusters = max(
                math.ceil(current_clusters / eta),
                target_n_clusters,
            )
            budget_per_cluster *= eta
            round_budget_bonus += round_budget_bonus_step

        if remaining > 0:
            budgets.append(remaining)

        return budgets

    def maximize(
        self,
        objective: Callable[[Dict[str, float]], float],
        search_space: Dict[str, Any],
        space_definition: Dict[str, Any],
        log_dir: str | None,
        run_metadata: Dict[str, Any] | None,
        **_,
    ) -> Dict[str, float]:
        """Run the Succesive Halving optimization process."""
        
        dependencies = _require_botorch()
        torch = dependencies["torch"]

        LogExpectedImprovement = dependencies["LogExpectedImprovement"]
        fit_gpytorch_mll = dependencies["fit_gpytorch_mll"]
        SingleTaskGP = dependencies["SingleTaskGP"]
        Normalize = dependencies["Normalize"]
        Standardize = dependencies["Standardize"]
        optimize_acqf = dependencies["optimize_acqf"]
        ExactMarginalLogLikelihood = dependencies["ExactMarginalLogLikelihood"]

        compression_backend = getattr(objective, "backend", None)
        methods: List[str] = getattr(compression_backend, "_methods", None)
        
        logical_index_lo, logical_index_hi = search_space["logical_method_index"]
        coefficient_index_lo, coefficient_index_hi = search_space["coefficient_method_index"]
        indices_index_lo, indices_index_hi = search_space["indices_method_index"]

        logical_index_range = range(int(round(logical_index_lo)), int(round(logical_index_hi)) + 1)
        coefficient_index_range = range(int(round(coefficient_index_lo)), int(round(coefficient_index_hi)) + 1)
        indices_index_range = range(int(round(indices_index_lo)), int(round(indices_index_hi)) + 1)

        pipeline_arms = [(i, j, k) for i in logical_index_range for j in coefficient_index_range for k in indices_index_range]

        space_definition = space_definition or {}
        logical_scale = str((space_definition.get("logical_method_error") or {}).get("scale", "linear"))
        coeff_scale = str((space_definition.get("coefficient_method_error") or {}).get("scale", "linear"))
        if logical_scale not in ("linear", "log") or coeff_scale not in ("linear", "log"):
            raise ValueError("Only linear/log scales are supported for *_method_error in SuccessiveHalvingOptimizer.")

        total_budget = max(1, int(self.total_budget))

        device = _resolve_torch_device(torch, self.device)
        self._gp_context = {
            "torch": torch,
            "LogExpectedImprovement": LogExpectedImprovement,
            "fit_gpytorch_mll": fit_gpytorch_mll,
            "SingleTaskGP": SingleTaskGP,
            "Normalize": Normalize,
            "Standardize": Standardize,
            "optimize_acqf": optimize_acqf,
            "ExactMarginalLogLikelihood": ExactMarginalLogLikelihood,
            "device": device,
            "dtype": torch.float64,
        }
        
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
        self.last_run_dir = log_run.get("run_dir") if log_run else None
        self._log_arm_inventory(
            log_run,
            pipeline_arms,
            methods,
            search_space,
            logical_scale,
            coeff_scale,
        )

        history: Dict[Tuple[int, int, int], Dict[str, Any]] = {
            arm: {"X": [], "y": [], "best": float("-inf")} for arm in pipeline_arms
        }

        state: Dict[str, Any] = {
            "best_x": None,
            "best_y": float("-inf"),
            "evaluations": 0,
        }

        init_points = max(1, int(self.init_points))
        target_n_clusters = max(1, int(_.get("target_n_clusters", 2)))

        clusters, inverted_index = self.cluster_pipelines(pipeline_arms, based_on="logical_method_index")
        best_score_per_cluster = {i: float("-inf") for i in range(len(clusters))}
        samples_per_iteration = self.compute_cluster_halving_budget(
            len(clusters),
            total_budget,
            target_n_clusters=target_n_clusters,
        )

        def _cluster_score(cluster: List[Tuple[int, int, int]]) -> float:
            cluster_values: List[float] = []
            for cluster_arm in cluster:
                cluster_values.extend(history[cluster_arm]["y"])
            if not cluster_values:
                return float("-inf")
            best_value = max(cluster_values)
            mean_value = float(np.mean(cluster_values))
            return 0.7 * best_value + 0.3 * mean_value

        def _quantile(lower_bound: float, upper_bound: float, scale: str, q: float) -> float:
            if scale == "log":
                return self._to_compression_space(
                    np.quantile(
                        [
                            self._to_model_space(lower_bound, scale),
                            self._to_model_space(upper_bound, scale),
                        ],
                        q,
                    ),
                    scale,
                )
            return np.quantile([lower_bound, upper_bound], q)
        
        round_idx = 0
        while state["evaluations"] < total_budget and len(samples_per_iteration) > 0 and len(clusters) > 0:
            samples_in_this_iteration = samples_per_iteration.pop(0)
            remaining_global = total_budget - state["evaluations"]
            round_remaining = min(samples_in_this_iteration, remaining_global)
            if round_remaining <= 0:
                break

            samples_per_cluster = max(1, math.ceil(round_remaining / len(clusters)))
            is_first_round = round_idx == 0
            is_final_phase = len(samples_per_iteration) == 0 or len(clusters) <= target_n_clusters
            if is_first_round:
                phase = "cluster_screening"
            elif is_final_phase:
                phase = "final_exploitation"
            else:
                phase = "cluster_refinement"

            shared_screening_candidate_parameters = None
            if is_first_round:
                # Two shared screening anchors: low bound and midpoint.
                screening_arm = clusters[0][0]
                screening_arm_raw_bounds = self._arm_error_bounds(screening_arm, methods, search_space)
                logical_lo, logical_hi = screening_arm_raw_bounds["logical_method_error"]
                coeff_lo, coeff_hi = screening_arm_raw_bounds["coefficient_method_error"]
                shared_screening_candidate_parameters = [
                    {
                        "logical_method_error": _quantile(logical_lo, logical_hi, logical_scale, 0.1),
                        "coefficient_method_error": _quantile(coeff_lo, coeff_hi, coeff_scale, 0.1),
                    },
                    {
                        "logical_method_error": _quantile(logical_lo, logical_hi, logical_scale, 0.5),
                        "coefficient_method_error": _quantile(coeff_lo, coeff_hi, coeff_scale, 0.5),
                    },
                ]

            for cluster in clusters:
                if state["evaluations"] >= total_budget or round_remaining <= 0:
                    break

                cluster_budget = min(samples_per_cluster, round_remaining)
                if cluster_budget <= 0:
                    continue

                print(f"[successive_halving] round={round_idx} cluster_id={inverted_index[cluster[0]]} cluster_size={len(cluster)} cluster_budget={cluster_budget} evaluations_so_far={state['evaluations']} best_score_in_cluster={best_score_per_cluster[inverted_index[cluster[0]]]:.5f}")

                if is_first_round:
                    logical_method_index = int(cluster[0][0])
                    sampled_arms = self._select_screening_arms(cluster)
                    cluster_evals = 0

                    for arm in sampled_arms:
                        for candidate_parameter in shared_screening_candidate_parameters:
                            if (
                                state["evaluations"] >= total_budget
                                or round_remaining <= 0
                                or cluster_evals >= cluster_budget
                            ):
                                break
                            print(f"[successive_halving] round={round_idx} cluster_id={inverted_index[cluster[0]]} arm={arm} and candidate_parameter={candidate_parameter}")
                            score = self._evaluate_arm_candidate(
                                objective=objective,
                                arm=arm,
                                candidate_parameter=candidate_parameter,
                                methods=methods,
                                logical_scale=logical_scale,
                                coeff_scale=coeff_scale,
                                history=history,
                                state=state,
                                log_run=log_run,
                                phase=phase,
                                proposal_source="screening_fixed_templates",
                                selection_context={
                                    "cluster_id": inverted_index[arm],
                                    "cluster_role": "screening",
                                    "logical_method_index": logical_method_index,
                                    "physical_method_index": int(arm[1]),
                                    "indices_method_index": int(arm[2]),
                                },
                            )
                            cluster_id = inverted_index[arm]
                            best_score_per_cluster[cluster_id] = max(best_score_per_cluster[cluster_id], score)
                            cluster_evals += 1
                            round_remaining -= 1

                elif round_idx == 1:
                    # Round 2: budget-aware physical compressor screening.
                    # Phase 1 — one evaluation per physical compressor (random indices arm).
                    # Phase 2 — UCB over physical compressors for any remaining cluster budget.
                    phys_groups_r2: Dict[int, List[Tuple[int, int, int]]] = {}
                    for arm in cluster:
                        phys_groups_r2.setdefault(int(arm[1]), []).append(arm)

                    cluster_evals = 0

                    def _eval_arm_r2(arm_r2, role_r2):
                        nonlocal cluster_evals
                        arm_raw_bounds_r2 = self._arm_error_bounds(arm_r2, methods, search_space)
                        candidate_r2, source_r2 = self._propose_candidate_for_arm(
                            arm=arm_r2,
                            arm_raw_bounds=arm_raw_bounds_r2,
                            logical_scale=logical_scale,
                            coeff_scale=coeff_scale,
                            history=history,
                            init_points=init_points,
                        )
                        print(f"[successive_halving] round=1 {role_r2} cluster_id={inverted_index[arm_r2]} arm={arm_r2} candidate={candidate_r2}")
                        sc = self._evaluate_arm_candidate(
                            objective=objective,
                            arm=arm_r2,
                            candidate_parameter=candidate_r2,
                            methods=methods,
                            logical_scale=logical_scale,
                            coeff_scale=coeff_scale,
                            history=history,
                            state=state,
                            log_run=log_run,
                            phase=phase,
                            proposal_source=source_r2,
                            selection_context={
                                "cluster_id": inverted_index[arm_r2],
                                "cluster_role": role_r2,
                                "physical_method_index": int(arm_r2[1]),
                            },
                        )
                        best_score_per_cluster[inverted_index[arm_r2]] = max(
                            best_score_per_cluster[inverted_index[arm_r2]], sc
                        )
                        cluster_evals += 1
                        return sc

                    # Phase 1: coverage — one random-indices arm per physical compressor.
                    for phys_idx, phys_arms in phys_groups_r2.items():
                        if state["evaluations"] >= total_budget or round_remaining <= 0 or cluster_evals >= cluster_budget:
                            break
                        arm_r2 = py_random.choice(phys_arms)
                        _eval_arm_r2(arm_r2, "physical_coverage")
                        round_remaining -= 1

                    # Phase 2: UCB over physical compressors for remaining cluster budget.
                    while cluster_evals < cluster_budget and round_remaining > 0 and state["evaluations"] < total_budget:
                        t_r2 = state["evaluations"] + 1
                        best_phys_ucb = float("-inf")
                        best_phys_idx = None
                        for phys_idx, phys_arms in phys_groups_r2.items():
                            phys_rewards = [r for a in phys_arms for r in history[a]["y"]]
                            n_obs = len(phys_rewards)
                            mean_r = float(np.mean(phys_rewards)) if n_obs > 0 else 0.0
                            explore = self.physical_ucb_beta * math.sqrt(math.log(t_r2 + 1.0) / (n_obs + 1.0))
                            novelty = self.physical_novelty_bonus if n_obs == 0 else 0.0
                            ucb = mean_r + explore + novelty
                            if ucb > best_phys_ucb:
                                best_phys_ucb = ucb
                                best_phys_idx = phys_idx
                        # Within the selected physical compressor, pick least-observed arm.
                        arm_r2 = min(phys_groups_r2[best_phys_idx], key=lambda a: len(history[a]["y"]))
                        _eval_arm_r2(arm_r2, "physical_ucb")
                        round_remaining -= 1

                if state["evaluations"] >= total_budget or round_remaining <= 0:
                    break

            # Round 3+ (refinement and final): BOMAB-style global UCB + GP/EI across all surviving arms.
            # UCB exploit uses each arm's best observed reward (same as BOMAB), not mean.
            if round_idx >= 2:
                all_surviving_arms = [arm for c in clusters for arm in c]
                role = "bomab_final" if is_final_phase else "bomab_refinement"
                print(f"[successive_halving] round={round_idx} {role} total_arms={len(all_surviving_arms)} round_remaining={round_remaining}")
                while round_remaining > 0 and state["evaluations"] < total_budget:
                    t = state["evaluations"] + 1
                    ucb_arm = None
                    ucb_best = float("-inf")
                    ucb_exploit_val = 0.0
                    ucb_explore_val = 0.0
                    ucb_rank_val = None

                    for rank, arm in enumerate(all_surviving_arms):
                        arm_state = history[arm]
                        n_obs = len(arm_state["y"])
                        # Use best observed reward per arm, matching BOMAB's UCB definition.
                        exploit = arm_state["best"] if n_obs > 0 else 0.0
                        explore = self.beta * math.sqrt(math.log(t + 1.0) / (n_obs + 1.0))
                        novelty = self.physical_novelty_bonus if n_obs == 0 else 0.0
                        ucb = exploit + explore + novelty
                        if ucb > ucb_best:
                            ucb_best = ucb
                            ucb_exploit_val = exploit
                            ucb_explore_val = explore + novelty
                            ucb_rank_val = rank
                            ucb_arm = arm

                    arm_raw_bounds = self._arm_error_bounds(ucb_arm, methods, search_space)
                    candidate_parameter, proposal_source = self._propose_candidate_for_arm(
                        arm=ucb_arm,
                        arm_raw_bounds=arm_raw_bounds,
                        logical_scale=logical_scale,
                        coeff_scale=coeff_scale,
                        history=history,
                        init_points=init_points,
                    )
                    print(f"[successive_halving] round={round_idx} {role} cluster_id={inverted_index[ucb_arm]} arm={ucb_arm} ucb={ucb_best:.4f} exploit={ucb_exploit_val:.4f} explore={ucb_explore_val:.4f} source={proposal_source}")
                    score = self._evaluate_arm_candidate(
                        objective=objective,
                        arm=ucb_arm,
                        candidate_parameter=candidate_parameter,
                        methods=methods,
                        logical_scale=logical_scale,
                        coeff_scale=coeff_scale,
                        history=history,
                        state=state,
                        log_run=log_run,
                        phase=phase,
                        proposal_source=proposal_source,
                        selection_context={
                            "cluster_id": inverted_index[ucb_arm],
                            "cluster_role": role,
                            "physical_method_index": int(ucb_arm[1]),
                            "ucb_score": ucb_best,
                            "exploit": ucb_exploit_val,
                            "explore": ucb_explore_val,
                            "rank": ucb_rank_val,
                        },
                    )
                    cluster_id = inverted_index[ucb_arm]
                    best_score_per_cluster[cluster_id] = max(best_score_per_cluster[cluster_id], score)
                    round_remaining -= 1

            if round_idx == 0:
                # After round 1: halve logical clusters by mean score (fewer samples, less bias).
                if len(clusters) > target_n_clusters:
                    keep_n = max(target_n_clusters, math.ceil(len(clusters) / 2))
                    clusters = sorted(clusters, key=_cluster_score, reverse=True)[:keep_n]
                    print(f"[successive_halving] after_round=0 surviving_logical_clusters={[inverted_index[c[0]] for c in clusters]}")

            elif round_idx == 1:
                # After round 2:
                # Step 1 — halve logical clusters using the weighted score (more data now).
                if len(clusters) > target_n_clusters:
                    keep_n = max(target_n_clusters, math.ceil(len(clusters) / 2))
                    clusters = sorted(clusters, key=_cluster_score, reverse=True)[:keep_n]
                    print(f"[successive_halving] after_round=1 surviving_logical_clusters={[inverted_index[c[0]] for c in clusters]}")

                # Step 2 — within each surviving logical cluster, prune physical compressor groups.
                def _physical_group_score(arms: List[Tuple[int, int, int]]) -> float:
                    values: List[float] = []
                    for arm in arms:
                        values.extend(history[arm]["y"])
                    if not values:
                        return float("-inf")
                    return 0.7 * max(values) + 0.3 * float(np.mean(values))

                pruned_clusters: List[List[Tuple[int, int, int]]] = []
                for cluster in clusters:
                    # Group arms by physical method index (arm[1]).
                    physical_groups: Dict[int, List[Tuple[int, int, int]]] = {}
                    for arm in cluster:
                        phys_idx = int(arm[1])
                        physical_groups.setdefault(phys_idx, []).append(arm)

                    if len(physical_groups) <= 1:
                        pruned_clusters.append(cluster)
                        continue

                    keep_n_phys = max(1, math.ceil(len(physical_groups) / 2))
                    top_phys_indices = sorted(
                        physical_groups.keys(),
                        key=lambda pi: _physical_group_score(physical_groups[pi]),
                        reverse=True,
                    )[:keep_n_phys]
                    surviving_arms = [
                        arm
                        for pi in top_phys_indices
                        for arm in physical_groups[pi]
                    ]
                    pruned_clusters.append(surviving_arms)
                    print(f"[successive_halving] after_round=1 logical_cluster={inverted_index[cluster[0]]} kept_physical_indices={top_phys_indices} surviving_arms={len(surviving_arms)}/{len(cluster)}")

                clusters = pruned_clusters

            else:
                # Round 3+: continue halving logical clusters.
                if len(clusters) > target_n_clusters:
                    keep_n = max(target_n_clusters, math.ceil(len(clusters) / 2))
                    clusters = sorted(clusters, key=_cluster_score, reverse=True)[:keep_n]

            round_idx += 1

        if state["best_x"] is None:
            message = "successive_halving failed to evaluate any candidate."
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
                status="failed",
                error=message,
            )
            self._log_event(log_run, {"event": "optimization_failed", "error": message})
            self._flush_log_buffers(log_run, force=True)
            raise RuntimeError(message)

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
