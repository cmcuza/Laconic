"""Adaptive Successive Halving over per-arm Bayesian optimization.

Budget-adaptive replacement for :class:`SuccessiveHalvingOptimizer`
(``optimizer/successive_halving.py``), which this repo keeps only to reproduce
older results. That implementation has two problems this one removes:

1. **Hard-coded pipelines and anchors.** Its screening round only ever
   evaluates arms with ``coefficient_method_index in (6, 9)`` and
   ``indices_method_index == 11`` - literal positions in the tersets_mab
   method list - at two fixed error-bound quantiles computed from a single
   arm's bounds. Any change to the method list or bounds either crashes it
   (``StopIteration``) or silently biases the search toward those pipelines.
   Here, nothing about the compressor is baked in: arms are enumerated from
   ``search_space``, screening candidates are sampled scale-aware from each
   arm's own bounds, and arms within a cluster are chosen by UCB.

2. **A rigid, budget-blind round structure.** Its rounds 0/1/2+ are
   special-cased by index with fixed per-cluster costs, so a small budget is
   consumed entirely by screening and a large one dumps the surplus into the
   last round. Here, the number of halving rounds is derived from the cluster
   count (``ceil(log_eta(n_clusters / target_n_clusters))`` halvings plus one
   final exploitation round) and the total budget is split evenly across
   rounds - the classic successive-halving allocation, which automatically
   gives surviving clusters more evaluations per round as the field narrows.
   Every budget from 1 upward is spent exactly.

Within a round, each surviving cluster gets an equal share of the round
budget; inside a cluster, arms are picked by UCB (best observed reward +
exploration + a novelty bonus for unevaluated arms) and candidates are
proposed per-arm: random scale-aware sampling until ``init_points``
observations, then GP + LogEI (with random fallback), exactly like BOMAB.
The final round pools all surviving arms and runs the same UCB + GP/EI loop
globally. After every non-final round, clusters are halved by
``0.7 * best + 0.3 * mean`` observed reward.
"""

from __future__ import annotations

import json
import math
from typing import Any, Callable, Dict, List, Tuple

import numpy as np

from optimizer.bomab import BOMABOptimizer, _require_botorch, _resolve_torch_device


class AdaptiveHalvingOptimizer(BOMABOptimizer):
    """Successive halving over logical-method clusters, sized by the budget.

    Parameters
    ----------
    total_budget:
        Total number of objective evaluations. The whole round schedule is
        derived from this number, so any value >= 1 is spent exactly.
    init_points:
        Per-arm observation count below which candidates are random-sampled
        instead of proposed by the arm's GP.
    beta:
        Exploration strength of the UCB arm-selection policy.
    eta:
        Halving rate: after each non-final round the surviving cluster count
        becomes ``ceil(n / eta)`` (floored at ``target_n_clusters``).
    target_n_clusters:
        Cluster count at which halving stops and the final exploitation
        round begins.
    novelty_bonus:
        UCB bonus for arms with no observations yet.
    """

    def __init__(
        self,
        total_budget: int = 100,
        init_points: int = 3,
        beta: float = 0.75,
        eta: int = 2,
        target_n_clusters: int = 2,
        novelty_bonus: float = 0.25,
        num_restarts: int = 5,
        raw_samples: int = 32,
        verbose: int = 1,
        alpha: float = 0.75,
        random_state: int = 32,
        device: str = "auto",
        log_dir: str | None = ".logs",
        log_subdir: str = "_adaptive_halving_opt",
        log_process: bool = True,
        log_ucb_scores: bool = True,
        log_flush_every: int = 64,
    ):
        super().__init__(
            init_points=init_points,
            n_iter=0,
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
        self.total_budget = max(1, int(total_budget))
        self.eta = max(2, int(eta))
        self.target_n_clusters = max(1, int(target_n_clusters))
        self.novelty_bonus = float(novelty_bonus)

    # ------------------------------------------------------------------
    # Schedule
    # ------------------------------------------------------------------

    def compute_round_budgets(self, n_clusters: int, total_budget: int) -> List[int]:
        """Even budget split over ``halvings + 1`` rounds, remainder to the end.

        The number of halving rounds is exactly what it takes to shrink
        ``n_clusters`` to ``target_n_clusters`` at rate ``eta``; one final
        exploitation round is always appended. Equal per-round budgets are the
        classic successive-halving allocation: as clusters are halved, the
        per-cluster share doubles each round.
        """
        halvings = 0
        current = int(n_clusters)
        while current > self.target_n_clusters:
            current = max(self.target_n_clusters, math.ceil(current / self.eta))
            halvings += 1
        n_rounds = halvings + 1

        base = total_budget // n_rounds
        budgets = [base] * n_rounds
        # Hand out the remainder one per round from the final round backwards,
        # so leftover evaluations favor exploitation.
        for offset in range(total_budget - base * n_rounds):
            budgets[n_rounds - 1 - offset] += 1
        return budgets

    # ------------------------------------------------------------------
    # Selection / proposal
    # ------------------------------------------------------------------

    def _select_arm_ucb(
        self,
        arms: List[Tuple[int, int, int]],
        history: Dict[Tuple[int, int, int], Dict[str, Any]],
        t: int,
    ) -> Tuple[Tuple[int, int, int], Dict[str, Any]]:
        """UCB over ``arms``: best observed reward + exploration + novelty."""
        best_arm = None
        best_context: Dict[str, Any] = {}
        best_ucb = float("-inf")
        for rank, arm in enumerate(arms):
            n_obs = len(history[arm]["y"])
            exploit = history[arm]["best"] if n_obs > 0 else 0.0
            explore = self.beta * math.sqrt(math.log(t + 1.0) / (n_obs + 1.0))
            novelty = self.novelty_bonus if n_obs == 0 else 0.0
            ucb = exploit + explore + novelty
            if ucb > best_ucb:
                best_ucb = ucb
                best_arm = arm
                best_context = {
                    "ucb_score": ucb,
                    "exploit": exploit,
                    "explore": explore + novelty,
                    "rank": rank,
                }
        return best_arm, best_context

    def _propose_candidate_for_arm(
        self,
        arm: Tuple[int, int, int],
        arm_raw_bounds: Dict[str, Tuple[float, float]],
        logical_scale: str,
        coeff_scale: str,
        history: Dict[Tuple[int, int, int], Dict[str, Any]],
    ) -> Tuple[Dict[str, float], str]:
        """Random scale-aware sample below ``init_points`` obs, else GP + LogEI."""
        rng = np.random

        def _random_candidate() -> Dict[str, float]:
            return {
                "logical_method_error": self._sample(
                    rng,
                    arm_raw_bounds["logical_method_error"][0],
                    arm_raw_bounds["logical_method_error"][1],
                    logical_scale,
                ),
                "coefficient_method_error": self._sample(
                    rng,
                    arm_raw_bounds["coefficient_method_error"][0],
                    arm_raw_bounds["coefficient_method_error"][1],
                    coeff_scale,
                ),
            }

        arm_state = history[arm]
        if len(arm_state["y"]) < self.init_points:
            return _random_candidate(), "random_warm_start"

        gp = self._gp_context
        torch = gp["torch"]
        dtype = gp["dtype"]
        device = gp["device"]
        arm_model_bounds = self._arm_model_bounds(arm_raw_bounds, logical_scale, coeff_scale)
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
            model = gp["SingleTaskGP"](
                train_X,
                train_Y,
                input_transform=gp["Normalize"](d=2, bounds=model_bounds_t),
                outcome_transform=gp["Standardize"](m=1),
            )
            mll = gp["ExactMarginalLogLikelihood"](model.likelihood, model)
            gp["fit_gpytorch_mll"](mll)

            acq = gp["LogExpectedImprovement"](model=model, best_f=float(train_Y.max().item()))
            cand, _ = gp["optimize_acqf"](
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
            return _random_candidate(), "random_gp_fallback"

    # ------------------------------------------------------------------
    # Logging identity
    # ------------------------------------------------------------------

    def _start_log_run(self, **kwargs) -> Dict[str, Any] | None:
        # Reuse BOMAB's logger plumbing but rewrite the optimizer identity.
        original_verbose = self.verbose
        self.verbose = 0
        try:
            log_run = super()._start_log_run(**kwargs)
        finally:
            self.verbose = original_verbose
        if log_run is None:
            return None

        metadata_path = log_run["paths"]["metadata"]
        with open(metadata_path, "r", encoding="utf-8") as f:
            metadata = json.load(f)
        metadata.setdefault("optimizer", {}).update(
            {
                "name": "adaptive_halving",
                "total_budget": self.total_budget,
                "init_points": self.init_points,
                "beta": self.beta,
                "eta": self.eta,
                "target_n_clusters": self.target_n_clusters,
                "novelty_bonus": self.novelty_bonus,
                "alpha": self.alpha,
                "num_restarts": self.num_restarts,
                "raw_samples": self.raw_samples,
                "random_state": self.random_state,
            }
        )
        with open(metadata_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2, sort_keys=True)
            f.write("\n")

        if self.verbose > 0:
            print(f"[adaptive_halving] logging optimization trace to {log_run['run_dir']}")
        return log_run

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------

    def maximize(
        self,
        objective: Callable[[Dict[str, float]], float],
        search_space: Dict[str, Any],
        space_definition: Dict[str, Any],
        log_dir: str,
        run_metadata: Dict[str, Any] | None,
        **_,
    ) -> Dict[str, float]:
        deps = _require_botorch()
        torch = deps["torch"]
        device = _resolve_torch_device(torch, self.device)
        self._gp_context = dict(deps, device=device, dtype=torch.float64)

        methods: List[str] = objective.backend._methods

        logical_index_lo, logical_index_hi = search_space["logical_method_index"]
        coefficient_index_lo, coefficient_index_hi = search_space["coefficient_method_index"]
        indices_index_lo, indices_index_hi = search_space["indices_method_index"]
        pipeline_arms = [
            (i, j, k)
            for i in range(int(round(logical_index_lo)), int(round(logical_index_hi)) + 1)
            for j in range(int(round(coefficient_index_lo)), int(round(coefficient_index_hi)) + 1)
            for k in range(int(round(indices_index_lo)), int(round(indices_index_hi)) + 1)
        ]

        logical_scale = str(space_definition["logical_method_error"]["scale"])
        coeff_scale = str(space_definition["coefficient_method_error"]["scale"])
        if logical_scale not in ("linear", "log") or coeff_scale not in ("linear", "log"):
            raise ValueError("Only linear/log scales are supported for *_method_error in AdaptiveHalvingOptimizer.")

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
        self._log_arm_inventory(log_run, pipeline_arms, methods, search_space, logical_scale, coeff_scale)

        history: Dict[Tuple[int, int, int], Dict[str, Any]] = {
            arm: {"X": [], "y": [], "best": float("-inf")} for arm in pipeline_arms
        }
        state: Dict[str, Any] = {"best_x": None, "best_y": float("-inf"), "evaluations": 0}

        # Cluster by logical method; each cluster is one "configuration" being halved.
        clusters_by_key: Dict[int, List[Tuple[int, int, int]]] = {}
        for arm in pipeline_arms:
            clusters_by_key.setdefault(int(arm[0]), []).append(arm)
        clusters = [clusters_by_key[key] for key in sorted(clusters_by_key)]

        def _cluster_score(cluster: List[Tuple[int, int, int]]) -> float:
            values = [v for cluster_arm in cluster for v in history[cluster_arm]["y"]]
            if not values:
                return float("-inf")
            return 0.7 * max(values) + 0.3 * float(np.mean(values))

        def _evaluate(arm, phase, role, selection_context):
            arm_raw_bounds = self._arm_error_bounds(arm, methods, search_space)
            candidate_parameter, proposal_source = self._propose_candidate_for_arm(
                arm, arm_raw_bounds, logical_scale, coeff_scale, history
            )
            selection_context = dict(selection_context)
            selection_context.update({"cluster_id": int(arm[0]), "cluster_role": role})
            if self.verbose > 1:
                print(f"[adaptive_halving] {phase} arm={arm} source={proposal_source} candidate={candidate_parameter}")
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
                phase=phase,
                proposal_source=proposal_source,
                selection_context=selection_context,
            )

        round_budgets = self.compute_round_budgets(len(clusters), total_budget)
        if self.verbose > 0:
            print(
                f"[adaptive_halving] budget={total_budget} clusters={len(clusters)} "
                f"eta={self.eta} target={self.target_n_clusters} round_budgets={round_budgets}"
            )

        for round_idx, round_budget in enumerate(round_budgets):
            is_final = round_idx == len(round_budgets) - 1
            phase = "final_exploitation" if is_final else f"halving_round_{round_idx}"

            if round_budget > 0 and is_final:
                # Pool all surviving arms; global UCB + per-arm GP/EI.
                surviving_arms = [arm for cluster in clusters for arm in cluster]
                for _ in range(round_budget):
                    arm, context = self._select_arm_ucb(surviving_arms, history, state["evaluations"] + 1)
                    _evaluate(arm, phase, "global_ucb", context)
            elif round_budget > 0:
                # Even split across surviving clusters; visit them in random
                # order so a sub-cluster-count round budget carries no
                # positional bias toward low logical indices.
                order = np.random.permutation(len(clusters))
                per_cluster = round_budget // len(clusters)
                remainder = round_budget % len(clusters)
                for position, cluster_position in enumerate(order):
                    cluster = clusters[int(cluster_position)]
                    cluster_budget = per_cluster + (1 if position < remainder else 0)
                    for _ in range(cluster_budget):
                        arm, context = self._select_arm_ucb(cluster, history, state["evaluations"] + 1)
                        _evaluate(arm, phase, "cluster_ucb", context)

            if not is_final and len(clusters) > self.target_n_clusters:
                keep_n = max(self.target_n_clusters, math.ceil(len(clusters) / self.eta))
                clusters = sorted(clusters, key=_cluster_score, reverse=True)[:keep_n]
                if self.verbose > 0:
                    survivors = sorted(int(cluster[0][0]) for cluster in clusters)
                    print(f"[adaptive_halving] after round {round_idx}: surviving logical clusters={survivors}")
                self._log_event(
                    log_run,
                    {
                        "event": "clusters_halved",
                        "round": round_idx,
                        "surviving_clusters": [int(cluster[0][0]) for cluster in clusters],
                    },
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
        self._write_final_summary(log_run, state=state, total_budget=total_budget, status="completed")
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
