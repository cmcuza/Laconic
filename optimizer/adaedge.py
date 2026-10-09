"""AdaEdge: bandit selection over the three single-parameter baselines.

A variation of

    AdaEdge: A Dynamic Compression Selection Framework for Resource
    Constrained Devices -- https://ieeexplore.ieee.org/document/10597763

where the compressor is chosen adaptively instead of fixed up front. Here the
candidates are the three baselines this repo already implements (MixPiece,
SerfXOR, SZ, via ``compression.backend.AdaEdgeBackend``).

Structure
---------
One *arm* per compressor, i.e. one integer ``method_index``. Every arm shares
the same single continuous dimension, ``adaedge_error``. Each iteration:

1) Score every arm by UCB: ``best_y_arm + beta * sqrt(log(t+1) / (n_arm+1))``.
2) Take the best-scoring arm.
3) Propose one error bound *inside that arm*: a local 1-D GP + LogEI once the
   arm has >= 2 observations, otherwise a uniform sample.

The budget is ``init_points + n_iter`` and is spent exactly, so the number this
run advertises as ``n_evaluations`` is the number of objective calls it makes.

Relationship to the other optimizers
------------------------------------
This is BOMAB's idea at a much smaller scale: 3 arms and 1 continuous dimension
instead of 72 arms and 2. It deliberately does *not* subclass
:class:`optimizer.bomab.BOMABOptimizer` - every one of that class's arm helpers
is typed to the TerseTS ``(logical, coefficient, indices)`` triple and its two
named error parameters, so inheriting would mean overriding all of them. For
the same reason it logs through :class:`optimizer.run_logger.RunLogger` (the
generic float-space trace, as ``bosmp`` does) rather than
``optimizer.arm_trace.ArmTraceLogging`` (the TerseTS-shaped one); the bandit
state that has no column in that schema is written as ``arm_selected`` events.

The 1-D duplicate guard is carried over from ``bosmp`` on purpose - see
``_is_duplicate``.
"""

from __future__ import annotations

import math
import traceback
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

import numpy as np

# Shared with bosmp rather than copied: bomab and bosmp already carry one
# byte-identical _require_botorch each, and a third copy is how the CSV field
# lists in the original repo drifted apart (see optimizer/arm_trace.py).
from optimizer.bosmp import _require_botorch, _resolve_torch_device
from optimizer.run_logger import RunLogger
from optimizer.space import parse_bounds_pair, parse_space_metadata
from profiling import stage_clock
from profiling.timed_call import SEAM_FIELDS, seam_extra, timed_call

ERROR_PARAM = "adaedge_error"
INDEX_PARAM = "method_index"


@dataclass(frozen=True)
class _ErrorDim:
    """The single continuous dimension, in raw and model (GP) space.

    Keeping both spaces in one object is what stops the two from drifting: the
    GP, its ``optimize_acqf`` bounds and the random sampler must all live in
    model space, while the objective only ever sees raw space.
    """

    lo_raw: float
    hi_raw: float
    scale: str

    @property
    def lo_model(self) -> float:
        return self.to_model(self.lo_raw)

    @property
    def hi_model(self) -> float:
        return self.to_model(self.hi_raw)

    def to_model(self, value: float) -> float:
        return float(math.log(value)) if self.scale == "log" else float(value)

    def to_raw(self, value: float) -> float:
        value = math.exp(value) if self.scale == "log" else value
        return float(min(max(value, self.lo_raw), self.hi_raw))

    def sample_model(self, rng: np.random.Generator) -> float:
        return float(rng.uniform(self.lo_model, self.hi_model))

    @classmethod
    def from_config(cls, search_space: Dict[str, Any], space_definition: Dict[str, Any]) -> "_ErrorDim":
        lo_raw, hi_raw = parse_bounds_pair(search_space[ERROR_PARAM], ERROR_PARAM)
        if ERROR_PARAM not in space_definition:
            raise ValueError(f"Parameter '{ERROR_PARAM}' is missing from the space definition.")
        scale = str(parse_space_metadata(space_definition[ERROR_PARAM], ERROR_PARAM)["scale"]).lower()
        if scale not in ("linear", "log"):
            raise ValueError(f"Unsupported scale '{scale}' for '{ERROR_PARAM}'. Expected linear/log.")
        if scale == "log" and lo_raw <= 0:
            raise ValueError(f"Log-scaled '{ERROR_PARAM}' needs positive bounds. Got {(lo_raw, hi_raw)}.")
        return cls(lo_raw=float(lo_raw), hi_raw=float(hi_raw), scale=scale)


class AdaEdgeOptimizer:
    """UCB over compressor choice, per-arm 1-D Bayesian optimization inside it.

    Parameters
    ----------
    init_points:
        Warm-start evaluations, dealt round-robin over the (shuffled) arms.
        Keep it a multiple of the arm count for a balanced warm start.
    n_iter:
        Adaptive evaluations after the warm start. ``total_budget`` is the sum.
    beta:
        Exploration weight on the UCB term. With rewards in [0, 1] the raw
        ``sqrt(log(t+1)/(n+1))`` term is O(1), so an unweighted version would
        swamp the exploit term and degenerate into round-robin.
    num_restarts, raw_samples:
        BoTorch acquisition optimizer settings.
    duplicate_tol:
        Relative distance below which a proposal counts as already-evaluated.
    verbose:
        If > 0, print each new global best and the trace directory.
    alpha:
        Fitness weight; carried for run identity (path segment, metadata).
    random_state:
        Seeds this optimizer's own RNG and torch.
    """

    def __init__(
        self,
        init_points: int = 21,
        n_iter: int = 79,
        beta: float = 0.75,
        num_restarts: int = 5,
        raw_samples: int = 32,
        verbose: int = 1,
        alpha: float = 0.75,
        random_state: int = 32,
        device: str = "auto",
        log_dir: str | None = ".logs",
        log_subdir: str = "_adaedge_opt",
        log_process: bool = True,
        duplicate_tol: float = 1e-4,
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
        self.duplicate_tol = float(duplicate_tol)
        self.rng = np.random.default_rng(self.random_state)
        self.last_run_dir: str | None = None

    # -- bandit ------------------------------------------------------------

    def _select_arm(self, history: Dict[int, Dict[str, Any]], t: int) -> Tuple[int, Dict[str, Any]]:
        """Highest-UCB arm, plus the score breakdown for the trace."""
        best_arm, best_context, best_ucb = None, {}, float("-inf")
        for arm, arm_state in history.items():
            n_obs = len(arm_state["y"])
            exploit = arm_state["best"] if n_obs else 0.0
            explore = self.beta * math.sqrt(2*math.log(t + 1.0) / (n_obs + 1.0))
            ucb = exploit + explore
            if ucb > best_ucb:
                best_arm, best_ucb = arm, ucb
                best_context = {
                    "num_observations": n_obs,
                    "exploit": exploit,
                    "explore": explore,
                    "ucb_score": ucb,
                }
        return best_arm, best_context

    # -- proposal ----------------------------------------------------------

    def _is_duplicate(self, candidate_model: float, arm_X: List[float], dim: _ErrorDim) -> bool:
        """Has this arm (numerically) already been evaluated here?

        The objective is deterministic, so re-evaluating a known point buys
        zero information at the cost of a full compress -> decompress ->
        predict cycle. In 1-D, LogEI's maximizer collapses onto the incumbent
        as soon as BO converges, so without this guard an arm spends most of
        its share of the budget re-measuring one point - measured on the
        single-parameter baselines, which is why bosmp grew the same guard.
        """
        if not arm_X:
            return False
        span = max(dim.hi_model - dim.lo_model, 1e-12)
        distances = np.abs(np.asarray(arm_X, dtype=float) - candidate_model) / span
        return bool((distances <= self.duplicate_tol).any())

    def _propose(self, arm_state: Dict[str, Any], dim: _ErrorDim) -> Tuple[float, str, str | None]:
        """Next model-space error bound for one arm: GP + LogEI, else random.

        Returns ``(point_model, proposal_source, fallback_reason)``.
        """
        gp = self._gp_context
        torch, dtype, device = gp["torch"], gp["dtype"], gp["device"]
        random_point = dim.sample_model(self.rng)

        if len(arm_state["y"]) < 2:
            return random_point, "random_insufficient_arm_data", None

        train_X = torch.tensor(arm_state["X"], dtype=dtype, device=device).reshape(-1, 1)
        train_Y = torch.tensor(arm_state["y"], dtype=dtype, device=device).reshape(-1, 1)
        if torch.isclose(train_Y.var(), torch.zeros(1, dtype=dtype, device=device)).item():
            # Standardize(m=1) divides by this variance - the GP is undefined
            # here, not merely inaccurate.
            return random_point, "random_gp_fallback", "Arm observations are constant."

        bounds_t = torch.tensor([[dim.lo_model], [dim.hi_model]], dtype=dtype, device=device)
        try:
            model = gp["SingleTaskGP"](
                train_X,
                train_Y,
                # Explicit search-space bounds, not data-inferred: SingleTaskGP's
                # default lengthscale prior assumes inputs in [0,1]^d, and with
                # 2-3 points a data-inferred Normalize is near-degenerate.
                input_transform=gp["Normalize"](d=1, bounds=bounds_t),
                outcome_transform=gp["Standardize"](m=1),
            )
            mll = gp["ExactMarginalLogLikelihood"](model.likelihood, model)
            gp["fit_gpytorch_mll"](mll)
            acq = gp["LogExpectedImprovement"](model=model, best_f=float(train_Y.max().item()))
            cand, _ = gp["optimize_acqf"](
                acq_function=acq,
                bounds=bounds_t,
                q=1,
                num_restarts=self.num_restarts,
                raw_samples=self.raw_samples,
            )
        except (NameError, AttributeError, ImportError) as exc:
            # A numerically-failed fit is a legitimate reason to fall back; a
            # broken symbol is a bug in this file, and degrading every proposal
            # to random would hide it completely (it did exactly that here once).
            raise RuntimeError(f"adaedge proposal step is broken, not merely failing: {exc!r}") from exc
        except Exception as exc:
            return random_point, "random_gp_fallback", f"{type(exc).__name__}: {exc}"

        point_model = float(cand[0, 0].item())
        if self._is_duplicate(point_model, arm_state["X"], dim):
            return random_point, "random_duplicate_proposal", f"within {self.duplicate_tol} of an evaluated point"
        return point_model, "log_expected_improvement", None

    # -- main loop ---------------------------------------------------------

    def maximize(
        self,
        objective,
        search_space,
        space_definition,
        log_dir,
        run_metadata,
        **_,
    ) -> Dict[str, float]:
        """Run the AdaEdge loop and return the best ``{method_index, error}``."""
        deps = _require_botorch()
        torch = deps["torch"]
        torch.manual_seed(self.random_state)
        device = _resolve_torch_device(torch, self.device, who="adaedge")
        self._gp_context = dict(deps, device=device, dtype=torch.double)

        methods: List[str] = objective.backend._methods
        dim = _ErrorDim.from_config(search_space, space_definition)
        lo_index, hi_index = parse_bounds_pair(search_space[INDEX_PARAM], INDEX_PARAM)
        arms = list(range(int(round(lo_index)), int(round(hi_index)) + 1))
        # self.total_budget is what this run reports as n_evaluations and writes
        # into its budget_<N> path, so the loop must spend exactly it.
        total_budget = self.total_budget

        selected_log_dir = (self.log_dir if log_dir is None else log_dir) if self.log_process else None
        logger = RunLogger(
            log_dir=selected_log_dir,
            run_metadata=run_metadata,
            optimizer_name="adaedge",
            optimizer_meta={
                "init_points": self.init_points,
                "n_iter": self.n_iter,
                "beta": self.beta,
                "num_restarts": self.num_restarts,
                "raw_samples": self.raw_samples,
                "duplicate_tol": self.duplicate_tol,
                "alpha": self.alpha,
                "random_state": self.random_state,
                "device": str(device),
                "arms": {arm: methods[arm] for arm in arms},
            },
            search_space=search_space,
            space_definition=space_definition,
            param_names=[INDEX_PARAM, ERROR_PARAM],
            total_budget=total_budget,
            log_subdir=self.log_subdir,
            verbose=self.verbose,
            # See docs/EXECUTION_TIME_STUDY_PLAN.md §5.4: objective_elapsed_sec
            # already measures this optimizer's own perf_counter pair exactly
            # (unlike genetic's pre-§5.4 batch average), so only the CPU column
            # is new here.
            # The per-seam columns are declared only when the stage clock is
            # on (i.e. under scripts/benchmark_execution_time.py), so an
            # ordinary run's trace schema is byte-identical to before.
            extra_eval_fields=["objective_cpu_sec"]
            + (SEAM_FIELDS if stage_clock.enabled() else []),
        )
        self.last_run_dir = logger.run_dir
        # See optimizer/genetic.py's identical attribute: the RunLogger itself,
        # so `last_logger.total_logging_sec` gives this run's T_log. For adaedge
        # that also covers the per-iteration "arm_selected" log_event, which is
        # one more open/close per evaluation than genetic pays (§2.2(C)).
        self.last_logger = logger
        # {stage_name: [total_seconds, count]}, summed across every
        # evaluation's TimingRecord.stages this call - the in-process
        # counterpart of optimizer/genetic.py::last_stage_totals (§7.2:
        # "the drained stage_clock totals... attributed to search... by
        # summing the returned dicts" - adaedge has no worker pool to sum
        # across, but timed_call() still drains per call and the running
        # total is otherwise discarded the same way the CPU sum used to be).
        self.last_stage_totals: Dict[str, list] = {}
        # Sum of every evaluation's own timed_call() wall_sec this call - the
        # "measured objective time" verification 5
        # (docs/EXECUTION_TIME_STUDY_PLAN.md §10) checks stage_clock's
        # decomposition against. adaedge has no pool, so `search`'s own wall
        # time also contains T_algo (GP fit + LogEI maximization) BETWEEN
        # evaluations - this is the sum of only the evaluations themselves.
        self.last_total_objective_wall_sec = 0.0

        history: Dict[int, Dict[str, Any]] = {
            arm: {"X": [], "y": [], "best": float("-inf")} for arm in arms
        }
        state: Dict[str, Any] = {"best_x": None, "best_y": float("-inf"), "evaluations": 0}

        def _evaluate(arm: int, point_model: float, phase: str, proposal_source: str) -> float:
            point_raw = {INDEX_PARAM: float(arm), ERROR_PARAM: dim.to_raw(point_model)}
            global_best_before = state["best_y"]
            try:
                score, timing = timed_call(objective, point_raw)
                score = float(score)
            except Exception as exc:
                logger.log_event(
                    {
                        "event": "evaluation_failed",
                        "evaluation": state["evaluations"] + 1,
                        "phase": phase,
                        "proposal_source": proposal_source,
                        "arm": arm,
                        "arm_method": methods[arm],
                        "candidate": point_raw,
                        "error_type": type(exc).__name__,
                        "error": str(exc),
                        "traceback": traceback.format_exc(),
                    }
                )
                raise
            elapsed_sec = timing["wall_sec"]
            self.last_total_objective_wall_sec += elapsed_sec
            for stage_name, (stage_total, stage_count) in timing["stages"].items():
                entry = self.last_stage_totals.setdefault(stage_name, [0.0, 0])
                entry[0] += stage_total
                entry[1] += stage_count

            arm_state = history[arm]
            arm_state["X"].append(point_model)
            arm_state["y"].append(score)
            arm_state["best"] = max(arm_state["best"], score)
            state["evaluations"] += 1
            is_new_global_best = score > state["best_y"]
            if is_new_global_best:
                state["best_y"] = score
                state["best_x"] = point_raw
                if self.verbose > 0:
                    print(f"[adaedge] new best={score:.5f} method={methods[arm]}")

            logger.log_evaluation(
                evaluation=state["evaluations"],
                params_raw=point_raw,
                params_model={INDEX_PARAM: float(arm), ERROR_PARAM: point_model},
                reward=score,
                elapsed_sec=elapsed_sec,
                global_best_before=global_best_before,
                global_best_after=state["best_y"],
                is_new_global_best=is_new_global_best,
                phase=phase,
                proposal_source=proposal_source,
                extra={"objective_cpu_sec": timing["cpu_sec"], **seam_extra(timing)},
            )
            return score

        # Warm start: deal init_points evaluations round-robin over the arms.
        # Shuffled so that when init_points is not a multiple of the arm count,
        # which arm gets the extra evaluation follows the seed, not list order.
        warm_arms = list(arms)
        self.rng.shuffle(warm_arms)
        for k in range(min(self.init_points, total_budget)):
            arm = warm_arms[k % len(warm_arms)]
            _evaluate(arm, dim.sample_model(self.rng), "warm_start", "random_warm_start")

        # Adaptive: UCB picks the arm, a local 1-D GP picks the error bound.
        while state["evaluations"] < total_budget:
            arm, context = self._select_arm(history, state["evaluations"] + 1)
            point_model, proposal_source, fallback_reason = self._propose(history[arm], dim)
            logger.log_event(
                {
                    "event": "arm_selected",
                    "evaluation": state["evaluations"] + 1,
                    "arm": arm,
                    "arm_method": methods[arm],
                    "proposal_source": proposal_source,
                    "fallback_reason": fallback_reason,
                    **context,
                }
            )
            _evaluate(arm, point_model, "adaptive", proposal_source)

        # A failed evaluation raises out of _evaluate, so best_x is set here.
        arm_summary = {
            methods[arm]: {
                "num_observations": len(arm_state["y"]),
                "best_reward": arm_state["best"] if arm_state["y"] else None,
                "mean_reward": float(np.mean(arm_state["y"])) if arm_state["y"] else None,
            }
            for arm, arm_state in history.items()
        }
        logger.write_summary(
            evaluations=state["evaluations"],
            total_budget=total_budget,
            best_reward=state["best_y"],
            best_params=state["best_x"],
            status="completed",
        )
        logger.log_event(
            {
                "event": "optimization_completed",
                "evaluations": state["evaluations"],
                "best_reward": state["best_y"],
                "best_params": state["best_x"],
                "arm_summary": arm_summary,
            }
        )
        if self.verbose > 0:
            allocation = ", ".join(
                f"{name}={info['num_observations']}" for name, info in arm_summary.items()
            )
            print(f"[adaedge] arm allocation: {allocation}")
        return state["best_x"]
