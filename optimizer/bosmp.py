from __future__ import annotations

import math
import traceback
from dataclasses import dataclass
from typing import Any, Dict, List

import numpy as np

from optimizer.run_logger import RunLogger
from optimizer.space import parse_bounds_pair, parse_space_metadata
from profiling.timed_call import timed_call


def _require_botorch():
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
            "bosmp requires optional dependencies: torch, gpytorch, botorch. "
            "Install them to use this optimizer."
        ) from exc


def _resolve_torch_device(torch, requested_device: str, who: str = "bosmp"):
    normalized = str(requested_device).lower()
    if normalized == "auto":
        normalized = "cuda" if torch.cuda.is_available() else "cpu"
    if normalized == "cuda" and not torch.cuda.is_available():
        print(f"Warning: CUDA requested for {who}, but no GPU is available. Falling back to CPU.")
        normalized = "cpu"
    if normalized not in {"cpu", "cuda"}:
        raise ValueError(
            f"Unsupported device '{requested_device}' for {who}. Expected one of: auto, cpu, cuda."
        )
    return torch.device(normalized)


@dataclass
class _Var:
    name: str
    lo_raw: float
    hi_raw: float
    lo_model: float
    hi_model: float
    scale: str
    vtype: str


class BOSimple:
    """Simple BoTorch optimizer (single surrogate, no MAB layer)."""

    def __init__(
        self,
        init_points: int = 20,
        n_iter: int = 60,
        num_restarts: int = 5,
        raw_samples: int = 32,
        verbose: int = 1,
        alpha: float = 0.75,
        random_state: int = 2048,
        device: str = "auto",
        log_dir: str | None = ".logs",
        log_subdir: str = "_bosmp_opt",
        log_process: bool = True,
        duplicate_tol: float = 1e-4,
    ):
        self.init_points = int(init_points)
        self.n_iter = int(n_iter)
        self.total_budget = self.init_points + self.n_iter
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

    def _is_duplicate(
        self, point_model: Dict[str, float], train_X: List[List[float]], vars_def: List[_Var]
    ) -> bool:
        if not train_X:
            return False
        candidate = np.array([point_model[v.name] for v in vars_def], dtype=float)
        spans = np.array([max(v.hi_model - v.lo_model, 1e-12) for v in vars_def], dtype=float)
        relative = np.abs(np.asarray(train_X, dtype=float) - candidate) / spans
        return bool((relative.max(axis=1) <= self.duplicate_tol).any())

    def _space_meta(self, space_definition: Dict[str, Any], key: str) -> Dict[str, Any]:
        if key not in space_definition:
            raise ValueError(f"Parameter '{key}' is missing from the space definition.")
        return parse_space_metadata(space_definition[key], key)

    def _build_vars(self, search_space: Dict[str, Any], space_definition: Dict[str, Any]) -> List[_Var]:
        vars_out: List[_Var] = []
        for key, raw_bound in search_space.items():
            lo_raw, hi_raw = parse_bounds_pair(raw_bound, key)
            meta = self._space_meta(space_definition, key)
            scale = str(meta["scale"]).lower()
            if scale not in ("linear", "log"):
                raise ValueError(f"Unsupported scale '{scale}' for '{key}'. Expected linear/log.")
            if scale == "log" and (lo_raw <= 0 or hi_raw <= 0):
                raise ValueError(f"Log-scaled parameter '{key}' must have positive bounds. Got {(lo_raw, hi_raw)}.")

            lo_model = math.log(lo_raw) if scale == "log" else lo_raw
            hi_model = math.log(hi_raw) if scale == "log" else hi_raw
            vtype = str(meta["type"]).lower()
            if vtype not in ("int", "float"):
                raise ValueError(f"Unsupported type '{vtype}' for '{key}'. Expected int/float.")

            vars_out.append(
                _Var(
                    name=key,
                    lo_raw=float(lo_raw),
                    hi_raw=float(hi_raw),
                    lo_model=float(lo_model),
                    hi_model=float(hi_model),
                    scale=scale,
                    vtype=vtype,
                )
            )
        if not vars_out:
            raise ValueError("BOSimple requires a non-empty search space.")
        return vars_out

    @staticmethod
    def _sample_model_point(rng: np.random.Generator, vars_def: List[_Var]) -> Dict[str, float]:
        return {v.name: float(rng.uniform(v.lo_model, v.hi_model)) for v in vars_def}

    @staticmethod
    def _model_to_raw(point_model: Dict[str, float], vars_def: List[_Var]) -> Dict[str, float]:
        out: Dict[str, float] = {}
        for v in vars_def:
            value = float(point_model[v.name])
            value = math.exp(value) if v.scale == "log" else value
            value = min(max(value, v.lo_raw), v.hi_raw)
            if v.vtype == "int":
                value = float(int(round(value)))
                value = min(max(value, v.lo_raw), v.hi_raw)
            out[v.name] = float(value)
        return out

    def maximize(
        self,
        objective,
        search_space,
        space_definition,
        log_dir,
        run_metadata,
        **_,
    ) -> Dict[str, float]:
        deps = _require_botorch()
        torch = deps["torch"]
        LogExpectedImprovement = deps["LogExpectedImprovement"]
        fit_gpytorch_mll = deps["fit_gpytorch_mll"]
        SingleTaskGP = deps["SingleTaskGP"]
        Normalize = deps["Normalize"]
        Standardize = deps["Standardize"]
        optimize_acqf = deps["optimize_acqf"]
        ExactMarginalLogLikelihood = deps["ExactMarginalLogLikelihood"]

        if not isinstance(space_definition, dict):
            raise ValueError("BOSimple requires a space_definition dict annotating every parameter.")
        vars_def = self._build_vars(search_space, space_definition)
        torch.manual_seed(self.random_state)
        device = _resolve_torch_device(torch, self.device)
        dtype = torch.double

        total_budget = self.total_budget
        n_warm = self.init_points

        selected_log_dir = (self.log_dir if log_dir is None else log_dir) if self.log_process else None
        logger = RunLogger(
            log_dir=selected_log_dir,
            run_metadata=run_metadata,
            optimizer_name="bosmp",
            optimizer_meta={
                "init_points": self.init_points,
                "n_iter": self.n_iter,
                "num_restarts": self.num_restarts,
                "raw_samples": self.raw_samples,
                "alpha": self.alpha,
                "random_state": self.random_state,
                "device": str(device),
            },
            search_space=search_space,
            space_definition=space_definition,
            param_names=[v.name for v in vars_def],
            total_budget=total_budget,
            log_subdir=self.log_subdir,
            verbose=self.verbose,
            extra_eval_fields=["objective_cpu_sec"],
        )
        self.last_run_dir = logger.run_dir
        self.last_logger = logger

        train_X: List[List[float]] = []
        train_y: List[float] = []
        best_x: Dict[str, float] | None = None
        best_y = float("-inf")
        evaluations = 0

        def _evaluate(point_model: Dict[str, float], phase: str, proposal_source: str) -> float:
            nonlocal best_x, best_y, evaluations
            point_raw = self._model_to_raw(point_model, vars_def)
            global_best_before = best_y
            try:
                score, timing = timed_call(objective, point_raw)
                score = float(score)
            except Exception as exc:
                logger.log_event(
                    {
                        "event": "evaluation_failed",
                        "evaluation": evaluations + 1,
                        "phase": phase,
                        "proposal_source": proposal_source,
                        "candidate": point_raw,
                        "error_type": type(exc).__name__,
                        "error": str(exc),
                        "traceback": traceback.format_exc(),
                    }
                )
                raise
            elapsed_sec = timing["wall_sec"]
            train_X.append([float(point_model[v.name]) for v in vars_def])
            train_y.append(score)
            evaluations += 1
            is_new_global_best = score > best_y
            if is_new_global_best:
                best_y = score
                best_x = point_raw
                if self.verbose > 0:
                    print(f"[bosmp] new best={best_y:.5f}")
            logger.log_evaluation(
                evaluation=evaluations,
                params_raw=point_raw,
                params_model={v.name: float(point_model[v.name]) for v in vars_def},
                reward=score,
                elapsed_sec=elapsed_sec,
                global_best_before=global_best_before,
                global_best_after=best_y,
                is_new_global_best=is_new_global_best,
                phase=phase,
                proposal_source=proposal_source,
                extra={"objective_cpu_sec": timing["cpu_sec"]},
            )
            return score

        for _ in range(n_warm):
            _evaluate(self._sample_model_point(self.rng, vars_def), "warm_start", "random_warm_start")

        gp_fallbacks = 0

        while len(train_y) < total_budget:
            if len(train_y) < 2:
                _evaluate(self._sample_model_point(self.rng, vars_def), "bayesian", "random_insufficient_data")
                continue

            bounds_t = torch.tensor(
                [
                    [v.lo_model for v in vars_def],
                    [v.hi_model for v in vars_def],
                ],
                dtype=dtype,
                device=device,
            )
            try:
                X_t = torch.tensor(np.asarray(train_X), dtype=dtype, device=device)
                y_t = torch.tensor(np.asarray(train_y).reshape(-1, 1), dtype=dtype, device=device)
                model = SingleTaskGP(
                    X_t,
                    y_t,
                    input_transform=Normalize(d=len(vars_def), bounds=bounds_t),
                    outcome_transform=Standardize(m=1),
                )
                mll = ExactMarginalLogLikelihood(model.likelihood, model)
                fit_gpytorch_mll(mll)

                acq = LogExpectedImprovement(model=model, best_f=float(y_t.max().item()))
                cand, _ = optimize_acqf(
                    acq_function=acq,
                    bounds=bounds_t,
                    q=1,
                    num_restarts=self.num_restarts,
                    raw_samples=self.raw_samples,
                )
                point_model = {v.name: float(cand[0, i].item()) for i, v in enumerate(vars_def)}
                proposal_source = "log_expected_improvement"
                if self._is_duplicate(point_model, train_X, vars_def):
                    logger.log_event(
                        {
                            "event": "duplicate_proposal",
                            "evaluation": evaluations + 1,
                            "candidate": point_model,
                            "duplicate_tol": self.duplicate_tol,
                        }
                    )
                    point_model = self._sample_model_point(self.rng, vars_def)
                    proposal_source = "random_duplicate_proposal"
            except (NameError, AttributeError, ImportError) as exc:
                raise RuntimeError(f"bosmp proposal step is broken, not merely failing: {exc!r}") from exc
            except Exception as exc:
                gp_fallbacks += 1
                if gp_fallbacks == 1:
                    print(f"Warning: bosmp GP proposal failed ({type(exc).__name__}: {exc}); "
                          "falling back to random sampling. Further occurrences counted in summary.json.")
                logger.log_event(
                    {
                        "event": "proposal_fallback",
                        "evaluation": evaluations + 1,
                        "fallback_reason": f"{type(exc).__name__}: {exc}",
                    }
                )
                point_model = self._sample_model_point(self.rng, vars_def)
                proposal_source = "random_gp_fallback"
            _evaluate(point_model, "bayesian", proposal_source)

        if gp_fallbacks:
            print(f"[bosmp] {gp_fallbacks}/{self.n_iter} GP proposals fell back to random sampling.")

        assert best_x is not None
        logger.write_summary(
            evaluations=evaluations,
            total_budget=total_budget,
            best_reward=best_y,
            best_params=best_x,
            status="completed",
        )
        logger.log_event(
            {
                "event": "optimization_completed",
                "evaluations": evaluations,
                "best_reward": best_y,
                "best_params": best_x,
            }
        )
        return best_x
