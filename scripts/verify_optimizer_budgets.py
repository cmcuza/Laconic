"""Verify every optimizer spends exactly its evaluation budget."""
from __future__ import annotations

import argparse
import math
import os
import sys
import warnings

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
warnings.filterwarnings("ignore")

import numpy as np

from experiment_config import resolve_genetic_percentages_for_budget
from optimizer.utils import build_optimizer, bounds_list_to_tuple


GENETIC_PERCENTAGE_KWARGS = {
    "pop_size": 12,
    "gens": 8,
    "elitism": 1,
    "tournament_k": 3,
}

_GENETIC_BUDGET_IDENTITY = GENETIC_PERCENTAGE_KWARGS["pop_size"] + (
    GENETIC_PERCENTAGE_KWARGS["gens"]
    * (GENETIC_PERCENTAGE_KWARGS["pop_size"] - GENETIC_PERCENTAGE_KWARGS["elitism"])
)
if _GENETIC_BUDGET_IDENTITY != 100:
    raise AssertionError(
        "Genetic percentages are not budget-preserving: pop% + gens * (pop% - "
        f"elite%) = {_GENETIC_BUDGET_IDENTITY}, expected 100. Retune gens "
        "alongside pop_size/elitism."
    )

METHODS = [
    "PoorMansCompressionMean", "SwingFilter", "SlideFilter", "VW", "MixPiece",
    "DFT", "BitPackedQuantization", "BitPackedBUFF", "MacaqueS", "MacaqueV",
    "DeltaEncoding", "DeltatoDeltaPFOREncoding", "DeltaEliasGammaEncoding",
]

BOUNDS = bounds_list_to_tuple({
    "logical_method_index": [0, 5],
    "coefficient_method_index": [6, 9],
    "indices_method_index": [10, 12],
    "logical_method_error": [0.001, 0.15],
    "coefficient_method_error": [1.0e-7, 1.0e-2],
})

SPACE = {
    "logical_method_index": {"type": "int", "scale": "linear"},
    "coefficient_method_index": {"type": "int", "scale": "linear"},
    "indices_method_index": {"type": "int", "scale": "linear"},
    "logical_method_error": {"type": "float", "scale": "linear"},
    "coefficient_method_error": {"type": "float", "scale": "log"},
}

RESHAPED_BOUNDS = bounds_list_to_tuple({
    "logical_method_index": [0, 3],
    "coefficient_method_index": [4, 5],
    "indices_method_index": [6, 7],
    "logical_method_error": [0.001, 0.15],
    "coefficient_method_error": [1.0e-7, 1.0e-2],
})

ADAEDGE_BOUNDS = bounds_list_to_tuple({
    "method_index": [0, 2],
    "adaedge_error": [0.01, 0.3],
})

ADAEDGE_SPACE = {
    "method_index": {"type": "int", "scale": "linear"},
    "adaedge_error": {"type": "float", "scale": "linear"},
}

ADAEDGE_RESHAPED_BOUNDS = bounds_list_to_tuple({
    "method_index": [0, 1],
    "adaedge_error": [0.01, 0.3],
})


class _FakeBackend:
    def __init__(self, methods):
        self._methods = methods

    def params_from_vector(self, vec):
        return {k: (self._methods[int(round(v))] if k.endswith("_index") else float(v))
                for k, v in vec.items()}


class _Objective:
    def __init__(self, methods=METHODS):
        self.backend = _FakeBackend(methods)
        self.calls = 0

    def __call__(self, params):
        self.calls += 1
        li = float(params["logical_method_index"])
        ci = float(params["coefficient_method_index"])
        le = float(params["logical_method_error"])
        ce = float(params["coefficient_method_error"])
        return (
            0.5 * math.exp(-((li - 2.0) ** 2) / 2.0)
            + 0.3 * math.exp(-((ci - 8.0) ** 2) / 2.0)
            + 0.2 * math.exp(-((math.log10(max(le, 1e-9)) + 2.0) ** 2))
            + 0.1 * math.exp(-((math.log10(max(ce, 1e-9)) + 4.0) ** 2))
        )


class _AdaEdgeObjective:
    def __init__(self, methods=("mixpiece", "serfxor", "sz")):
        self.backend = _FakeBackend(list(methods))
        self.calls = 0

    def __call__(self, params):
        self.calls += 1
        index = float(params["method_index"])
        error = float(params["adaedge_error"])
        return (
            0.6 * math.exp(-((index - 1.0) ** 2) / 2.0)
            + 0.4 * math.exp(-((error - 0.08) ** 2) / 0.01)
        )


def _problem_for(name: str):
    if name == "adaedge":
        return ADAEDGE_BOUNDS, ADAEDGE_SPACE, _AdaEdgeObjective, ADAEDGE_RESHAPED_BOUNDS
    return BOUNDS, SPACE, _Objective, RESHAPED_BOUNDS


def _kwargs_for(name: str, budget: int) -> dict:
    common = dict(verbose=0, alpha=0.75, random_state=32)
    if name == "bosmp":
        init_points = 5
        return dict(common, init_points=init_points, n_iter=max(1, budget - init_points),
                    num_restarts=2, raw_samples=16, device="cpu", log_process=False)
    if name == "adaedge":
        init_points = 6
        return dict(common, init_points=init_points, n_iter=max(1, budget - init_points),
                    num_restarts=2, raw_samples=16, device="cpu", log_process=False)
    if name == "genetic":
        percentage_kwargs = dict(
            common,
            **GENETIC_PERCENTAGE_KWARGS,
            early_stop_patience=999,
            n_workers=1,
        )
        resolved_kwargs, _ = resolve_genetic_percentages_for_budget(
            percentage_kwargs, budget
        )
        return resolved_kwargs
    raise ValueError(f"No kwargs template for optimizer '{name}'")


def _budget_fidelity(name: str, budget: int, advertised: int) -> tuple[str, str]:
    if name in ("bosmp", "adaedge"):
        warm_start = 5 if name == "bosmp" else 6
        expected = max(budget, warm_start + 1)
        note = "floored at init_points + 1" if expected != budget else f"expected {budget}"
        return ("ok" if advertised == expected else "FAIL", note)
    if name == "genetic":
        drift = advertised - budget
        return ("ok" if drift == 0 else "FAIL", f"{drift:+d}; expected {budget}")
    raise ValueError(name)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--budgets", default="50,100,200",
                    help="Comma-separated budgets to test.")
    ap.add_argument("--optimizers", default="genetic,bosmp,adaedge",
                    help="Comma-separated optimizer names")
    args = ap.parse_args()
    budgets = [int(b) for b in args.budgets.split(",")]
    names = [n.strip() for n in args.optimizers.split(",")]

    failures = []
    for name in names:
        for budget in budgets:
            np.random.seed(32)
            bounds, space, make_objective, _ = _problem_for(name)
            objective = make_objective()
            optimizer = build_optimizer(name, _kwargs_for(name, budget))
            try:
                optimizer.maximize(objective, search_space=bounds, space_definition=space,
                                   log_dir=None, run_metadata=None)
            except Exception as exc:
                failures.append((name, budget, f"CRASH {type(exc).__name__}: {exc}"))
                print(f"[FAIL] {name:>19} budget={budget:>4}: CRASH {exc}")
                continue
            advertised = optimizer.total_budget
            if objective.calls != advertised:
                failures.append((name, budget,
                                 f"spent {objective.calls} evals but advertises "
                                 f"total_budget={advertised}"))
                print(f"[{'MISMATCH':>8}] {name:>19} budget={budget:>4} "
                      f"evals={objective.calls:>4} advertised={advertised:>4}")
                continue

            verdict, note = _budget_fidelity(name, budget, advertised)
            if verdict == "FAIL":
                failures.append((name, budget, f"advertised={advertised} requested={budget} ({note})"))
            print(f"[{verdict:>8}] {name:>19} budget={budget:>4} "
                  f"evals={objective.calls:>4} advertised={advertised:>4}  {note}")

    for name in names:
        if name in ("genetic", "bosmp"):
            continue
        np.random.seed(32)
        _, space, make_objective, reshaped_bounds = _problem_for(name)
        objective = make_objective(methods=[f"m{i}" for i in range(8)])
        optimizer = build_optimizer(name, _kwargs_for(name, 30))
        try:
            optimizer.maximize(objective, search_space=reshaped_bounds, space_definition=space,
                               log_dir=None, run_metadata=None)
            print(f"[      ok] {name:>19} reshaped search space: evals={objective.calls}")
        except Exception as exc:
            note = ""
            print(f"[{'known' if note else 'FAIL':>8}] {name:>19} reshaped search space: "
                  f"{type(exc).__name__}{note}")
            if not note:
                failures.append((name, "reshaped", f"{type(exc).__name__}: {exc}"))

    print()
    if failures:
        print("FAILURES:")
        for failure in failures:
            print("  ", failure)
        return 1
    print("All optimizers spent what they advertise, and advertise what their "
          "budget entitles them to.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
