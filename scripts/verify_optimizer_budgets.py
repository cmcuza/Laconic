"""Verify every optimizer honors an arbitrary evaluation budget - no datasets,
no models, no botorch-sized runtimes beyond the GP fits themselves.

Three checks per optimizer:
1. Self-consistency: the objective call count equals the optimizer's own
   advertised ``total_budget``. That number is what lands in every results
   row's ``n_evaluations`` and in the ``budget_<N>`` directory segment, so an
   optimizer that advertises one budget and spends another mislabels its own
   results.
2. Budget fidelity: the advertised ``total_budget`` matches what the requested
   budget entitles it to (exact for random/successive_halving/adaptive_halving
   /genetic/preference; bosmp/adaedge floor at init_points + 1).
3. Generality (halving optimizers): runs on a *reshaped* search space
   (different method count / index ranges) - this is what catches hard-coded
   pipeline indices (the legacy successive_halving crashes here by design;
   it's reported, not counted as a failure).

Check 2 is deliberately derived from each optimizer's *contract*, never by
calling the same code that configured the optimizer - an expectation computed
by the function under test agrees with itself no matter how wrong it is. For
genetic that contract is the percentage identity in
``_GENETIC_BUDGET_IDENTITY`` below rather than a re-typed copy of
``resolve_genetic_percentages_for_budget``.

Run after adding/changing an optimizer, before any real experiment:

    python scripts/verify_optimizer_budgets.py
    python scripts/verify_optimizer_budgets.py --budgets 5,50,250 --optimizers adaptive_halving
"""
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
from optimizer.oracle import DeterministicPreferenceOracle


GENETIC_PERCENTAGE_KWARGS = {
    "pop_size": 12,
    "gens": 8,
    "elitism": 1,
    "tournament_k": 3,
}

# What makes the percentages exact without help from initial_pop_size at all.
# Genetic's steady state spends pop_size + gens * (pop_size - elitism)
# evaluations; with pop_size and elitism expressed as percentages of the
# budget, that total is the budget itself only when the percentages satisfy:
#
#     pop% + gens * (pop% - elite%) == 100
#
# This is the property the ladder in docs/GENETIC_OPTIMIZER.md rests on. Off
# this identity, resolve_genetic_percentages_for_budget() still lands exactly
# on the requested budget at budget 100 (initial_pop_size absorbs the gap),
# but only by shrinking or inflating the initial population away from its
# intended size - e.g. raising pop_size to 15% while leaving gens at 8 would
# force initial_pop_size down to its elite_count floor (collapsing the
# population meant to be 15 down to 1) rather than landing cleanly on 100
# evaluations at the intended population size. Checked at import, so it fails
# here rather than three sweeps later.
_GENETIC_BUDGET_IDENTITY = GENETIC_PERCENTAGE_KWARGS["pop_size"] + (
    GENETIC_PERCENTAGE_KWARGS["gens"]
    * (GENETIC_PERCENTAGE_KWARGS["pop_size"] - GENETIC_PERCENTAGE_KWARGS["elitism"])
)
if _GENETIC_BUDGET_IDENTITY != 100:
    raise AssertionError(
        "Genetic percentages are not budget-preserving: pop% + gens * (pop% - "
        f"elite%) = {_GENETIC_BUDGET_IDENTITY}, expected 100. Retune gens "
        "alongside pop_size/elitism (see docs/GENETIC_OPTIMIZER.md)."
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

# adaedge searches its own shape (one method index + one shared error bound),
# not the TerseTS triple, so it gets its own problem definition below.
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
    """Deterministic multimodal objective; counts calls."""

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
    """Deterministic objective over (method_index, adaedge_error); counts calls."""

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


class _PreferenceObjective:
    """Deterministic (task_metric, avg_cr) pair with a genuine tradeoff -
    PreferenceGeneticOptimizer's contract is objective(candidate) ->
    (task_metric, avg_cr), not the rest of the repo's single scalar."""

    def __init__(self, methods=METHODS):
        self.backend = _FakeBackend(methods)
        self.calls = 0

    def __call__(self, params):
        self.calls += 1
        error = float(params["logical_method_error"])  # [0.001, 0.15]
        return 1.0 - 6.0 * error, 1.0 + 60.0 * error


def _problem_for(name: str):
    """(bounds, space, objective_factory, reshaped_bounds) for an optimizer.

    Optimizers do not all search the same parameter shape, so the harness picks
    the problem that matches the one under test instead of forcing every
    optimizer through the TerseTS triple.
    """
    if name == "adaedge":
        return ADAEDGE_BOUNDS, ADAEDGE_SPACE, _AdaEdgeObjective, ADAEDGE_RESHAPED_BOUNDS
    if name == "preference":
        return BOUNDS, SPACE, _PreferenceObjective, RESHAPED_BOUNDS
    return BOUNDS, SPACE, _Objective, RESHAPED_BOUNDS


def _kwargs_for(name: str, budget: int) -> dict:
    """cfg/optimizer/<name>.yaml-shaped kwargs with run_experiments.py's
    _apply_budget_override translation applied."""
    common = dict(verbose=0, alpha=0.75, random_state=32)
    if name == "random":
        return dict(common, n_iter=budget, log_process=False)
    if name == "adaptive_halving":
        return dict(common, total_budget=budget, init_points=3, device="cpu", log_process=False)
    if name == "successive_halving":
        return dict(common, total_budget=budget, init_points=3, n_iter=max(1, budget - 3),
                    device="cpu", log_process=False)
    if name == "bosmp":
        init_points = 5
        return dict(common, init_points=init_points, n_iter=max(1, budget - init_points),
                    num_restarts=2, raw_samples=16, device="cpu", log_process=False)
    if name == "adaedge":
        init_points = 6  # multiple of the 3 arms, like the real config
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
    if name == "preference":
        percentage_kwargs = dict(
            common, **GENETIC_PERCENTAGE_KWARGS, n_workers=1,
            oracle=DeterministicPreferenceOracle(hidden_weight=common["alpha"], reliability=1.0, beta=300.0),
        )
        resolved_kwargs, _ = resolve_genetic_percentages_for_budget(
            percentage_kwargs, budget
        )
        return resolved_kwargs
    raise ValueError(f"No kwargs template for optimizer '{name}'")


def _budget_fidelity(name: str, budget: int, advertised: int) -> tuple[str, str]:
    """Judge an optimizer's advertised total_budget against the requested one.

    Returns ``(verdict, note)`` where verdict is "ok" or "FAIL".

    Nothing here calls the code that built the optimizer: the expectation is
    restated from each optimizer's documented contract so that a bug in the
    translation shows up as a disagreement instead of being reproduced on both
    sides of the comparison.
    """
    if name in ("random", "adaptive_halving", "successive_halving"):
        return ("ok" if advertised == budget else "FAIL", f"expected {budget}")
    if name in ("bosmp", "adaedge"):
        warm_start = 5 if name == "bosmp" else 6  # must match _kwargs_for
        expected = max(budget, warm_start + 1)
        note = "floored at init_points + 1" if expected != budget else f"expected {budget}"
        return ("ok" if advertised == expected else "FAIL", note)
    if name in ("genetic", "preference"):
        # Any rounding gap between pop_size/elitism's percentages and budget is
        # absorbed by the initial population alone (initial_pop_size in
        # resolve_genetic_percentages_for_budget), so this is exact the same
        # way random/adaptive_halving are - not just on the 100/200/500 ladder.
        drift = advertised - budget
        return ("ok" if drift == 0 else "FAIL", f"{drift:+d}; expected {budget}")
    raise ValueError(name)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--budgets", default="50,100,200",
                    help="Comma-separated budgets to test (default 50,100,200). "
                         "Experiments never run below 50, so smaller budgets are "
                         "not a supported configuration and are not exercised.")
    ap.add_argument("--optimizers", default="adaptive_halving,random,genetic,bosmp,adaedge,preference",
                    help="Comma-separated optimizer names (successive_halving may be added; "
                         "it is excluded by default as the legacy implementation)")
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
            # 1. The optimizer spent exactly what it advertises. total_budget is
            #    what becomes n_evaluations and the budget_<N> path segment, so
            #    a mismatch mislabels the run's own results.
            advertised = optimizer.total_budget
            if objective.calls != advertised:
                failures.append((name, budget,
                                 f"spent {objective.calls} evals but advertises "
                                 f"total_budget={advertised}"))
                print(f"[{'MISMATCH':>8}] {name:>19} budget={budget:>4} "
                      f"evals={objective.calls:>4} advertised={advertised:>4}")
                continue

            # 2. What it advertises is what the requested budget entitles it to.
            verdict, note = _budget_fidelity(name, budget, advertised)
            if verdict == "FAIL":
                failures.append((name, budget, f"advertised={advertised} requested={budget} ({note})"))
            print(f"[{verdict:>8}] {name:>19} budget={budget:>4} "
                  f"evals={objective.calls:>4} advertised={advertised:>4}  {note}")

    # Generality check: reshaped search space must not crash arm-aware optimizers.
    for name in names:
        if name in ("genetic", "bosmp", "preference"):
            continue  # generic-space optimizers, nothing arm-shaped to check
        np.random.seed(32)
        _, space, make_objective, reshaped_bounds = _problem_for(name)
        objective = make_objective(methods=[f"m{i}" for i in range(8)])
        optimizer = build_optimizer(name, _kwargs_for(name, 30))
        try:
            optimizer.maximize(objective, search_space=reshaped_bounds, space_definition=space,
                               log_dir=None, run_metadata=None)
            print(f"[      ok] {name:>19} reshaped search space: evals={objective.calls}")
        except Exception as exc:
            note = " (known: hard-coded arm indices)" if name == "successive_halving" else ""
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
