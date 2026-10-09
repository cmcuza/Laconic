"""End-to-end check that GeneticOptimizer/PreferenceGeneticOptimizer spend
exactly the requested evaluation budget for arbitrary (not just 100/200/500
ladder) budgets -- not just that the resolver's arithmetic works out
(test_genetic_budget_scaling.py already covers that), but that a real
maximize() run against a real objective actually makes that many calls.
`total_budget` is what tags every result row's `n_evaluations` and its
`budget_<N>` directory, so a drift here would mislabel results.
"""
from pathlib import Path
import sys

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from experiment_config import resolve_genetic_percentages_for_budget
from optimizer.genetic import GeneticOptimizer
from optimizer.preference import PreferenceGeneticOptimizer


PERCENTAGE_KWARGS = {
    "pop_size": 12,
    "gens": 8,
    "elitism": 1,
    "tournament_k": 3,
}

SEARCH_SPACE = {"x": [0.0, 1.0], "y": [0.0, 1.0]}
SPACE_DEFINITION = {
    "x": {"type": "float", "scale": "linear"},
    "y": {"type": "float", "scale": "linear"},
}

BUDGETS = [50, 75, 100, 150, 200]


class _CountingObjective:
    """Cheap deterministic objective; counts every call it receives."""

    def __init__(self):
        self.calls = 0

    def __call__(self, candidate):
        self.calls += 1
        return -((candidate["x"] - 0.3) ** 2 + (candidate["y"] - 0.7) ** 2)


class _CountingComponentsObjective:
    """PreferenceGeneticOptimizer's contract: candidate -> (task_metric, avg_cr).

    Both driven by the same variable in opposite directions -- a genuine
    Pareto trade-off, so the non-dominated front spans a range instead of
    collapsing to the single point that independent x/y objectives produce.
    """

    def __init__(self):
        self.calls = 0

    def __call__(self, candidate):
        self.calls += 1
        error = candidate["x"]
        return 1.0 - error, 1.0 + error


@pytest.mark.parametrize("budget", BUDGETS)
def test_genetic_optimizer_spends_exactly_the_requested_budget(budget: int) -> None:
    resolved_kwargs, realized_budget = resolve_genetic_percentages_for_budget(
        PERCENTAGE_KWARGS, budget
    )
    assert realized_budget == budget

    optimizer = GeneticOptimizer(
        **resolved_kwargs, n_workers=1, early_stop_patience=10**9, verbose=0,
    )
    objective = _CountingObjective()

    optimizer.maximize(
        objective, SEARCH_SPACE, SPACE_DEFINITION, log_dir=None, run_metadata=None,
    )

    assert optimizer.total_budget == budget
    assert objective.calls == budget


@pytest.mark.parametrize("budget", BUDGETS)
def test_preference_optimizer_spends_exactly_the_requested_budget(
    budget: int,
) -> None:
    resolved_kwargs, realized_budget = resolve_genetic_percentages_for_budget(
        PERCENTAGE_KWARGS, budget
    )
    assert realized_budget == budget

    optimizer = PreferenceGeneticOptimizer(
        alpha=0.75, **resolved_kwargs, n_workers=1, verbose=0,
    )
    objective = _CountingComponentsObjective()

    optimizer.maximize(
        objective, SEARCH_SPACE, SPACE_DEFINITION, log_dir=None, run_metadata=None,
    )

    assert optimizer.total_budget == budget
    assert objective.calls == budget
