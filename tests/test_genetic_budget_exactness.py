"""GeneticOptimizer spends exactly the requested budget."""
from pathlib import Path
import sys

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from experiment_config import resolve_genetic_percentages_for_budget
from optimizer.genetic import GeneticOptimizer


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
    def __init__(self):
        self.calls = 0

    def __call__(self, candidate):
        self.calls += 1
        return -((candidate["x"] - 0.3) ** 2 + (candidate["y"] - 0.7) ** 2)


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
