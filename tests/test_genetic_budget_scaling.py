from pathlib import Path
import sys

import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from experiment_config import resolve_genetic_percentages_for_budget


PERCENTAGE_KWARGS = {
    "pop_size": 12,
    "gens": 8,
    "elitism": 1,
    "tournament_k": 3,
}


@pytest.mark.parametrize(
    (
        "budget",
        "expected_population",
        "expected_elitism",
        "expected_evaluations",
    ),
    [
        (100, 12, 1, 100),
        (200, 24, 2, 200),
        (500, 60, 5, 500),
    ],
)
def test_standard_genetic_budgets_resolve_percentages(
    budget: int,
    expected_population: int,
    expected_elitism: int,
    expected_evaluations: int,
) -> None:
    resolved_kwargs, realized_budget = resolve_genetic_percentages_for_budget(
        PERCENTAGE_KWARGS, budget
    )

    assert resolved_kwargs["pop_size"] == expected_population
    assert resolved_kwargs["elitism"] == expected_elitism
    assert resolved_kwargs["tournament_k"] == 3
    assert resolved_kwargs["gens"] == 8
    assert realized_budget == expected_evaluations
    assert expected_population / budget == pytest.approx(0.12)
    assert expected_elitism == round(budget * 0.01)


def test_genetic_percentage_resolution_does_not_mutate_configuration() -> None:
    original_kwargs = dict(PERCENTAGE_KWARGS)

    resolve_genetic_percentages_for_budget(original_kwargs, 500)

    assert original_kwargs == PERCENTAGE_KWARGS


@pytest.mark.parametrize("budget", [100, 200, 500])
def test_ladder_budgets_leave_room_for_tournament_children(budget: int) -> None:
    resolved_kwargs, _ = resolve_genetic_percentages_for_budget(
        PERCENTAGE_KWARGS, budget
    )
    tournament_children = resolved_kwargs["pop_size"] - 2 * resolved_kwargs["elitism"]
    assert tournament_children >= 1


def test_genetic_elitism_percentage_must_be_below_population_percentage() -> None:
    invalid_kwargs = {**PERCENTAGE_KWARGS, "elitism": 12}

    with pytest.raises(ValueError, match="lower than pop_size"):
        resolve_genetic_percentages_for_budget(invalid_kwargs, 100)
