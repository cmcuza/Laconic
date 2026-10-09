from pathlib import Path
import sys

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import optimizer.genetic as genetic_module
from optimizer.genetic import GeneticOptimizer


def _optimizer(**overrides) -> GeneticOptimizer:
    kwargs = {
        "pop_size": 4,
        "gens": 2,
        "elitism": 1,
        "tournament_k": 3,
        "cx_prob": 0.0,
        "mut_prob": 0.0,
        "early_stop_patience": 99,
        "random_state": 7,
        "n_workers": 1,
        "verbose": 0,
    }
    kwargs.update(overrides)
    return GeneticOptimizer(**kwargs)


def test_tournament_samples_unique_contestants_and_honors_exclusion() -> None:
    class RecordingRng:
        def __init__(self):
            self.calls = []

        def choice(self, eligible_count, *, size, replace):
            self.calls.append((eligible_count, size, replace))
            return np.arange(size)

    rng = RecordingRng()
    optimizer = _optimizer()

    winner = optimizer._tournament(rng, [0.1, 0.9, 0.3, 0.2], excluded_parent_index=1)

    assert winner == 2
    assert rng.calls == [(3, 3, False)]


def test_parent_tournaments_return_distinct_candidates() -> None:
    """A crossover's second parent is drawn excluding the first."""
    optimizer = _optimizer()

    for seed in range(50):
        rng = np.random.default_rng(seed)
        population_scores = [0.1, 0.9, 0.3, 0.2]
        first_parent_index = optimizer._tournament(rng, population_scores)
        second_parent_index = optimizer._tournament(
            rng, population_scores, excluded_parent_index=first_parent_index
        )
        assert first_parent_index != second_parent_index


def test_unique_child_returns_its_reusable_candidate_key() -> None:
    class UniqueSpace:
        @staticmethod
        def crossover(rng, first_parent, second_parent):
            return {"x": 0.5}

    optimizer = _optimizer(cx_prob=1.0, mut_prob=0.0)
    population = [{"x": 0.1}, {"x": 0.2}, {"x": 0.3}, {"x": 0.4}]
    parameter_names = ["x"]
    seen_keys = {
        optimizer._candidate_key(candidate, parameter_names) for candidate in population
    }

    offspring, offspring_key = optimizer._propose_unique_child(
        UniqueSpace(), population, [0.1, 0.2, 0.3, 0.4], parameter_names, seen_keys
    )

    assert offspring_key == optimizer._candidate_key(offspring, parameter_names)
    assert offspring_key not in seen_keys


def test_no_crossover_selects_only_one_parent() -> None:
    class ForcedMutationSpace:
        @staticmethod
        def mutate(rng, candidate, **kwargs):
            return {"x": 0.5}

    optimizer = _optimizer(cx_prob=0.0, mut_prob=0.0)
    tournament_calls = []

    def tournament(rng, population_scores, excluded_parent_index=None):
        tournament_calls.append(excluded_parent_index)
        return 0

    optimizer._tournament = tournament
    proposal = optimizer._propose_unique_child(
        ForcedMutationSpace(),
        [{"x": 0.1}, {"x": 0.2}],
        [0.1, 0.2],
        ["x"],
        {(0.1,), (0.2,)},
    )

    assert proposal == ({"x": 0.5}, (0.5,))
    assert tournament_calls == [None]


def test_initial_population_does_not_evaluate_duplicates() -> None:
    class Backend:
        @staticmethod
        def params_from_vector(candidate):
            return candidate

    class Objective:
        backend = Backend()

        def __init__(self):
            self.evaluated = []

        def __call__(self, candidate):
            self.evaluated.append(candidate["x"])
            return candidate["x"]

    objective = Objective()
    optimizer = _optimizer(pop_size=4)
    optimizer.maximize(
        objective,
        search_space={"x": (0, 2)},
        space_definition={"x": {"type": "int", "scale": "linear"}},
        log_dir=None,
        run_metadata=None,
    )

    assert sorted(objective.evaluated) == [0.0, 1.0, 2.0]


def test_parallel_executor_initializes_objective_once(monkeypatch) -> None:
    class Backend:
        @staticmethod
        def params_from_vector(candidate):
            return candidate

    class Objective:
        backend = Backend()

        def __init__(self):
            self.calls = 0

        def __call__(self, candidate):
            self.calls += 1
            return candidate["x"]

    class RecordingExecutor:
        initializer_calls = 0
        mapped_functions = []

        def __init__(
            self,
            *,
            max_workers,
            mp_context,
            initializer,
            initargs,
        ):
            self._initializer = initializer
            self._initargs = initargs
            initializer(*initargs)
            type(self).initializer_calls += 1

        def map(self, function, candidates, timeout):
            type(self).mapped_functions.append(function)
            return [function(candidate) for candidate in candidates]

        def shutdown(self, *, wait):
            return None

    monkeypatch.setattr(
        genetic_module.cf, "ProcessPoolExecutor", RecordingExecutor
    )
    objective = Objective()
    optimizer = _optimizer(pop_size=4, gens=0, n_workers=2)
    optimizer.maximize(
        objective,
        search_space={"x": (0.0, 1.0)},
        space_definition={"x": {"type": "float", "scale": "linear"}},
        log_dir=None,
        run_metadata=None,
    )

    assert RecordingExecutor.initializer_calls == 1
    assert RecordingExecutor.mapped_functions == [
        genetic_module._evaluate_worker_candidate
    ]
    assert objective.calls == 4


def test_clone_prone_configuration_does_not_repeat_evaluations() -> None:
    class Backend:
        @staticmethod
        def params_from_vector(candidate):
            return candidate

    class Objective:
        backend = Backend()

        def __init__(self):
            self.evaluated = []

        def __call__(self, candidate):
            key = tuple(candidate.items())
            self.evaluated.append(key)
            return candidate["x"]

    objective = Objective()
    optimizer = _optimizer()

    optimizer.maximize(
        objective,
        search_space={"x": (0.0, 1.0)},
        space_definition={"x": {"type": "float", "scale": "linear"}},
        log_dir=None,
        run_metadata=None,
    )

    assert len(objective.evaluated) == optimizer.total_budget
    assert len(set(objective.evaluated)) == optimizer.total_budget
