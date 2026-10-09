from pathlib import Path
import math
import sys

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from optimizer.space import Space


class DeterministicMutation:
    def random(self):
        return 0.0

    def normal(self, mean, std):
        return 0.4

    def integers(self, lower_bound, upper_bound):
        return upper_bound - 1


def test_linear_integer_sampling_is_discrete_and_includes_both_bounds() -> None:
    class RecordingRng:
        def __init__(self):
            self.calls = []

        def integers(self, low, high):
            self.calls.append((low, high))
            return high - 1

        def uniform(self, low, high):
            raise AssertionError(
                "linear integer sampling must not use continuous uniform"
            )

    space = Space(
        {"method_index": (0, 5)},
        {"method_index": {"type": "int", "scale": "linear"}},
    )
    rng = RecordingRng()

    assert space.sample(rng) == {"method_index": 5.0}
    assert rng.calls == [(0, 6)]


def test_mutation_restores_integer_variables_before_returning() -> None:
    space = Space(
        {"method_index": (0, 5)},
        {"method_index": {"type": "int", "scale": "linear"}},
    )

    mutated = space.mutate(DeterministicMutation(), {"method_index": 2.0}, prob=1.0)

    assert mutated == {"method_index": 5.0}


def test_integer_mutation_samples_every_other_primitive_uniformly() -> None:
    class CyclingMutation:
        def __init__(self):
            self.integer_samples = iter(range(5))

        def random(self):
            return 0.0

        def integers(self, lower_bound, upper_bound):
            assert (lower_bound, upper_bound) == (0, 5)
            return next(self.integer_samples)

    space = Space(
        {"method_index": (0, 5)},
        {"method_index": {"type": "int", "scale": "linear"}},
    )
    rng = CyclingMutation()

    sampled = [
        space.mutate(rng, {"method_index": 2.0}, prob=1.0)["method_index"]
        for _ in range(5)
    ]

    assert sampled == [0.0, 1.0, 3.0, 4.0, 5.0]


def test_crossover_selects_each_parent_primitive_with_equal_probability() -> None:
    class ParentChoiceRng:
        def __init__(self, value):
            self.value = value

        def random(self):
            return self.value

    space = Space(
        {"method_index": (0, 5)},
        {"method_index": {"type": "int", "scale": "linear"}},
    )
    low = {"method_index": 0.0}
    high = {"method_index": 5.0}

    left = space.crossover(ParentChoiceRng(0.49), low, high)
    right = space.crossover(ParentChoiceRng(0.50), low, high)

    assert left == low
    assert right == high


def test_clip_preserves_float_variables_while_casting_integers() -> None:
    space = Space(
        {"method_index": (0, 5), "error": (0.0, 1.0)},
        {
            "method_index": {"type": "int", "scale": "linear"},
            "error": {"type": "float", "scale": "linear"},
        },
    )

    assert space.clip({"method_index": 2.6, "error": 0.35}) == {
        "method_index": 3.0,
        "error": 0.35,
    }


def test_significant_digits_canonicalize_equivalent_error_bounds() -> None:
    space = Space(
        {
            "logical_method_error": (0.001, 0.15),
            "coefficient_method_error": (1.0e-7, 1.0e-2),
        },
        {
            "logical_method_error": {
                "type": "float",
                "scale": "linear",
                "significant_digits": 2,
            },
            "coefficient_method_error": {
                "type": "float",
                "scale": "log",
                "significant_digits": 2,
            },
        },
    )

    class ErrorSamplingRng:
        def __init__(self):
            self.values = iter([0.069421, math.log(0.00601)])

        def uniform(self, lower_bound, upper_bound):
            return next(self.values)

    assert space.sample(ErrorSamplingRng()) == {
        "logical_method_error": 0.069,
        "coefficient_method_error": 0.006,
    }

    first_candidate = space.clip(
        {
            "logical_method_error": 0.069421,
            "coefficient_method_error": 0.006,
        }
    )
    second_candidate = space.clip(
        {
            "logical_method_error": 0.069497,
            "coefficient_method_error": 0.00601,
        }
    )

    assert (
        first_candidate
        == second_candidate
        == {
            "logical_method_error": 0.069,
            "coefficient_method_error": 0.006,
        }
    )
    assert space.clip(
        {
            "logical_method_error": 0.001234,
            "coefficient_method_error": 1.234e-6,
        }
    ) == {
        "logical_method_error": 0.0012,
        "coefficient_method_error": 1.2e-6,
    }


@pytest.mark.parametrize("significant_digits", [0, -1, 2.5, True])
def test_significant_digits_must_be_a_positive_integer(significant_digits) -> None:
    with pytest.raises(ValueError, match="significant_digits"):
        Space(
            {"error": (0.001, 0.15)},
            {
                "error": {
                    "type": "float",
                    "scale": "linear",
                    "significant_digits": significant_digits,
                }
            },
        )
