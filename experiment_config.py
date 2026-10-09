"""Dependency-light experiment configuration and override resolution."""

from dataclasses import dataclass
from typing import Any, Dict, List


def resolve_experiment_alpha(
    experiment_alpha: float | None,
    optimizer_kwargs: Dict[str, Any],
) -> float:
    """Resolve alpha with the experiment value taking precedence."""
    if experiment_alpha is not None:
        return float(experiment_alpha)
    try:
        return float(optimizer_kwargs["alpha"])
    except KeyError as exc:
        raise ValueError(
            "alpha must be supplied by the experiment or configured in the optimizer kwargs"
        ) from exc


def resolve_genetic_percentages_for_budget(
    optimizer_kwargs: Dict[str, Any], budget: int
) -> tuple[Dict[str, Any], int]:
    """Resolve percentage-based GA sizes into concrete candidate counts."""
    try:
        population_percentage = float(optimizer_kwargs["pop_size"])
        elitism_percentage = float(optimizer_kwargs["elitism"])
        generations = int(optimizer_kwargs["gens"])
    except KeyError as exc:
        raise ValueError(
            "Genetic configuration requires pop_size, gens, and elitism."
        ) from exc

    population_size = int(round(budget * population_percentage / 100.0))
    elite_count = max(1, int(round(budget * elitism_percentage / 100.0)))
    offspring_per_generation = population_size - elite_count
    initial_pop_size = max(
        elite_count, budget - generations * offspring_per_generation
    )
    realized_budget = initial_pop_size + generations * offspring_per_generation

    resolved_kwargs = {
        **optimizer_kwargs,
        "pop_size": population_size,
        "elitism": elite_count,
        "initial_pop_size": initial_pop_size,
    }
    return resolved_kwargs, realized_budget


@dataclass
class ExperimentConfig:
    task: str
    model_name: str
    model_kwargs: Dict[str, Any]
    metrics: Dict[str, Any]

    dataset: str
    loader_name: str
    loader_kwargs: Dict[str, Any]
    split: Dict[str, Any]

    compressor: str
    compressor_bounds: Dict[str, List[float]]
    compressor_space: Dict[str, Dict[str, str]]
    compressor_methods: Dict[str, Any]

    optimizer: str
    optimizer_kwargs: Dict[str, Any]
    alpha: float

    random_state: int
    out_dir: str
    logs_dir: str
    mlflow_enabled: bool = True
    seeded_trace_dirs: bool = False

    def __post_init__(self) -> None:
        """Make experiment alpha authoritative for every optimizer."""
        self.alpha = float(self.alpha)
        self.optimizer_kwargs = {**self.optimizer_kwargs, "alpha": self.alpha}
