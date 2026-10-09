"""Dependency-light experiment configuration and override resolution."""

from dataclasses import dataclass
from typing import Any, Dict, List


def resolve_experiment_alpha(
    experiment_alpha: float | None,
    optimizer_kwargs: Dict[str, Any],
) -> float:
    """Resolve alpha with the experiment value taking precedence.

    Optimizer YAMLs retain ``kwargs.alpha`` as a fallback for direct or legacy
    invocations, but an alpha supplied by the experiment runner is the source
    of truth for that experiment.
    """
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
    """Resolve percentage-based GA sizes into concrete candidate counts.

    Both ``pop_size`` and ``elitism`` are percentages of the evaluation
    budget. ``gens`` and ``tournament_k`` are absolute algorithm parameters and
    are left unchanged.

    The percentages are applied as written, with one floor: ``elitism`` is at
    least 1 at any budget, because an elitist GA with no elite is a different
    algorithm, not a smaller one.

    Rounding ``pop_size``/``elitism`` to integers rarely lands the steady-state
    formula (``population_size + gens * (population_size - elite_count)``)
    exactly on ``budget``. Rather than let genetic silently under/overshoot,
    the gap is absorbed entirely by the *initial* random population
    (``initial_pop_size``): every later generation still produces exactly
    ``population_size - elite_count`` offspring, so the steady-state search
    this schedule was tuned at (selection pressure, elitism ratio) is
    unaffected from generation 1 onward - only the amount of upfront random
    exploration changes. ``initial_pop_size`` floors at ``elite_count`` so
    generation 1 can still draw a full elite set; only a budget too small for
    that (well below the 50-evaluation floor experiments already stay above)
    can't be hit exactly, which shows up as a ``realized_budget`` short of
    ``budget``.
    """
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
    # task / analytics
    task: str
    model_name: str
    model_kwargs: Dict[str, Any]
    metrics: Dict[str, Any]

    # data
    dataset: str
    loader_name: str
    loader_kwargs: Dict[str, Any]
    split: Dict[str, Any]

    # compression
    compressor: str
    compressor_bounds: Dict[str, List[float]]
    compressor_space: Dict[str, Dict[str, str]]
    compressor_methods: Dict[str, Any]

    # optimizer
    optimizer: str
    optimizer_kwargs: Dict[str, Any]
    alpha: float

    # execution
    random_state: int
    out_dir: str
    logs_dir: str
    mlflow_enabled: bool = True
    # TRANSITIONAL: put an rs<N> segment in the optimizer trace path so two runs
    # differing only by random_state cannot overwrite each other. Off by default
    # while sweeps written under the old layout are still in flight - see
    # optimizer/run_logger.py for the retirement plan.
    seeded_trace_dirs: bool = False

    def __post_init__(self) -> None:
        """Make experiment alpha authoritative for every optimizer.

        The objective reads ``cfg.alpha`` while optimizer paths and metadata
        read the constructor's ``alpha`` kwarg. Keeping them synchronized here
        prevents conflicting optimizer YAML values from splitting one run
        across different alpha identities. The kwargs are copied so resolving
        one experiment does not mutate a shared loaded YAML configuration.
        """
        self.alpha = float(self.alpha)
        self.optimizer_kwargs = {**self.optimizer_kwargs, "alpha": self.alpha}
