from __future__ import annotations
from typing import Any, Dict, List, Protocol, Tuple
from optimizer.bosmp import BOSimple
from optimizer.genetic import GeneticOptimizer
from optimizer.adaedge import AdaEdgeOptimizer
from optimizer.space import parse_bounds_pair


class Optimizer(Protocol):
    """Interface every optimizer implements."""
    total_budget: int
    last_run_dir: str | None

    def maximize(
        self,
        objective: Any,
        search_space: Dict[str, Any],
        space_definition: Dict[str, Any],
        log_dir: str | None,
        run_metadata: Dict[str, Any] | None,
        **kwargs: Any,
    ) -> Dict[str, float]: ...


def build_optimizer(name: str, kwargs: Dict[str, Any]) -> Optimizer:
    if name == "bosmp": return BOSimple(**kwargs)
    if name == "genetic": return GeneticOptimizer(**kwargs)
    if name == "adaedge": return AdaEdgeOptimizer(**kwargs)
    raise ValueError(f"Unknown tuner: {name}")


def bounds_list_to_tuple(bounds: Dict[str, List[float]]) -> Dict[str, Tuple[float, float]]:
    """Normalize a config's ``bounds`` block into ``{param: (lo, hi)}``."""
    return {key: parse_bounds_pair(value, key) for key, value in bounds.items()}
