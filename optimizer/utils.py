from __future__ import annotations
from typing import Any, Dict, List, Protocol, Tuple
from optimizer.random import RandomOptimizer
from optimizer.bomab import BOMABOptimizer
from optimizer.successive_halving import SuccessiveHalvingOptimizer
from optimizer.adaptive_halving import AdaptiveHalvingOptimizer
from optimizer.preference import PreferenceGeneticOptimizer
from optimizer.bosmp import BOSimple
from optimizer.genetic import GeneticOptimizer
from optimizer.adaedge import AdaEdgeOptimizer
from optimizer.space import parse_bounds_pair

# Trimmed to the optimizers currently in active use (see project CLAUDE.md):
# random/successive_halving/genetic for tersets_mab, bosmp for the sz/mixpiece/
# serfxor baselines. bayesian/bossa/gabo/prebo/rfei/shrfei/mfrfei are parked,
# not deleted - port them back the same way (add the module + a branch here).

class Optimizer(Protocol):
    """What every optimizer must implement (see CLAUDE.md).

    Every argument is required: an optimizer that silently ran without a
    space definition or without writing its trace is the failure mode this
    signature exists to prevent. Pass ``log_dir=None`` to opt out of file
    logging explicitly (only scripts/verify_optimizer_budgets.py does).

    This replaced a hand-maintained ``LOGGING_OPTIMIZERS`` set the runners
    consulted before deciding whether to pass log_dir/run_metadata - an
    optimizer missing from it produced no trace files and said nothing.
    """
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
    if name == "random": return RandomOptimizer(**kwargs)
    if name == "successive_halving": return SuccessiveHalvingOptimizer(**kwargs)
    if name == "adaptive_halving": return AdaptiveHalvingOptimizer(**kwargs)
    # Not a CLI-exposed --optimizer choice (see run_experiments.py) - kept only
    # because SuccessiveHalvingOptimizer subclasses BOMABOptimizer and reuses
    # its logging internals. Buildable here for direct/programmatic use.
    if name == "bomab": return BOMABOptimizer(**kwargs)
    if name == "bosmp": return BOSimple(**kwargs)
    if name == "genetic": return GeneticOptimizer(**kwargs)
    if name == "adaedge": return AdaEdgeOptimizer(**kwargs)
    if name == "preference": return PreferenceGeneticOptimizer(**kwargs)
    raise ValueError(f"Unknown tuner: {name}")


def bounds_list_to_tuple(bounds: Dict[str, List[float]]) -> Dict[str, Tuple[float, float]]:
    """Normalize a config's ``bounds`` block into ``{param: (lo, hi)}``.

    Every runner funnels ``cfg.compressor_bounds`` through here before handing a
    search space to any optimizer, so this is the one place a malformed bounds
    entry can be caught for all of them - hence the validation lives in
    ``parse_bounds_pair`` and is shared rather than re-implemented downstream.
    """
    return {key: parse_bounds_pair(value, key) for key, value in bounds.items()}
