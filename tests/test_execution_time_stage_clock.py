"""Tests for stage_clock accumulation."""
from __future__ import annotations

import time

import pytest

from optimizer.adaedge import AdaEdgeOptimizer
from optimizer.genetic import GeneticOptimizer
from profiling import stage_clock
from scripts.verify_optimizer_budgets import (
    ADAEDGE_BOUNDS,
    ADAEDGE_SPACE,
    BOUNDS,
    SPACE,
    _AdaEdgeObjective,
    _Objective,
)


def _burn(n: int = 500) -> int:
    total = 0
    for i in range(n):
        total += i * i
    return total


def _fire_marshal_and_codec(n_series: int = 3) -> None:
    for _ in range(n_series):
        t0 = time.perf_counter()
        _burn()
        stage_clock.tick("marshal", time.perf_counter() - t0)
        t0 = time.perf_counter()
        _burn()
        stage_clock.tick("codec", time.perf_counter() - t0)


class _StageClockObjective:
    def __init__(self, inner):
        self.inner = inner

    @property
    def backend(self):
        return self.inner.backend

    @property
    def calls(self):
        return self.inner.calls

    def __call__(self, params):
        with stage_clock.stage("compress"):
            _fire_marshal_and_codec(n_series=3)
        with stage_clock.stage("inference"):
            _burn()
        with stage_clock.stage("metric"):
            _burn()
        return self.inner(params)


@pytest.fixture(autouse=True)
def _stage_clock_isolated():
    stage_clock.drain()
    stage_clock.enable()
    yield
    stage_clock.disable()
    stage_clock.drain()


def test_stage_parents_mark_marshal_and_codec_as_children_of_compress():
    assert stage_clock.STAGE_PARENTS == {"marshal": "compress", "codec": "compress"}
    for leaf in ("compress", "inference", "metric"):
        assert leaf not in stage_clock.STAGE_PARENTS


def test_genetic_last_stage_totals_sums_across_candidates_and_children_le_parent():
    inner = _Objective()
    objective = _StageClockObjective(inner)
    optimizer = GeneticOptimizer(
        pop_size=10, initial_pop_size=3, gens=2, elitism=1, tournament_k=2,
        early_stop_patience=999, n_workers=1, verbose=0, random_state=1,
    )
    best = optimizer.maximize(objective, search_space=BOUNDS, space_definition=SPACE,
                              log_dir=None, run_metadata=None)
    assert best is not None
    assert inner.calls == optimizer.total_budget

    totals = optimizer.last_stage_totals
    for seam in ("compress", "marshal", "codec", "inference", "metric"):
        assert seam in totals, f"{seam!r} missing from last_stage_totals: {totals}"

    n = optimizer.total_budget
    assert totals["compress"][1] == n
    assert totals["inference"][1] == n
    assert totals["metric"][1] == n
    assert totals["marshal"][1] == 3 * n
    assert totals["codec"][1] == 3 * n

    children_sum = totals["marshal"][0] + totals["codec"][0]
    assert children_sum <= totals["compress"][0] + 1e-6

    assert stage_clock.drain() == {}


def test_fold_finalize_drain_is_independent_of_search_attribution():
    inner = _Objective()
    objective = _StageClockObjective(inner)
    optimizer = GeneticOptimizer(
        pop_size=10, initial_pop_size=3, gens=1, elitism=1, tournament_k=2,
        early_stop_patience=999, n_workers=1, verbose=0, random_state=2,
    )
    optimizer.maximize(objective, search_space=BOUNDS, space_definition=SPACE,
                       log_dir=None, run_metadata=None)
    search_totals = dict(optimizer.last_stage_totals)

    assert stage_clock.drain() == {}
    with stage_clock.stage("compress"):
        _fire_marshal_and_codec(n_series=1)
    with stage_clock.stage("inference"):
        _burn()
    fold_finalize_totals = stage_clock.drain()

    assert fold_finalize_totals["compress"][1] == 1
    assert fold_finalize_totals["inference"][1] == 1
    assert fold_finalize_totals["marshal"][1] == 1
    assert "metric" not in fold_finalize_totals

    n = optimizer.total_budget
    assert search_totals["compress"][1] == n
    assert search_totals["inference"][1] == n
    assert fold_finalize_totals["compress"][1] != search_totals["compress"][1]


def test_adaedge_last_stage_totals_and_objective_wall_closes():
    inner = _AdaEdgeObjective()
    objective = _StageClockObjective(inner)
    optimizer = AdaEdgeOptimizer(
        init_points=6, n_iter=6, beta=0.75, num_restarts=2, raw_samples=16,
        verbose=0, alpha=0.75, random_state=3, device="cpu",
        log_dir=None, log_process=False,
    )
    optimizer.maximize(objective, search_space=ADAEDGE_BOUNDS, space_definition=ADAEDGE_SPACE,
                       log_dir=None, run_metadata=None)

    totals = optimizer.last_stage_totals
    n = optimizer.total_budget
    assert inner.calls == n
    assert totals["compress"][1] == n
    assert totals["inference"][1] == n
    assert totals["metric"][1] == n

    top_level_sum = totals["compress"][0] + totals["inference"][0] + totals["metric"][0]
    assert top_level_sum <= optimizer.last_total_objective_wall_sec + 1e-6
    residual = ((optimizer.last_total_objective_wall_sec - top_level_sum)
               / optimizer.last_total_objective_wall_sec)
    assert residual < 0.5, (
        f"stage_clock breakdown accounts for too little of the objective's "
        f"own measured wall time: residual={residual:.2%}"
    )
