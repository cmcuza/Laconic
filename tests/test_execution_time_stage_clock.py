"""Dataset-free test for the stage_clock accumulation wiring
`scripts/benchmark_execution_time.py`'s Phase 3 harness builds on
(§5.2/§7.2 of `docs/EXECUTION_TIME_STUDY_PLAN.md`).

No dataset, no model, no compression codec: this reuses the same
deterministic synthetic objectives `scripts/verify_optimizer_budgets.py`
already built for budget-exactness checking (imported from there, never
re-expressed - the same reuse `scripts/benchmark_optimizer_overhead.py`
makes for Phase 2), wrapped to additionally fire the same stage_clock seam
names `experiments/objectives.py`'s `_compute_components` methods fire around
a real compress/predict/score call
(`compress`/`marshal`/`codec`/`inference`/`metric`). That exercises
`optimizer/genetic.py`'s and `optimizer/adaedge.py`'s `last_stage_totals`
accumulation - the exact mechanism
`scripts/benchmark_execution_time.py::emit_stage_breakdown` reads from -
against real per-candidate `TimingRecord.stages` dicts, without needing any
real data.

Runs in well under a second: `n_workers=1` for genetic needs no forkserver
pool (the serial branch of `_evaluate_batch` uses the identical `timed_call`
helper the pool path does - see `optimizer/genetic.py`'s module docstring
note on this), so no `LACONIC_STAGE_CLOCK` environment variable is needed
either - the seams fire in the same interpreter that already has
`stage_clock` enabled via the fixture below.
"""
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
    """Trivial CPU work so `perf_counter` measures a real, nonzero duration -
    stands in for a seam's actual work (compression, model inference) without
    needing any."""
    total = 0
    for i in range(n):
        total += i * i
    return total


def _fire_marshal_and_codec(n_series: int = 3) -> None:
    """Mirrors `compression/utils.py::compress_and_decompress_batch_cr`'s
    per-series loop: `marshal`/`codec` fire `n_series` times each, both
    children of the enclosing `compress` stage
    (`profiling/stage_clock.py::STAGE_PARENTS`)."""
    for _ in range(n_series):
        t0 = time.perf_counter()
        _burn()
        stage_clock.tick("marshal", time.perf_counter() - t0)
        t0 = time.perf_counter()
        _burn()
        stage_clock.tick("codec", time.perf_counter() - t0)


class _StageClockObjective:
    """Wraps a Phase-2-style synthetic objective
    (`scripts/verify_optimizer_budgets.py`), additionally firing the same
    stage_clock seams `experiments/objectives.py`'s `_compute_components`
    methods fire around one real compress/predict/score call, so
    `optimizer/genetic.py`'s and `optimizer/adaedge.py`'s
    `last_stage_totals` accumulation gets real (synthetic-timed, but
    genuinely wall-clock-measured) data to sum."""

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
    """`profiling.stage_clock` is process-global module state - isolate this
    test's accumulation from anything else that ran in this pytest process
    (or leaked into it), both before and after."""
    stage_clock.drain()
    stage_clock.enable()
    yield
    stage_clock.disable()
    stage_clock.drain()


def test_stage_parents_mark_marshal_and_codec_as_children_of_compress():
    """The `parent` column `scripts/benchmark_execution_time.py::emit_stage_breakdown`
    writes comes straight from this mapping - if it drifts, every downstream
    "children <= parent, don't sum as siblings" check (verification 5) drifts
    with it."""
    assert stage_clock.STAGE_PARENTS == {"marshal": "compress", "codec": "compress"}
    for leaf in ("compress", "inference", "metric"):
        assert leaf not in stage_clock.STAGE_PARENTS


def test_genetic_last_stage_totals_sums_across_candidates_and_children_le_parent():
    """§7.2: "the drained stage_clock totals... attributed to `search`... by
    summing the returned dicts" - `optimizer/genetic.py::last_stage_totals`
    is that sum, built regardless of `n_workers`/`log_dir`. `n_workers=1`
    keeps this dataset-free and fast (no forkserver pool)."""
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

    # Per-candidate dicts from several candidates sum correctly: exactly one
    # compress/inference/metric firing per evaluation, three marshal/codec
    # firings per evaluation (n_series=3 in _fire_marshal_and_codec above).
    n = optimizer.total_budget
    assert totals["compress"][1] == n
    assert totals["inference"][1] == n
    assert totals["metric"][1] == n
    assert totals["marshal"][1] == 3 * n
    assert totals["codec"][1] == 3 * n

    # Sum of children stays at or below the parent - marshal/codec fire
    # INSIDE compress's own `with` block, so compress's measured total can
    # only be >= what its children alone measured (the `with` block also
    # covers whatever falls between/around the tick() calls).
    children_sum = totals["marshal"][0] + totals["codec"][0]
    assert children_sum <= totals["compress"][0] + 1e-6

    # A drain after maximize() returns is empty: the serial branch of
    # _evaluate_batch calls timed_call() per candidate (optimizer/genetic.py),
    # which already drains everything down to {} after each one - nothing is
    # left over to leak into a subsequently-timed fold_finalize block.
    assert stage_clock.drain() == {}


def test_fold_finalize_drain_is_independent_of_search_attribution():
    """`scripts/benchmark_execution_time.py` drains `stage_clock` a SECOND
    time around its own `fold_finalize` block, attributing that separately
    from `search`'s `last_stage_totals` (§5.2/§7.2) - simulated here with the
    same primitives the harness calls, without the harness itself."""
    inner = _Objective()
    objective = _StageClockObjective(inner)
    optimizer = GeneticOptimizer(
        pop_size=10, initial_pop_size=3, gens=1, elitism=1, tournament_k=2,
        early_stop_patience=999, n_workers=1, verbose=0, random_state=2,
    )
    optimizer.maximize(objective, search_space=BOUNDS, space_definition=SPACE,
                       log_dir=None, run_metadata=None)
    search_totals = dict(optimizer.last_stage_totals)  # copy - more draining below

    # Mirrors benchmark_execution_time.py's fold_finalize block: drain first
    # (clearing anything left over - already {} per the previous test), fire
    # a much smaller number of new seams (one compress + one inference call,
    # matching e.g. classification/clustering's val-only fold_finalize
    # shape), then drain again.
    assert stage_clock.drain() == {}
    with stage_clock.stage("compress"):
        _fire_marshal_and_codec(n_series=1)
    with stage_clock.stage("inference"):
        _burn()
    fold_finalize_totals = stage_clock.drain()

    # The two attributions never mix: fold_finalize's counts are exactly what
    # fold_finalize fired here (not search's much larger counts), and
    # search's own totals (captured into a plain dict before the extra
    # draining above) are untouched.
    assert fold_finalize_totals["compress"][1] == 1
    assert fold_finalize_totals["inference"][1] == 1
    assert fold_finalize_totals["marshal"][1] == 1
    assert "metric" not in fold_finalize_totals  # this fold_finalize shape fired no metric seam

    n = optimizer.total_budget
    assert search_totals["compress"][1] == n
    assert search_totals["inference"][1] == n
    assert fold_finalize_totals["compress"][1] != search_totals["compress"][1]


def test_adaedge_last_stage_totals_and_objective_wall_closes():
    """adaedge has no worker pool - `last_stage_totals` accumulates
    in-process, per evaluation, via the same `timed_call()` helper genetic's
    workers use. Also exercises `last_total_objective_wall_sec` (added
    alongside `last_stage_totals` for verification 5:
    "Sigma(marshal+codec+inference+metric) matches measured objective time
    within ~5% for the sequential methods") directly against the seam sum -
    the dataset-free equivalent of the real-cell check already run once
    against a real Coffee/adaedge trace (see docs/EXECUTION_TIME_STUDY_PLAN.md
    §13)."""
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
    # The seams are a SUBSET of each evaluation's own wall time (there is a
    # little more: params_from_vector/combine_fitness-equivalent overhead
    # around them, deliberately not seamed - see profiling/stage_clock.py's
    # module docstring), so the sum can never exceed the measured total.
    assert top_level_sum <= optimizer.last_total_objective_wall_sec + 1e-6
    residual = ((optimizer.last_total_objective_wall_sec - top_level_sum)
               / optimizer.last_total_objective_wall_sec)
    # Verification 5's real tolerance is ~5% on real (multi-millisecond)
    # evaluations; this synthetic burn() is orders of magnitude cheaper, so
    # relative jitter is naturally larger - this checks the residual stays a
    # small fraction of the total rather than pinning an exact percentage.
    assert residual < 0.5, (
        f"stage_clock breakdown accounts for too little of the objective's "
        f"own measured wall time: residual={residual:.2%}"
    )
