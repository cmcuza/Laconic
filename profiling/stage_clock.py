"""Off-by-default, process-local stage-time accumulator.

See `docs/EXECUTION_TIME_STUDY_PLAN.md` §5.1. This exists to answer "where
inside one evaluation does the wall clock go", without perturbing the shipping
hot path when nobody is asking.

Design constraints (from the plan, restated because they explain the two
different calling conventions below):

- **Off by default**, so the shipping hot path is effectively byte-identical.
  Enabled by an explicit `enable()` call, or by the environment variable
  `LACONIC_STAGE_CLOCK=1`, read once at import. The environment variable is
  what reaches forkserver children (`optimizer/genetic.py`): they inherit the
  parent's environment but not its module state, so `enable()` called in the
  parent has no effect on already-spawned/future workers - set the env var
  instead if workers need to be profiled.
- **Process-local.** State lives in whatever process runs the objective - in a
  genetic worker that is the child (see CLAUDE.md's forkserver note and
  `docs/EXECUTION_TIME_STUDY_PLAN.md` §2.4).
- **Call counts are kept alongside totals**, so a per-evaluation mean can be
  formed without the caller separately tracking the evaluation count.
- **Two calling conventions, not one.** `stage()` is a context manager for the
  three once-per-evaluation seams in `experiments/objectives.py`. The two
  seams inside `compression/utils.py`'s per-series loop (`marshal`, `codec`)
  fire `2 x n_series x D` times per evaluation - a `@contextmanager` costs
  roughly 1us per entry even when its body does nothing, which is a real cost
  at that call count (~9,800 for LiveFuelMoistureContent). Those two use
  `tick()` instead: the caller takes its own `perf_counter()` pair with a bare
  `if stage_clock.enabled():` guard and hands the elapsed time here.
"""
from __future__ import annotations

import os
import time
from contextlib import contextmanager
from typing import Dict, Iterator, Tuple

# Read once at import - see module docstring. This is the hot-path flag:
# `compression/utils.py`'s per-series seams test it directly
# (`stage_clock._ENABLED`) to avoid even a function-call's overhead at
# `2 x n_series x D` calls/evaluation; `enabled()` below is the same value for
# callers where a function call is negligible (i.e. everywhere else).
_ENABLED = os.environ.get("LACONIC_STAGE_CLOCK", "0") == "1"
_TOTALS: Dict[str, float] = {}
_COUNTS: Dict[str, int] = {}

# Parent/child stage relationships, for any consumer that aggregates over
# these names: `marshal` and `codec` fire *inside* `compress`'s per-series
# loop (`compression/utils.py::compress_and_decompress_batch_cr`), so they are
# children of it, not siblings - summing all of `stages` as flat totals would
# double-count `compress`'s own wall time. Encoded here once so no consumer
# (scripts/benchmark_execution_time.py, scripts/execution_time_report.py,
# tests) has to hard-code the relationship a second time.
STAGE_PARENTS: Dict[str, str] = {
    "marshal": "compress",
    "codec": "compress",
}


def enable() -> None:
    """Turn stage timing on in this process. Does not reach already-running
    or future forkserver children - set LACONIC_STAGE_CLOCK=1 in the
    environment for that (see module docstring)."""
    global _ENABLED
    _ENABLED = True


def disable() -> None:
    global _ENABLED
    _ENABLED = False


def enabled() -> bool:
    return _ENABLED


@contextmanager
def stage(name: str) -> Iterator[None]:
    """Time a once-per-evaluation seam. A single bool test and nothing else
    when disabled - no `perf_counter()` call, no dict work."""
    if not _ENABLED:
        yield
        return
    start = time.perf_counter()
    try:
        yield
    finally:
        elapsed = time.perf_counter() - start
        _TOTALS[name] = _TOTALS.get(name, 0.0) + elapsed
        _COUNTS[name] = _COUNTS.get(name, 0) + 1


def tick(name: str, elapsed_sec: float) -> None:
    """Record one manually-timed call for a per-series seam (`marshal`/`codec`).

    Callers own their own `perf_counter()` pair, guarded by `enabled()`, so
    that pair is skipped entirely when the clock is off - see module
    docstring for why this isn't a context manager."""
    if not _ENABLED:
        return
    _TOTALS[name] = _TOTALS.get(name, 0.0) + elapsed_sec
    _COUNTS[name] = _COUNTS.get(name, 0) + 1


def drain() -> Dict[str, Tuple[float, int]]:
    """Return `{stage_name: (total_seconds, count)}` for everything recorded
    since the last drain, and reset the accumulators.

    Called once per objective evaluation by `profiling.timed_call.timed_call`,
    so what comes back is that one evaluation's decomposition - never a
    cross-evaluation running total. Empty when the clock is disabled, since
    nothing was ever recorded."""
    global _TOTALS, _COUNTS
    out = {name: (_TOTALS[name], _COUNTS[name]) for name in _TOTALS}
    _TOTALS = {}
    _COUNTS = {}
    return out
