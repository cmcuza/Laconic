"""Per-candidate wall/CPU/stage timing for one objective call.

See `docs/EXECUTION_TIME_STUDY_PLAN.md` §5.4. Before this, `genetic`'s worker
(`optimizer/genetic.py::_evaluate_worker_candidate`) returned a bare float, so
every wall-clock comparison against LACONIC in the plan was an "up to" bound
derived from the batch schedule, not a measurement (§2.4, §8.1). `timed_call`
is the one place that gets fixed: every optimizer that evaluates a candidate
(`genetic`, `adaedge`, `bosmp`) calls this instead of the objective directly,
so every one of them now ships wall time, CPU time and (when the stage clock
is on) the drained per-stage decomposition back with the score - at
`n_workers=1` and `n_workers=8` alike, since the child in a forkserver worker
calls this too.

Scores are untouched: `timed_call` calls `objective(...)` exactly once and
returns exactly what it returned, alongside the timing. See verification 2
(`tests/test_clustering_end_to_end.py`, identical numbers before/after).
"""
from __future__ import annotations

import time
from typing import Any, Callable, Dict, Tuple

from profiling import stage_clock

# A plain dict of floats/ints (plus a nested dict of the same), deliberately -
# not a custom class - so it pickles across a forkserver pool boundary with no
# surprises and serializes into a trace row with no extra step. Keys:
#   wall_sec: float                          - time.perf_counter() around the call
#   cpu_sec: float                           - time.process_time() around the call
#   stages: Dict[str, Tuple[float, int]]     - stage_clock.drain(); {} when the
#                                               clock is disabled
TimingRecord = Dict[str, Any]


# The once-per-evaluation seams worth attributing to a single candidate. Logged
# as per-evaluation trace columns (see seam_extra) so a breakdown can be cut by
# WHICH pipeline was evaluated - for `adaedge` that means per compressor arm
# (mixpiece/serfxor/sz, its `method_index`), and for `genetic` per TerseTS
# primitive triple. Summing them over a whole run, as the harness does, answers
# "how much compression" but not "compression by which compressor".
SEAMS = ("compress", "marshal", "codec", "inference", "metric")
SEAM_FIELDS = [f"{name}_sec" for name in SEAMS]


def seam_extra(record: TimingRecord) -> Dict[str, Any]:
    """`{'<seam>_sec': seconds}` from one TimingRecord's drained stages.

    Values are None for a seam that did not fire. Empty dict when the stage
    clock is off, in which case the optimizers do not declare the columns
    either, so a normal run's trace schema is unchanged.
    """
    stages = record.get("stages") or {}
    if not stages:
        return {}
    return {f"{name}_sec": (stages[name][0] if name in stages else None)
            for name in SEAMS}


def timed_call(objective: Callable[..., float], candidate: Any, *args: Any, **kwargs: Any) -> Tuple[float, TimingRecord]:
    """Call `objective(candidate, *args, **kwargs)`, returning `(score, TimingRecord)`.

    `wall_sec` is wall-clock around the call. `cpu_sec` is
    `time.process_time()` around the call, so it covers the calling process
    and its own threads (e.g. BLAS) - not a forkserver pool's other workers,
    each of which reports its own. `stages` is `stage_clock.drain()` taken
    immediately after the call returns: whatever seams fired *inside* this one
    call, and only this one - `drain()` resets the accumulators, so nothing
    leaks into the next candidate's record.

    Cost: one `perf_counter()` pair and one `process_time()` pair per call,
    regardless of whether the stage clock is enabled - nothing per series.
    """
    wall_start = time.perf_counter()
    cpu_start = time.process_time()
    score = objective(candidate, *args, **kwargs)
    wall_sec = time.perf_counter() - wall_start
    cpu_sec = time.process_time() - cpu_start
    stages = stage_clock.drain()
    record: TimingRecord = {"wall_sec": wall_sec, "cpu_sec": cpu_sec, "stages": stages}
    return score, record
