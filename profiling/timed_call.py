"""Per-candidate wall/CPU/stage timing for one objective call."""
from __future__ import annotations

import time
from typing import Any, Callable, Dict, Tuple

from profiling import stage_clock

TimingRecord = Dict[str, Any]


SEAMS = ("compress", "marshal", "codec", "inference", "metric")
SEAM_FIELDS = [f"{name}_sec" for name in SEAMS]


def seam_extra(record: TimingRecord) -> Dict[str, Any]:
    """`{'<seam>_sec': seconds}` from one TimingRecord's drained stages."""
    stages = record.get("stages") or {}
    if not stages:
        return {}
    return {f"{name}_sec": (stages[name][0] if name in stages else None)
            for name in SEAMS}


def timed_call(objective: Callable[..., float], candidate: Any, *args: Any, **kwargs: Any) -> Tuple[float, TimingRecord]:
    """Call `objective(candidate, *args, **kwargs)`, returning `(score, TimingRecord)`."""
    wall_start = time.perf_counter()
    cpu_start = time.process_time()
    score = objective(candidate, *args, **kwargs)
    wall_sec = time.perf_counter() - wall_start
    cpu_sec = time.process_time() - cpu_start
    stages = stage_clock.drain()
    record: TimingRecord = {"wall_sec": wall_sec, "cpu_sec": cpu_sec, "stages": stages}
    return score, record
