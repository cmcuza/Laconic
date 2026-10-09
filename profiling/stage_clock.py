"""Off-by-default, process-local stage-time accumulator."""
from __future__ import annotations

import os
import time
from contextlib import contextmanager
from typing import Dict, Iterator, Tuple

_ENABLED = os.environ.get("LACONIC_STAGE_CLOCK", "0") == "1"
_TOTALS: Dict[str, float] = {}
_COUNTS: Dict[str, int] = {}

STAGE_PARENTS: Dict[str, str] = {
    "marshal": "compress",
    "codec": "compress",
}


def enable() -> None:
    """Turn stage timing on in this process."""
    global _ENABLED
    _ENABLED = True


def disable() -> None:
    global _ENABLED
    _ENABLED = False


def enabled() -> bool:
    return _ENABLED


@contextmanager
def stage(name: str) -> Iterator[None]:
    """Time a once-per-evaluation seam."""
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
    """Record one manually-timed call for a per-series seam (`marshal`/`codec`)."""
    if not _ENABLED:
        return
    _TOTALS[name] = _TOTALS.get(name, 0.0) + elapsed_sec
    _COUNTS[name] = _COUNTS.get(name, 0) + 1


def drain() -> Dict[str, Tuple[float, int]]:
    """Return and reset `{stage: (total_seconds, count)}`."""
    global _TOTALS, _COUNTS
    out = {name: (_TOTALS[name], _COUNTS[name]) for name in _TOTALS}
    _TOTALS = {}
    _COUNTS = {}
    return out
