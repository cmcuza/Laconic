"""Where a run's results CSV lands, and whether one is already complete.

The path formula is duplicated verbatim at the tail of all four
``experiments/*_runner.py`` files. This module states it once so a caller that
wants to know "has this been run already?" cannot drift from the writer - the
same reason ``analysis/figure_paths.py`` owns figure discovery and
``analysis/method_registry.py`` owns method identity.

Wiring the runners themselves onto this is deliberately NOT done yet: a
clustering sweep is in flight as of 2026-09-03 and each of its invocations
re-imports the runners from disk, so editing them mid-sweep would run different
code for different cells. Do that once the sweep finishes.

Completeness, not existence, is the question worth asking. A runner writes its
CSV only after every fold succeeds (the ``to_csv`` sits inside the ``try``,
before the ``finally``), and the fold-0 mean row is concatenated last - so a
fold-0 row proves the run reached the end. ``random_state`` is the one identity
field NOT encoded in the path: every seed's rows share one file, so a check that
ignores it would call a file "done" on the strength of a different seed's run.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

import pandas as pd


def result_csv_path(
    results_root: str,
    task: str,
    compressor: str,
    optimizer: str,
    model_name: str,
    budget: int,
    alpha: float,
    dataset: str,
) -> Path:
    """The CSV a run with this identity writes.

    Mirrors the runners exactly, including that only the *filename* is
    lowercased (``model_name`` keeps its capitalisation: ``.../Rocket/...``).
    """
    return (Path(results_root) / task / compressor / optimizer / model_name
            / f"budget_{int(budget)}" / f"alpha_{float(alpha):g}"
            / f"{dataset}.csv".lower())


def is_complete(path: os.PathLike | str, random_state: int) -> bool:
    """True if ``path`` holds a finished run for ``random_state``.

    False for a missing, empty, unreadable or seed-mismatched file - never
    raises, because the caller's next move on any of those is the same: run it.
    """
    path = Path(path)
    if not path.is_file() or path.stat().st_size == 0:
        return False
    try:
        frame = pd.read_csv(path, usecols=["fold", "random_state"])
    except (ValueError, pd.errors.ParserError, pd.errors.EmptyDataError, OSError):
        return False
    done = frame[(frame["fold"] == 0) & (frame["random_state"] == int(random_state))]
    return not done.empty


def missing_datasets(
    datasets: list[str],
    results_root: str,
    task: str,
    compressor: str,
    optimizer: str,
    model_name: str,
    budget: int,
    alpha: float,
    random_state: int,
) -> list[str]:
    """Subset of ``datasets`` with no finished run for this identity."""
    return [
        ds for ds in datasets
        if not is_complete(
            result_csv_path(results_root, task, compressor, optimizer,
                            model_name, budget, alpha, ds),
            random_state,
        )
    ]
