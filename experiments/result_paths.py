"""Where a run's results CSV lands, and whether one is already complete."""
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
    """The CSV a run with this identity writes."""
    return (Path(results_root) / task / compressor / optimizer / model_name
            / f"budget_{int(budget)}" / f"alpha_{float(alpha):g}"
            / f"{dataset}.csv".lower())


def is_complete(path: os.PathLike | str, random_state: int) -> bool:
    """True if ``path`` holds a finished run for ``random_state``."""
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
