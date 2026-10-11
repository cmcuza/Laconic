"""Split the labels of every corpus series into a validation and a test range.

The cut is placed between two events. Around the cut a dead zone of half-width r is
left out of both ranges, with r = ceil(w / 2) and w the window TSB-AD's
find_length_rank estimates on the series. An event that crosses an edge of the zone
is moved into the zone. Of all cuts with at least MIN_EVENTS_PER_SIDE events on each
side, the one with the most even number of events is chosen; ties go to the cut
closest to the middle of the series.

    python scripts/anomaly/build_split.py --root data/anomaly/TSB-AD-U \
        --file-list cfg/datasets/anomaly/file_lists/multievent32.csv \
        --out cfg/datasets/anomaly/splits/multievent32_split.csv
"""
from __future__ import annotations
import argparse
import os
from typing import Optional, Tuple
import numpy as np
import pandas as pd

MIN_EVENTS_PER_SIDE = 3
# Window used when find_length_rank fails, as in TSB-AD.
DEFAULT_WINDOW = 125


def events(labels: np.ndarray):
    """(starts, ends) of the maximal runs of label 1, ends exclusive."""
    d = np.diff(np.concatenate([[0], labels, [0]]))
    return np.where(d == 1)[0], np.where(d == -1)[0]


def window(x: np.ndarray) -> int:
    from TSB_AD.utils.slidingWindows import find_length_rank
    try:
        return max(int(find_length_rank(x.reshape(-1, 1), rank=1)), 1)
    except Exception:                             # noqa: BLE001 - fall back to the default
        return DEFAULT_WINDOW


def mask_split(starts: np.ndarray, ends: np.ndarray, r: int, n: int) -> Optional[Tuple[int, int]]:
    """(val_end, test_start) of the best cut, or None if no cut is acceptable.

    Args:
      starts: Start index of every event.
      ends: End index of every event, exclusive.
      r: Half-width of the dead zone around the cut.
      n: Length of the series.
    """
    if len(starts) < 2 * MIN_EVENTS_PER_SIDE:
        return None
    best = None
    for k in range(1, len(starts)):
        # The cut is the midpoint of the gap between event k-1 and event k.
        c = int((ends[k - 1] + starts[k]) // 2)
        lo, hi = c - r, c + r

        # Widen the zone to the start or end of an event that crosses its edge.
        crossing = (starts < lo) & (ends > lo)
        if crossing.any():
            lo = int(starts[crossing].min())
        crossing = (starts < hi) & (ends > hi)
        if crossing.any():
            hi = int(ends[crossing].max())

        if not 0 < lo < hi < n:
            continue
        n_val = int((ends <= lo).sum())
        n_test = int((starts >= hi).sum())
        if n_val < MIN_EVENTS_PER_SIDE or n_test < MIN_EVENTS_PER_SIDE:
            continue

        key = (abs(n_val / (n_val + n_test) - 0.5), abs((lo + hi) / 2 / n - 0.5))
        if best is None or key < best[0]:
            best = (key, lo, hi)
    return None if best is None else (best[1], best[2])


def split_row(root: str, name: str) -> dict:
    df = pd.read_csv(os.path.join(root, name)).dropna()
    x = df.iloc[:, 0].to_numpy(dtype=float)
    labels = (df["Label"].to_numpy() > 0).astype(int)
    n = len(labels)
    starts, ends = events(labels)
    w = window(x)
    r = max(1, -(-w // 2))

    cut = mask_split(starts, ends, r, n)
    if cut is None:
        raise ValueError(f"{name}: no cut with {MIN_EVENTS_PER_SIDE} events on each side")
    val_end, test_start = cut
    events_val = int((ends <= val_end).sum())
    events_test = int((starts >= test_start).sum())
    return {"series": name, "n_points": n, "window": w, "half_span": r,
            "n_events": len(starts), "val_end": val_end, "test_start": test_start,
            "events_val": events_val, "events_test": events_test,
            "events_masked": len(starts) - events_val - events_test}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="data/anomaly/TSB-AD-U")
    ap.add_argument("--file-list", default="cfg/datasets/anomaly/file_lists/multievent32.csv")
    ap.add_argument("--out", default="cfg/datasets/anomaly/splits/multievent32_split.csv")
    args = ap.parse_args()

    names = pd.read_csv(args.file_list)["file_name"].astype(str).str.strip()
    out = pd.DataFrame([split_row(args.root, name) for name in names])
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    out.to_csv(args.out, index=False)
    print(f"{len(out)} series split -> {args.out}")


if __name__ == "__main__":
    main()
