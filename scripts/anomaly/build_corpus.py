"""Select the multievent32 corpus from TSB-AD-U and write its file list.

A series is admitted if it has at least `--min-events` anomaly events and at least
`--min-points` points. An event is a maximal run of points labelled 1.

    python scripts/anomaly/build_corpus.py --root data/anomaly/TSB-AD-U \
        --out cfg/datasets/anomaly/file_lists/multievent32.csv
"""
from __future__ import annotations
import argparse
import glob
import os
import numpy as np
import pandas as pd


def events(labels: np.ndarray):
    """(starts, ends) of the maximal runs of label 1, ends exclusive."""
    d = np.diff(np.concatenate([[0], labels, [0]]))
    return np.where(d == 1)[0], np.where(d == -1)[0]


def census(root: str) -> pd.DataFrame:
    """Length and number of events of every series under `root`."""
    rows = []
    for path in sorted(glob.glob(os.path.join(root, "*.csv"))):
        name = os.path.basename(path)
        # Rows with missing values are dropped, as the loader does.
        labels = (pd.read_csv(path).dropna()["Label"].to_numpy() > 0).astype(int)
        starts, _ = events(labels)
        rows.append({"series": name, "source": name.split("_")[1],
                     "n_points": len(labels), "n_events": len(starts)})
    return pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="data/anomaly/TSB-AD-U")
    ap.add_argument("--out", default="cfg/datasets/anomaly/file_lists/multievent32.csv")
    ap.add_argument("--min-events", type=int, default=50)
    ap.add_argument("--min-points", type=int, default=20_000)
    args = ap.parse_args()

    df = census(args.root)
    keep = (df.n_events >= args.min_events) & (df.n_points >= args.min_points)
    # Ordered by source, then by number of events, largest first.
    sel = df[keep].sort_values(["source", "n_events"], ascending=[True, False], kind="mergesort")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    pd.DataFrame({"file_name": sel.series}).to_csv(args.out, index=False)
    print(f"{len(df)} series in {args.root}, {len(sel)} admitted -> {args.out}")
    print(f"  sources: {sel.source.value_counts().sort_index().to_dict()}")
    print(f"  points:  {int(sel.n_points.sum()):,}")


if __name__ == "__main__":
    main()
