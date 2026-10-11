"""Tables behind the anomaly detection figures and numbers of the paper.

Reads the result CSV of each method for one detector and one alpha, and writes:

    anomaly_perseries_<dataset>_<detector>_alpha<a>.csv  one row per (method, series)
    anomaly_tradeoff_<dataset>_<detector>_alpha<a>.csv   one row per method (trade-off figure)
    anomaly_loss_<dataset>_<detector>_alpha<a>.csv       one row per method (quality loss)

With --exemplar it also writes the curves of the example figure:

    anomaly_example_<series>_<lo>_<hi>.csv

All numbers are on the test range. Run from the repository root:

    python analysis/anomaly_paper_tables.py --alpha 0.95 \
        --exemplar results/anomaly/exemplar/266_IOPS_id_7_WebService_tr_4010_1st_4110.npz
"""
from __future__ import annotations
import argparse
import os
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import numpy as np
import pandas as pd
from scipy.stats import rankdata

from analysis.anomaly_figure_style import label

# (compressor, optimizer), in legend order.
METHODS = [("sz", "bosmp"), ("mixpiece", "bosmp"), ("serfxor", "bosmp"),
           ("adaedge", "adaedge"), ("laconic", "genetic")]
# Name of each compressor's arrays in the exemplar file.
ARM = {"sz": "SZ", "mixpiece": "MixPiece", "serfxor": "SerfXOR", "adaedge": "AdaEdge",
       "laconic": "Laconic"}

# Window of the example figure and the event it shows.
EXAMPLE_WINDOW = (74700, 74785)
EXAMPLE_EVENT = (74707, 74733)


def events(labels: np.ndarray):
    """(starts, ends) of the maximal runs of label 1, ends exclusive."""
    d = np.diff(np.concatenate([[0], labels.astype(int), [0]]))
    return np.where(d == 1)[0], np.where(d == -1)[0]


def percentile(score: np.ndarray, test_start: int) -> np.ndarray:
    """Percentile (0-100, ties averaged) of every test-range score among the test-range scores."""
    t = np.asarray(score[test_start:], float)
    return 100.0 * (rankdata(t, method="average") - 1) / max(len(t) - 1, 1)


def gmean(s: pd.Series) -> float:
    return float(np.exp(np.log(s).mean()))


def per_series(results_root, dataset, detector, alpha, budget) -> pd.DataFrame:
    rows = []
    for comp, opt in METHODS:
        path = os.path.join(results_root, comp, opt, detector, f"budget_{budget}",
                            f"alpha_{alpha}", f"{dataset}.csv")
        d = pd.read_csv(path)
        d = d[d.fold >= 1]
        if not (d.status == "ok").all():
            raise ValueError(f"{path}: rows with status other than ok")
        rows.append(pd.DataFrame({
            "method": label(comp), "compressor": comp,
            "series": d.series.str.replace(".csv", "", regex=False),
            "q_pre": d.q_pre_test, "q_post": d.q_post_test,
            "dq": d.q_pre_test - d.q_post_test,
            "cr": d.test_avg_cr, "zstd_cr": d.baseline_cr, "best_params": d.best_params,
        }))
    return pd.concat(rows, ignore_index=True)


def tradeoff(P: pd.DataFrame) -> pd.DataFrame:
    """Mean quality and geometric-mean compression ratio per method."""
    return P.groupby("method", sort=False).agg(
        q_post=("q_post", "mean"), q_pre=("q_pre", "mean"),
        cr=("cr", gmean), zstd_cr=("zstd_cr", gmean)).reset_index()


def loss(P: pd.DataFrame) -> pd.DataFrame:
    """Quality lost on the test range, dq = q_pre - q_post, per method."""
    g = P.groupby("method", sort=False).dq
    return pd.DataFrame({
        "mean": g.mean(), "median": g.median(), "worst": g.max(),
        "n_degraded": g.apply(lambda s: int((s > 0).sum())),
        "worst_series": P.loc[g.idxmax(), "series"].values,
    }).reset_index()


def example(P: pd.DataFrame, npz_path: str, window, event):
    """Raw series, reconstructions and raw-series score percentile inside `window`."""
    z = np.load(npz_path, allow_pickle=True)
    test_start = int(z["test_start"])
    starts, ends = events(z["y"])
    keep = starts >= test_start
    if (int(event[0]), int(event[1])) not in set(zip(starts[keep].tolist(), ends[keep].tolist())):
        raise ValueError(f"{event} is not an event of the test range")

    lo, hi = window
    df = pd.DataFrame({"t": np.arange(lo, hi), "label": z["y"][lo:hi].astype(int),
                       "raw": z["x"][lo:hi]})
    for comp, _opt in METHODS:
        df[label(comp)] = z[f"x_rec_{ARM[comp]}"][lo:hi]
    df["score_raw_pct"] = percentile(z["score_pre"], test_start)[lo - test_start:hi - test_start]

    series = os.path.basename(npz_path).replace(".npz", "")
    S = P[P.series == series].set_index("method")
    notes = [f"#   uncompressed VUS-PR {S.q_pre.iloc[0]:.3f}"]
    notes += [f"#   {m:8s} VUS-PR {r.q_post:.3f}  CR {r.cr:.1f}  {r.compressor} {r.best_params}"
              for m, r in S.iterrows()]
    return df, series, notes


def write(path: str, df: pd.DataFrame, header) -> None:
    with open(path, "w") as fh:
        fh.write("\n".join(header) + "\n")
        df.to_csv(fh, index=False, float_format="%.7g")
    print(f"-> {path}  ({len(df)} rows)")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-root", default="results/anomaly")
    ap.add_argument("--dataset", default="multievent32")
    ap.add_argument("--detector", default="Sub_PCA")
    ap.add_argument("--alpha", default="0.75")
    ap.add_argument("--budget", type=int, default=100)
    ap.add_argument("--exemplar", default=None, help="npz written by scripts/anomaly/replay_exemplar.py")
    ap.add_argument("--window", type=int, nargs=2, default=EXAMPLE_WINDOW)
    ap.add_argument("--event", type=int, nargs=2, default=EXAMPLE_EVENT)
    ap.add_argument("--out", default="results/figures/anomaly_detection")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    P = per_series(args.results_root, args.dataset, args.detector, args.alpha, args.budget)
    stem = f"{args.dataset}_{args.detector}_alpha{args.alpha}"
    source = [f"# detector {args.detector}, alpha {args.alpha}, budget {args.budget}, "
              f"{P.series.nunique()} series of {args.dataset}, test range",
              f"# read from {args.results_root}/<compressor>/<optimizer>/{args.detector}/"
              f"budget_{args.budget}/alpha_{args.alpha}/{args.dataset}.csv, rows with fold >= 1"]

    write(os.path.join(args.out, f"anomaly_perseries_{stem}.csv"), P, source + [
        "# q_pre: Q of the detector on the uncompressed series; q_post: Q on the reconstruction",
        "# dq = q_pre - q_post; cr: compression ratio; zstd_cr: compression ratio of ZSTD"])
    write(os.path.join(args.out, f"anomaly_tradeoff_{stem}.csv"), tradeoff(P), source + [
        "# q_post, q_pre: arithmetic mean over the series; cr, zstd_cr: geometric mean"])
    write(os.path.join(args.out, f"anomaly_loss_{stem}.csv"), loss(P), source + [
        "# statistics of dq = q_pre - q_post over the series; n_degraded counts dq > 0"])

    if args.exemplar:
        E, series, notes = example(P, args.exemplar, args.window, args.event)
        lo, hi = args.window
        write(os.path.join(args.out, f"anomaly_example_{series}_{lo}_{hi}.csv"), E, source + [
            f"# series {series}, time indices [{lo}, {hi}), event [{args.event[0]}, {args.event[1]})",
            "# raw: uncompressed series; one column per method: its reconstruction",
            "# score_raw_pct: detector score on the uncompressed series, as a percentile of",
            "#   its scores on the test range",
            "# whole test range, per method:"] + notes)


if __name__ == "__main__":
    main()
