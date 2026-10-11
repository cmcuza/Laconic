"""Compression ratio against VUS-PR for anomaly detection, one panel per alpha.

Each marker is one method: mean test VUS-PR over the series against the geometric
mean of its compression ratio. The hollow ring is ZSTD, which is lossless and so sits
at the VUS-PR of the uncompressed series. Reads the trade-off tables written by
analysis/anomaly_paper_tables.py. Run from the repository root:

    python analysis/anomaly_tradeoff_figure.py --alphas 0.75 0.95
"""
from __future__ import annotations
import argparse
import os
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.ticker import MultipleLocator

from analysis.anomaly_figure_style import color, label, marker, use_libertine

# Compressor of each method, in legend order.
METHODS = ["sz", "mixpiece", "serfxor", "adaedge", "laconic"]

FIGSIZE = (3.34, 1.3)   # ACM single column, inches
S = {
    "name_fs": 9.0,     # alpha label inside the axes
    "label_fs": 8.0,
    "tick_fs": 7.0,
    "legend_fs": 7.0,
    "marker": 26,       # scatter area, pt^2
    "zstd": 28,
    "legend_ms": 3.6,
}
# Fraction of the marker span left free below and above.
Y_PAD, Y_PAD_TOP = 0.14, 0.16
X_LIM, X_STEP = (-10, 160), 50


def load(tables: str, dataset: str, detector: str, alphas) -> pd.DataFrame:
    rows = []
    for alpha in alphas:
        d = pd.read_csv(os.path.join(tables, f"anomaly_tradeoff_{dataset}_{detector}_alpha{alpha}.csv"),
                        comment="#")
        d["alpha"] = alpha
        rows.append(d)
    return pd.concat(rows, ignore_index=True)


def panel(ax, d: pd.DataFrame, name: str, ylabel) -> None:
    for key in METHODS:
        r = d[d.method == label(key)].iloc[0]
        ax.scatter([r.cr], [r.q_post], s=S["marker"], marker=marker(key), color=color(key),
                   edgecolor="black", linewidth=0.4, zorder=4)
    ax.scatter([d.zstd_cr.iloc[0]], [d.q_pre.iloc[0]], s=S["zstd"], marker="o",
               facecolors="none", edgecolors="0.25", linewidth=0.9, zorder=3)

    ax.text(0.035, 0.96, name, transform=ax.transAxes, ha="left", va="top",
            fontsize=S["name_fs"], zorder=5,
            bbox=dict(facecolor="white", edgecolor="none", pad=0.8, alpha=0.85))

    ax.set_xlabel("Compression Ratio", fontsize=S["label_fs"], labelpad=0.0)
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=S["label_fs"], labelpad=1.0)
    ax.tick_params(axis="both", labelsize=S["tick_fs"], pad=1.0, length=2.5, width=0.8)
    ax.set_axisbelow(True)
    ax.set_xlim(*X_LIM)
    ax.xaxis.set_major_locator(MultipleLocator(X_STEP))
    ax.grid(True, which="major", alpha=0.45, linewidth=0.45)
    ax.yaxis.set_major_locator(MultipleLocator(0.02))
    for s in ax.spines.values():
        s.set_linewidth(0.6)


def legend_handles():
    handles = [Line2D([], [], linestyle="none", marker="o", markersize=S["legend_ms"],
                      markerfacecolor="none", markeredgecolor="0.25",
                      markeredgewidth=0.8, label="ZSTD")]
    handles += [Line2D([], [], linestyle="none", marker=marker(key), markersize=S["legend_ms"],
                       markerfacecolor=color(key), markeredgecolor="black",
                       markeredgewidth=0.4, label=label(key))
                for key in METHODS]
    return handles


def build(data: pd.DataFrame, alphas):
    fig = plt.figure(figsize=FIGSIZE, layout="constrained")
    fig.get_layout_engine().set(h_pad=0.01, w_pad=0.015, hspace=0.0, wspace=0.03,
                                rect=(0.004, 0.004, 0.992, 0.992))
    axes = fig.subplots(1, len(alphas), sharex=True, sharey=True)
    for i, (ax, alpha) in enumerate(zip(axes, alphas)):
        panel(ax, data[data.alpha == alpha], rf"$\alpha = {alpha}$", "VUS-PR" if i == 0 else None)
    for ax in axes[1:]:
        ax.tick_params(axis="y", labelleft=False)

    # Both panels share the y range of all markers.
    ys = pd.concat([data.q_post, data.q_pre])
    lo, hi = ys.min(), ys.max()
    axes[0].set_ylim(lo - Y_PAD * (hi - lo), hi + Y_PAD_TOP * (hi - lo))

    fig.legend(handles=legend_handles(), loc="outside right upper", ncol=1,
               frameon=True, fontsize=S["legend_fs"], handletextpad=0.15,
               handlelength=1.0, labelspacing=0.25, borderaxespad=0.15, borderpad=0.3)
    return fig


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tables", default="results/figures/anomaly_detection")
    ap.add_argument("--dataset", default="multievent32")
    ap.add_argument("--detector", default="Sub_PCA")
    ap.add_argument("--alphas", nargs="+", default=["0.75", "0.95"])
    ap.add_argument("--out", default="results/figures/anomaly_detection")
    args = ap.parse_args()

    use_libertine(mathtext=True)
    data = load(args.tables, args.dataset, args.detector, args.alphas)
    fig = build(data, args.alphas)

    os.makedirs(args.out, exist_ok=True)
    stem = os.path.join(args.out, f"anomaly_tradeoff_{args.dataset}_{args.detector}")
    fig.savefig(f"{stem}.pdf", dpi=400)
    fig.savefig(f"{stem}.png", dpi=400)
    with open(f"{stem}.csv", "w") as fh:
        fh.write(f"# {args.detector} on {args.dataset}, one panel per alpha\n"
                 "# q_post: mean test VUS-PR over the series (marker y); cr: geometric mean"
                 " compression ratio (marker x)\n"
                 "# q_pre: mean test VUS-PR of the uncompressed series (ZSTD ring y);"
                 " zstd_cr: geometric mean ZSTD ratio (ZSTD ring x)\n")
        data[["alpha", "method", "q_post", "q_pre", "cr", "zstd_cr"]].to_csv(fh, index=False)
    print(f"-> {stem}.pdf")


if __name__ == "__main__":
    main()
