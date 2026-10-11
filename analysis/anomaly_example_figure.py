"""Raw and reconstructed series around one anomaly, with the detector score on the raw series.

The left axis shows the uncompressed series and the reconstruction of the pipeline
each method selected. The right axis shows the detector score on the uncompressed
series, as a percentile of its scores on the test range (dotted). The shaded band is
the labelled anomaly. Reads the example table written by analysis/anomaly_paper_tables.py.
Run from the repository root:

    python analysis/anomaly_example_figure.py \
        --table results/figures/anomaly_detection/anomaly_example_266_IOPS_id_7_WebService_tr_4010_1st_4110_74700_74785.csv
"""
from __future__ import annotations
import argparse
import os
import shutil
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

from analysis.anomaly_figure_style import color, label, use_libertine

# Compressor of each method, in legend order.
METHODS = ["sz", "mixpiece", "serfxor", "adaedge", "laconic"]
LACONIC = "laconic"
# ADA is dashed: in this window its reconstruction equals SXOR's.
DASHED = {"adaedge"}

FIGSIZE = (241.14749 / 72.27, 1.2107)   # ACM single column, inches
PANEL = [0.10131, 0.26913, 0.79616, 0.62694]   # [left, bottom, width, height], figure fractions
METHOD_LEGEND_AT = (0.49939, 0.95010)          # centre of the legend row, figure fractions
VALUE_LIM = (19.97684, 66.93925)
SCORE_LIM = (-13.32889, 112.59210)
EVENT = (74707, 74733)
SERIES_NAME = "266_IOPS"

S = {
    "label_fs": 8.0,
    "tick_fs": 7.0,
    "legend_fs": 7.0,
    "name_fs": 9.0,
    "raw_lw": 0.55,
    "rec_lw": 0.5,
    "laconic_lw": 0.75,
    "score_lw": 0.55,
    "spine_lw": 0.8,
}
RAW = "#1b1b1f"
BAND = "#f2c9c4"
BAND_ALPHA = 0.5
DASH = (0, (2.2, 1.6))


def check(d: pd.DataFrame) -> None:
    """Every curve must lie inside the fixed axis limits."""
    values = d[["raw"] + [label(k) for k in METHODS]]
    lo, hi = float(values.min().min()), float(values.max().max())
    if not (VALUE_LIM[0] < lo and hi < VALUE_LIM[1]):
        raise ValueError(f"values [{lo}, {hi}] outside VALUE_LIM {VALUE_LIM}")
    if not (SCORE_LIM[0] < d.score_raw_pct.min() and d.score_raw_pct.max() < SCORE_LIM[1]):
        raise ValueError(f"scores outside SCORE_LIM {SCORE_LIM}")


def build(d: pd.DataFrame):
    t = d.t
    fig = plt.figure(figsize=FIGSIZE)
    ax = fig.add_axes(PANEL)
    tw = ax.twinx()

    ax.axvspan(*EVENT, color=BAND, alpha=BAND_ALPHA, linewidth=0, zorder=0)
    line = dict(solid_capstyle="round", solid_joinstyle="round", dash_capstyle="butt")
    # Reconstructions go below the raw series, LACONIC on top.
    for key in METHODS:
        lw, z = (S["laconic_lw"], 4) if key == LACONIC else (S["rec_lw"], 2)
        ax.plot(t, d[label(key)], color=color(key), linewidth=lw,
                linestyle=DASH if key in DASHED else "-", alpha=0.9, zorder=z, **line)
    ax.plot(t, d.raw, color=RAW, linewidth=S["raw_lw"], zorder=3, **line)

    tw.plot(t, d.score_raw_pct, color=RAW, linewidth=S["score_lw"], linestyle=":",
            alpha=0.85, zorder=1, **line)

    ax.text(0.988, 0.035, SERIES_NAME, transform=ax.transAxes, ha="right", va="bottom",
            fontsize=S["name_fs"], zorder=6)

    ax.set_xlim(t.iloc[0], t.iloc[-1] + 1)
    ax.set_ylim(*VALUE_LIM)
    tw.set_ylim(*SCORE_LIM)
    ax.xaxis.set_major_locator(MultipleLocator(10))
    ax.yaxis.set_major_locator(MultipleLocator(20))
    tw.set_yticks([0, 50, 100])
    ax.grid(True, axis="x", color="#b0b0b0", linewidth=0.6, alpha=0.45)

    ax.set_xlabel("Time Index", fontsize=S["label_fs"], labelpad=0.0)
    ax.set_ylabel("Time Series Value", fontsize=S["label_fs"], labelpad=2)
    tw.set_ylabel("Raw Anomaly Score", fontsize=S["label_fs"], labelpad=1)
    ax.tick_params(labelsize=S["tick_fs"], length=2.5, width=S["spine_lw"])
    tw.tick_params(labelsize=S["tick_fs"], length=2.5, width=S["spine_lw"], pad=1)
    for s in ax.spines.values():
        s.set_linewidth(S["spine_lw"])
    tw.spines["right"].set_linewidth(S["spine_lw"])

    handles = [Line2D([], [], color=RAW, linewidth=0.9, label="Raw")]
    handles += [Line2D([], [], color=color(key), linewidth=1.0, label=label(key),
                       linestyle=DASH if key in DASHED else "-")
                for key in METHODS]
    fig.legend(handles=handles, loc="center", bbox_to_anchor=METHOD_LEGEND_AT,
               ncol=len(handles), frameon=False, fontsize=S["legend_fs"], handlelength=1.2,
               handletextpad=0.3, columnspacing=0.8, borderpad=0.0, borderaxespad=0.0)
    return fig


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--table", required=True)
    ap.add_argument("--out", default="results/figures/anomaly_detection")
    args = ap.parse_args()

    use_libertine()
    d = pd.read_csv(args.table, comment="#")
    check(d)
    fig = build(d)

    os.makedirs(args.out, exist_ok=True)
    stem = os.path.join(args.out, os.path.basename(args.table).replace(".csv", "_figure"))
    fig.savefig(f"{stem}.pdf", dpi=300, facecolor="white")
    fig.savefig(f"{stem}.png", dpi=300, facecolor="white")
    # The figure draws the table's columns unchanged, so the table is the sidecar.
    shutil.copyfile(args.table, f"{stem}.csv")
    print(f"-> {stem}.pdf")


if __name__ == "__main__":
    main()
