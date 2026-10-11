"""Re-run the pipeline each method selected on one series and save the curves of the example figure.

No search is run. For each method the selected pipeline is read from its result CSV,
the series is compressed and reconstructed, and the detector scores the
reconstruction. The recomputed Q and AUC-PR must match the stored values on both
ranges, otherwise the script stops. Run from the repository root:

    python scripts/anomaly/replay_exemplar.py --alpha 0.95 \
        --series 266_IOPS_id_7_WebService_tr_4010_1st_4110.csv
"""
from __future__ import annotations
import argparse
import json
import os
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import numpy as np
import pandas as pd
import yaml

from compression.backend import build_backend
from compression.utils import reconstruct_array
from data.loaders import build_loaders
from evals.anomaly_metrics import anomaly_q
from models.anomaly import TSBADDetector
from optimizer.utils import bounds_list_to_tuple

# (compressor, optimizer, name of its arrays in the output file).
METHODS = [("sz", "bosmp", "SZ"), ("mixpiece", "bosmp", "MixPiece"),
           ("serfxor", "bosmp", "SerfXOR"), ("adaedge", "adaedge", "AdaEdge"),
           ("laconic", "genetic", "Laconic")]
# Stored values are rounded to 6 decimals.
TOLERANCE = 5e-7


def load_backend(cfg_root: str, compressor: str):
    with open(os.path.join(cfg_root, "compression", f"{compressor}.yaml")) as f:
        c = yaml.safe_load(f)
    return build_backend(c["name"], bounds_list_to_tuple(c["bounds"]),
                         {"methods": c.get("methods"), "space_definition": c["space"]})


def check(name: str, got: float, stored: float) -> None:
    if abs(got - stored) > TOLERANCE:
        raise AssertionError(f"{name}: recomputed {got!r}, stored {stored!r}")


def measure(y, scores, ranges, q_measure, w) -> dict:
    out = {}
    for side, sl in ranges.items():
        out[f"q_{side}"] = anomaly_q(y[sl], scores[sl], q_measure, w)
        out[f"aucpr_{side}"] = anomaly_q(y[sl], scores[sl], "AUC-PR")
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-root", default="results/anomaly")
    ap.add_argument("--cfg_root", default="cfg")
    ap.add_argument("--dataset", default="multievent32")
    ap.add_argument("--model", default="sub_pca", help="model yaml in cfg/analytics/anomaly/")
    ap.add_argument("--alpha", default="0.95")
    ap.add_argument("--budget", type=int, default=100)
    ap.add_argument("--series", default="266_IOPS_id_7_WebService_tr_4010_1st_4110.csv")
    ap.add_argument("--out", default="results/anomaly/exemplar")
    args = ap.parse_args()

    with open(os.path.join(args.cfg_root, "analytics", "anomaly", f"{args.model}.yaml")) as f:
        model_cfg = yaml.safe_load(f)
    with open(os.path.join(args.cfg_root, "datasets", "anomaly", f"{args.dataset}.yaml")) as f:
        data_cfg = yaml.safe_load(f)
    detector_name = model_cfg["model"]["name"]
    q_measure = model_cfg["metrics"].get("q_measure", "AUC-PR")

    # Pipeline each method selected
    winners = {}
    for comp, opt, _arm in METHODS:
        path = os.path.join(args.results_root, comp, opt, detector_name, f"budget_{args.budget}",
                            f"alpha_{args.alpha}", f"{args.dataset}.csv")
        d = pd.read_csv(path, float_precision="round_trip")
        d = d[(d.fold > 0) & (d.series == args.series)]
        if len(d) != 1 or d.status.iloc[0] != "ok" or d.q_measure.iloc[0] != q_measure:
            raise ValueError(f"{path}: expected one ok {q_measure} row for {args.series}")
        winners[comp] = d.iloc[0]

    # Series and split
    loader = build_loaders(data_cfg["loader"]["name"], **data_cfg["loader"]["kwargs"])
    x, y, train_index = loader.load_series(args.series)
    val_end, test_start = loader.load_split(args.series)
    ranges = {"pre_val": slice(0, val_end), "pre_test": slice(test_start, None)}

    from TSB_AD.utils.slidingWindows import find_length_rank
    w = int(find_length_rank(x[:, 0].reshape(-1, 1), rank=1))
    first = next(iter(winners.values()))
    if (w, val_end, test_start) != (int(first.sliding_window), int(first.val_end), int(first.test_start)):
        raise AssertionError("window or split differs from the stored run")

    detector = TSBADDetector(detector_name, **(model_cfg["model"]["kwargs"] or {}))
    arrays = {"x": x.ravel().astype(np.float32), "y": y.astype(np.int8)}

    # Uncompressed series
    score_pre = detector.score(x)
    for key, value in measure(y, score_pre, ranges, q_measure, w).items():
        check(f"uncompressed {key}", value, float(first[key]))
    arrays["score_pre"] = score_pre.astype(np.float32)

    # Reconstruction of each selected pipeline
    post = {"post_val": slice(0, val_end), "post_test": slice(test_start, None)}
    for comp, _opt, arm in METHODS:
        r = winners[comp]
        x_rec, cr = reconstruct_array(x, load_backend(args.cfg_root, comp), json.loads(r.best_params))
        score = detector.score(np.asarray(x_rec, float))
        for key, value in measure(y, score, post, q_measure, w).items():
            check(f"{comp} {key}", value, float(r[key]))
        print(f"  {comp:12s} CR {cr:9.2f} (stored {r.test_avg_cr:9.2f})  {q_measure} test {r.q_post_test:.6f}")
        arrays[f"x_rec_{arm}"] = np.asarray(x_rec, np.float32).ravel()
        arrays[f"score_post_{arm}"] = score.astype(np.float32)

    os.makedirs(args.out, exist_ok=True)
    out = os.path.join(args.out, args.series.replace(".csv", ".npz"))
    np.savez_compressed(out, w=w, val_end=val_end, test_start=test_start,
                        train_index=train_index, **arrays)
    print(f"all values match the stored run -> {out}")


if __name__ == "__main__":
    main()
