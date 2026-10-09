"""Rewrite classification/clustering results so every fold's TEST columns come
from ONE model per (dataset, random_state, model), fit on all the training data.

The bug this repairs
--------------------
Those runners fit a model per CV fold and then scored the fixed test split with
*that fold's* model. Since `best_val` selects a different fold per method, two
methods on the same dataset were compared through two different models - the
test data was identical, the mapping from data to prediction was not. The
baseline metric moved across folds within one (dataset, random_state) in 50 of
52 cells, by up to 34% (classification `Wine`) and 49% (clustering `Coffee`) -
the two datasets that carry nearly every anomaly flag their tasks produce.

`experiments/deployment_model.py` is the fix at source; this script applies the
same rule to results produced before it, so they need no re-run. The two agree
by construction: both fit on all of `X_train` with the same cache key.

What it touches, and the check that proves it
---------------------------------------------
Only the test-side METRIC columns - `baseline_<m>`, `test_<m>`, `acc_delta`,
`acc_impact_%`. Everything else is left byte-identical: every `val_*` column,
`best_params`, `optimization_time`, every identity column, and - the useful part -
**both compression-ratio columns**.

`test_avg_cr` and `baseline_cr` MUST NOT MOVE. The test data is unchanged and
`best_params` is unchanged, so recompressing the test split has to reproduce the
stored ratio exactly. That gives a self-check the forecasting repair could not
have (there the window itself changed): recompute the CR, compare it to what is
stored, and refuse to write the file if they disagree - a mismatch means the
split or the parameter dict was rebuilt wrong, and the metrics would be wrong
in a way no amount of eyeballing would catch.

After the rewrite, `baseline_<m>` is constant across folds within each
(dataset, random_state, model). That is the post-condition worth grepping for.

    python scripts/fix_deployment_model.py --dry-run
    python scripts/fix_deployment_model.py --tasks classification
    python scripts/fix_deployment_model.py --datasets Coffee,Wine
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Tuple

# 8, not 1. The 1 that used to be here is the right cap for code running INSIDE
# a parallel optimizer's worker (see optimizer/genetic.py), but this script is a
# standalone batch job: capping it starves models that ask for parallelism -
# cfg/analytics/classification/proximity_forest.yaml requests n_jobs: 10 - and
# ProximityForest's predict over a large test set is the dominant cost here.
# setdefault, so an explicit env var from the caller still wins.
for _var in ("NUMBA_NUM_THREADS", "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS",
             "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_var, "8")

import numpy as np
import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from compression.backend import build_backend
from compression.utils import compress_and_decompress_batch
from data.loaders import build_loaders, build_splitter
from evals.metrics import evaluate_metrics, get_metric
from experiments.deployment_model import fit_deployment_model
from optimizer.utils import bounds_list_to_tuple

TASKS = ("classification", "clustering")
# Relative tolerance on the reproduced compression ratio. 5%, not the 0.2% that
# would be exact-to-rounding (the CSV stores test_avg_cr to 4 dp and SZ3/MixPiece
# round their own per-series CR to 2 before averaging). MixPiece does not reproduce
# its own logged compression ratio: the drift is small, one-directional
# (recomputed > stored) and already documented and accepted in
# nrmse_vs_degradation.py::build, measured there at +1.6%..+3.7% and here at
# +0.2%..+3.3%; adaedge inherits it whenever it selects MixPiece as its arm.
# Rejecting those would leave the files unrepaired for a reason unrelated to
# this script. The check still does its job - it exists to catch a test split or
# a parameter dict rebuilt WRONG, which is off by far more than a few percent
# (the stale successive_halving results it does reject drift +6% and +10.6%).
# Accepted drift is never silent: anything above NOTABLE_DRIFT is reported.
CR_TOLERANCE = 5e-2
NOTABLE_DRIFT = 2e-3


# ----------------------------------------------------------------------
# config lookup
# ----------------------------------------------------------------------
def _dataset_group(task: str, dataset: str) -> Dict[str, Any]:
    for path in sorted((Path("cfg/datasets") / task).glob("*.yaml")):
        cfg = yaml.safe_load(path.read_text())
        if dataset in (cfg.get("datasets") or []):
            return cfg
    raise SystemExit(f"No cfg/datasets/{task}/*.yaml lists dataset '{dataset}'.")


def _analytics_cfg(task: str, model_name: str) -> Dict[str, Any]:
    for path in sorted((Path("cfg/analytics") / task).glob("*.yaml")):
        cfg = yaml.safe_load(path.read_text())
        if str(cfg["model"]["name"]).lower() == model_name.lower():
            return cfg
    raise SystemExit(f"No cfg/analytics/{task}/*.yaml defines model '{model_name}'.")


def _backend_for(compressor: str):
    cfg = yaml.safe_load(Path(f"cfg/compression/{compressor}.yaml").read_text())
    return build_backend(cfg["name"], bounds_list_to_tuple(cfg["bounds"]),
                         {"methods": cfg.get("methods"), "space_definition": cfg["space"]})


# ----------------------------------------------------------------------
# the deployment model, cached across every file that needs it
# ----------------------------------------------------------------------
class Deployment:
    """Per (task, dataset, model, random_state): the fixed test split, the
    model fit on all training data, and the baseline metrics it produces."""

    def __init__(self, logs_root: str) -> None:
        self._cache: Dict[Tuple[str, str, str, int], Any] = {}
        # A ROOT, joined with the task below. run_experiments.py passes the
        # runner `os.path.join(args.logs, args.analytics)`, so the model cache
        # lives at .logs/<task>/_model_cache/... - writing to .logs/_model_cache
        # instead puts it in a tree the runners never read, and every model gets
        # refitted on both sides forever.
        self._logs_root = logs_root
        self.n_fits = 0
        self.planned: set = set()

    def note(self, task: str, dataset: str, model_name: str, random_state: int) -> None:
        """Record that a key WOULD be needed, without fitting anything."""
        self.planned.add((task, dataset, model_name, int(random_state)))

    def get(self, task: str, dataset: str, model_name: str, random_state: int):
        key = (task, dataset, model_name, int(random_state))
        if key in self._cache:
            return self._cache[key]

        from models.classification import build_model as build_cls
        from models.clustering import build_model as build_clu
        build = build_cls if task == "classification" else build_clu

        group = _dataset_group(task, dataset)
        acfg = _analytics_cfg(task, model_name)
        loader = build_loaders(group["loader"]["name"], **(group["loader"].get("kwargs") or {}))
        X_train, y_train = loader.load_train(dataset)
        X_test, y_test = loader.load_test(dataset)

        # Verbatim, as classification_runner/clustering_runner do - only the
        # regression runner injects shuffle/random_state. The name is part of
        # the model-cache key, so it has to match the runner exactly.
        splitter_name = type(build_splitter(group["split"])).__name__

        kwargs = dict(acfg["model"].get("kwargs") or {})
        kwargs["random_state"] = int(random_state)
        report_metrics = acfg["metrics"]["extra"]

        if task == "clustering":
            n_clusters = len(np.unique(y_train))
            kwargs["n_clusters"] = n_clusters
            model = fit_deployment_model(build, model_name, kwargs, X_train,
                                         dataset=dataset, random_state=int(random_state),
                                         splitter_name=splitter_name, logs_dir=os.path.join(self._logs_root, task))
            classes: Any = n_clusters
            baseline = evaluate_metrics(report_metrics, y_test, model.predict(X_test), None,
                                        classes=classes, analytics=task)
            needs_score = False
        else:
            model = fit_deployment_model(build, model_name, kwargs, X_train, y_train,
                                         dataset=dataset, random_state=int(random_state),
                                         splitter_name=splitter_name, logs_dir=os.path.join(self._logs_root, task))
            classes = model.get_classes()
            needs_score = any(get_metric(m).needs_score for m in report_metrics)
            pred, score = model.predict_both(X_test, need_score=needs_score)
            baseline = evaluate_metrics(report_metrics, y_test, pred, score,
                                        classes=classes, analytics=task)

        self.n_fits += 1
        entry = dict(model=model, X_test=X_test, y_test=y_test, classes=classes,
                     baseline=baseline, report_metrics=report_metrics,
                     needs_score=needs_score, task=task)
        self._cache[key] = entry
        return entry


def _score(entry, X_rec) -> Dict[str, float]:
    if entry["task"] == "clustering":
        pred, score = entry["model"].predict(X_rec), None
    else:
        pred, score = entry["model"].predict_both(X_rec, need_score=entry["needs_score"])
    return evaluate_metrics(entry["report_metrics"], entry["y_test"], pred, score,
                            classes=entry["classes"], analytics=entry["task"])


# ----------------------------------------------------------------------
# one results file
# ----------------------------------------------------------------------
def rewrite_row(row: pd.Series, entry, backend, primary: str) -> Tuple[Dict[str, Any], float]:
    """Updates for one fold row, plus the recomputed CR for the self-check."""
    params = json.loads(row["best_params"])
    X_rec, cr = compress_and_decompress_batch(entry["X_test"], backend, params)
    metrics = _score(entry, X_rec)

    updates: Dict[str, Any] = {}
    for name, value in entry["baseline"].items():
        updates[f"baseline_{name}"] = round(float(value), 4)
    for name, value in metrics.items():
        updates[f"test_{name}"] = round(float(value), 4)

    ba = float(entry["baseline"][primary])
    ta = float(metrics[primary])
    updates["acc_delta"] = None if ba == 0 else round(ba - ta, 4)
    updates["acc_impact_%"] = None if ba == 0 else round((ba - ta) / ba * 100.0, 2)
    return updates, float(cr)


def process(path: Path, task: str, deployment: Deployment, args) -> Tuple[int, bool]:
    frame = pd.read_csv(path)
    if "best_params" not in frame.columns or frame.empty:
        return 0, True

    folds = frame[frame["fold"] >= 1]
    if folds.empty:
        return 0, True
    if args.skip_done:
        # Exact, not a heuristic: the repair sets baseline_<m> to the deployment
        # model's own value, so a file whose stored baseline already equals it
        # has been done. Cheap - the model is cached after the first lookup, and
        # nothing is compressed to decide. (Constancy across folds alone would
        # not do: an easy dataset can have identical per-fold baselines by luck.)
        first = folds.iloc[0]
        entry = deployment.get(task, first["dataset"], first["model"], first["random_state"])
        done = True
        for name, value in entry["baseline"].items():
            col = f"baseline_{name}"
            if col not in frame.columns:
                continue
            stored = pd.to_numeric(folds[col], errors="coerce")
            if stored.isna().all() or not np.allclose(stored, round(float(value), 4), atol=5e-5):
                done = False
                break
        if done:
            return 0, True

    if args.dry_run:
        # Cheap on purpose: report scope without fitting or compressing anything.
        # A dry run that did the real work would double the cost of the whole
        # operation for no safety gain - the real run is already per-file atomic
        # (nothing is written unless every row's CR reproduces) and backs up first.
        for _, row in folds.iterrows():
            deployment.note(task, row["dataset"], row["model"], row["random_state"])
        return len(folds), True
    compressor = path.parts[2]
    backend = _backend_for(compressor)
    primary = str(folds.iloc[0]["primary_metric"]).strip().lower()

    pending: List[Tuple[int, Dict[str, Any]]] = []
    mismatches: List[str] = []
    max_drift = 0.0
    for idx, row in folds.iterrows():
        entry = deployment.get(task, row["dataset"], row["model"], row["random_state"])
        updates, cr = rewrite_row(row, entry, backend, primary)
        stored = float(row["test_avg_cr"])
        # The test data and best_params are unchanged, so the ratio must be too.
        drift = abs(cr - stored) / max(abs(stored), 1e-12)
        max_drift = max(max_drift, drift)
        if drift > CR_TOLERANCE:
            mismatches.append(f"fold {int(row['fold'])} rs {int(row['random_state'])}: "
                              f"test_avg_cr stored={stored:g} recomputed={cr:g} "
                              f"({drift:+.1%})")
        pending.append((idx, updates))

    if mismatches and not args.force:
        print(f"  SKIPPED (compression ratio did not reproduce): {path}")
        for line in mismatches[:3]:
            print(f"    {line}")
        return 0, False

    if max_drift > NOTABLE_DRIFT:
        print(f"  note: compression ratio drifted up to {max_drift:.1%} "
              f"(accepted, under the {CR_TOLERANCE:.0%} tolerance): {path.name}")

    if args.dry_run:
        return len(pending), True

    for idx, updates in pending:
        for column, value in updates.items():
            if column in frame.columns:
                frame.loc[idx, column] = value

    # fold 0 is the mean across folds, recomputed per random_state for the
    # columns we touched and nothing else.
    touched = sorted({c for _, u in pending for c in u})
    for rs, group in frame[frame["fold"] >= 1].groupby("random_state"):
        mask = (frame["fold"] == 0) & (frame["random_state"] == rs)
        if not mask.any():
            continue
        for column in touched:
            if column not in frame.columns:
                continue
            values = pd.to_numeric(group[column], errors="coerce").dropna()
            if len(values):
                frame.loc[mask, column] = round(float(values.mean()), 2)
    frame.to_csv(path, index=False)
    return len(pending), True


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--tasks", nargs="*", default=list(TASKS), choices=list(TASKS))
    parser.add_argument("--results-root", type=Path, default=Path("results"))
    parser.add_argument("--logs-dir", default=".logs")
    parser.add_argument("--datasets", default=None, help="comma-separated subset")
    parser.add_argument("--budgets", default=None,
                        help="comma-separated budgets (default: every budget_* directory found)")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--backup-dir", type=Path, default=None)
    parser.add_argument("--no-backup", action="store_true")
    parser.add_argument("--skip-done", action="store_true",
                        help="skip files already carrying the deployment model's baseline "
                             "- makes an interrupted run resumable")
    parser.add_argument("--force", action="store_true",
                        help="write even when the compression ratio does not reproduce")
    args = parser.parse_args(argv)

    wanted = {d.strip() for d in args.datasets.split(",")} if args.datasets else None
    budgets = ({int(b) for b in args.budgets.split(",") if b.strip()}
               if args.budgets else None)
    files: List[Tuple[Path, str]] = []
    for task in args.tasks:
        for path in sorted((args.results_root / task).rglob("*.csv")):
            # budget_<N> is a path segment, so filter before opening the file
            if budgets is not None and not any(
                    part.startswith("budget_") and int(part.split("_", 1)[1]) in budgets
                    for part in path.parts):
                continue
            if wanted is not None:
                head = pd.read_csv(path, usecols=["dataset"], nrows=1)
                if head.empty or head["dataset"].iloc[0] not in wanted:
                    continue
            files.append((path, task))
    if not files:
        raise SystemExit("No matching results files.")

    print(f"tasks           : {', '.join(args.tasks)}")
    print(f"files           : {len(files)}")
    print(f"budgets         : {sorted(budgets) if budgets else 'all'}")
    print(f"rule            : one model per (dataset, random_state, model), fit on all training data")
    if not args.dry_run and not args.no_backup:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        # Never under results/: measure_test_decompression_error.py and
        # test_error_vs_accuracy.py glob results/*/*/*/*/budget_100/alpha_0.75/*.csv,
        # which a copy under results/ would match, doubling every row they read.
        backup = args.backup_dir or Path(f".backups/deployment_model_{stamp}")
        if backup.exists():
            raise SystemExit(f"Backup target {backup} already exists.")
        backup.mkdir(parents=True)
        for task in args.tasks:
            shutil.copytree(args.results_root / task, backup / task)
        print(f"backup          : {backup}")
    elif args.no_backup:
        print("backup          : SKIPPED (--no-backup)")

    deployment = Deployment(args.logs_dir)
    started = time.time()
    total_rows = written = skipped = already = 0
    for index, (path, task) in enumerate(files, start=1):
        rows, ok = process(path, task, deployment, args)
        total_rows += rows
        if ok and rows == 0 and args.skip_done:
            already += 1
            continue                      # already repaired; keep the log quiet
        written += 1 if ok else 0
        skipped += 0 if ok else 1
        print(f"[{index:3d}/{len(files)}] {path.relative_to(args.results_root)}  {rows} rows"
              + ("" if ok else "  SKIPPED"), flush=True)

    if args.dry_run:
        # Same helper the cache itself uses, so the preview cannot disagree
        # with where the real run will look (it lowercases the whole path).
        from models.base import _CachedModelMixin
        from experiments.deployment_model import DEPLOYMENT_FOLD
        def _cached(t, d, m, r) -> bool:
            return Path(_CachedModelMixin._cache_dir(
                os.path.join(args.logs_dir, t), d, m, r, DEPLOYMENT_FOLD)).is_dir()
        cached = sum(1 for key in deployment.planned if _cached(*key))
        print(f"\nwould rewrite {total_rows} fold rows in {len(files)} files")
        print(f"deployment models needed: {len(deployment.planned)} "
              f"({cached} already cached, {len(deployment.planned) - cached} to fit)")
        for key in sorted(deployment.planned):
            t, d, m, r = key
            print(f"  {t:15s} {d:24s} {m:18s} rs{r:<4d} "
                  f"{'cached' if _cached(*key) else 'TO FIT'}")
        print(f"\nscoped in {time.time() - started:.1f}s - no model was fitted and nothing "
              f"was compressed. Run without --dry-run to do the work.")
        return 0
    if already:
        print(f"\nskipped {already} file(s) already repaired (--skip-done)")
    print(f"\nrewrote {total_rows} fold rows in {written} of {len(files) - already} remaining files "
          f"({deployment.n_fits} model fits) in {time.time() - started:.0f}s")
    if skipped:
        print(f"\n!! {skipped} file(s) left untouched because their stored test_avg_cr "
              f"could not be reproduced. That means the split or the parameter dict was "
              f"rebuilt differently, not that the metrics changed - investigate before "
              f"passing --force.")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
