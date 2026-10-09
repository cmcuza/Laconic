"""Cross-fold consensus selection (the assessment's Design B), computed offline."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))
if str(Path(__file__).resolve().parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parent))

from fix_deployment_model import _analytics_cfg, _backend_for, _dataset_group

from experiments.deployment_model import DEPLOYMENT_FOLD, fit_deployment_model

from compression.utils import (compress_and_decompress_batch,
                               compress_and_decompress_batch_cr, BatchCR)
from data.loaders import build_loaders, build_splitter
from evals.metrics import combine_fitness, evaluate_metrics, get_metric
from models.base import _CachedModelMixin
from experiments.objectives import (
    ClassificationObjective,
    ClusteringObjective,
    ForecastingObjective,
    RegressionObjective,
)
from evals.rank_aggregation import (SCORE_AGGREGATIONS, consensus_order,
                                    mean_fitness_scores, rank_positions)

TASKS = ("classification", "clustering", "regression", "forecasting")
MATRIX_NAME = "cross_evaluation.csv"
_SOURCE_LOG_SUBDIR = {"genetic": "_genetic_opt", "adaedge": "_adaedge_opt",
                      "bosmp": "_bosmp_opt"}
_NON_PARAM_COLUMNS = {
    "run_id", "evaluation", "timestamp_utc", "phase", "proposal_source", "reward",
    "objective_elapsed_sec", "global_best_before", "global_best_after", "is_new_global_best",
}


def _param_columns(frame: pd.DataFrame) -> List[str]:
    return [c for c in frame.columns
            if c not in _NON_PARAM_COLUMNS and not c.endswith("_model_space")]


def _build_model_for(task: str):
    if task == "classification":
        from models.classification import build_model
    elif task == "clustering":
        from models.clustering import build_model
    elif task == "regression":
        from models.regression import build_model
    else:
        from models.forecasting import build_model
    return build_model


class FoldVoters:
    """The K per-fold objectives a candidate is scored against."""

    def __init__(self, task: str, dataset: str, model_name: str, random_state: int,
                 compressor: str, alpha: float, primary_metric: str, logs_root: str):
        self.task, self.dataset, self.alpha = task, dataset, float(alpha)
        group = _dataset_group(task, dataset)
        acfg = _analytics_cfg(task, model_name)
        loader = build_loaders(group["loader"]["name"], **(group["loader"].get("kwargs") or {}))
        build_model = _build_model_for(task)
        kwargs = dict(acfg["model"].get("kwargs") or {})
        kwargs["random_state"] = int(random_state)
        spec = get_metric(primary_metric)
        logs_dir = os.path.join(logs_root, task)
        self.backend = _backend_for(compressor)
        self.objectives: List[Any] = []

        if task == "forecasting":
            folds = list(loader.forward_chain(dataset))
            kwargs["horizon"] = loader.frc_h
            for fold, (X_tr, X_val, _unused_test) in enumerate(folds, start=1):
                model = build_model(model_name, kwargs)
                model.fit_cached(X_tr, dataset=dataset, random_state=int(random_state),
                                 fold=fold, splitter_name="forward_chain",
                                 train_indices=np.arange(len(X_tr)), out_dir=logs_dir)
                y_val, _ = model.predict(X_val)
                self.objectives.append(ForecastingObjective(
                    model=model, X_val=X_val, y_val=y_val,
                    backend=self.backend, alpha=self.alpha, spec=spec))
            return

        X_train, y_train = loader.load_train(dataset)
        np.random.seed(int(random_state))
        splitter = build_splitter(group["split"])
        splitter_name = type(splitter).__name__
        for fold, (tr_idx, val_idx) in enumerate(splitter.split(X_train, y_train), start=1):
            self._verify_split(logs_dir, dataset, model_name, random_state, fold, tr_idx)
            fold_kwargs = dict(kwargs)
            if task == "clustering":
                fold_kwargs["n_clusters"] = len(np.unique(y_train))
            model = build_model(model_name, fold_kwargs)
            fit_args = (X_train[tr_idx],) if task == "clustering" else (X_train[tr_idx], y_train[tr_idx])
            model.fit_cached(*fit_args, dataset=dataset, random_state=int(random_state),
                             fold=fold, splitter_name=splitter_name,
                             train_indices=tr_idx, out_dir=logs_dir)
            common = dict(model=model, X_val=X_train[val_idx], y_val=y_train[val_idx],
                          backend=self.backend, alpha=self.alpha, spec=spec)
            if task == "classification":
                self.objectives.append(ClassificationObjective(classes=model.get_classes(), **common))
            elif task == "clustering":
                self.objectives.append(ClusteringObjective(classes=len(np.unique(y_train)), **common))
            else:
                self.objectives.append(RegressionObjective(**common))

    @staticmethod
    def _verify_split(logs_dir: str, dataset: str, model_name: str, random_state: int,
                      fold: int, train_indices: np.ndarray) -> None:
        meta_path = (Path(_CachedModelMixin._cache_dir(logs_dir, dataset, model_name,
                                                       int(random_state), fold)) / "meta.json")
        if not meta_path.exists():
            return
        stored = json.loads(meta_path.read_text()).get("train_idx_sha1")
        actual = hashlib.sha1(np.asarray(train_indices, dtype=np.int64).tobytes()).hexdigest()
        if stored and stored != actual:
            raise SystemExit(
                f"{dataset} fold {fold}: reconstructed train split does not match the one the "
                f"search used (cached {stored[:12]}..., rebuilt {actual[:12]}...). Refusing to "
                "cross-evaluate against a model trained on different data."
            )

    def __len__(self) -> int:
        return len(self.objectives)

    def cross_evaluate(self, candidates: List[Dict[str, float]]) -> np.ndarray:
        """(n_candidates, K, 2) of (task_metric, avg_cr)."""
        out = np.empty((len(candidates), len(self.objectives), 2), dtype=float)
        for c, candidate in enumerate(candidates):
            for k, objective in enumerate(self.objectives):
                out[c, k] = objective._compute_components(candidate)
        return out


def top_candidates(trace_paths: List[Path], n_best: int
                   ) -> Tuple[List[Dict[str, float]], List[Dict[int, int]]]:
    """Pool the `n_best` distinct pipelines from each fold with their per-fold ranks."""
    pooled: Dict[Tuple, Tuple[Dict[str, float], Dict[int, int]]] = {}
    for fold, path in enumerate(trace_paths, start=1):
        frame = pd.read_csv(path)
        columns = _param_columns(frame)
        ranked = (frame.sort_values("reward", ascending=False)
                       .drop_duplicates(subset=columns).head(n_best))
        for rank, (_, row) in enumerate(ranked.iterrows(), start=1):
            candidate = {c: float(row[c]) for c in columns}
            key = tuple(round(candidate[c], 10) for c in columns)
            if key not in pooled:
                pooled[key] = (candidate, {})
            pooled[key][1][fold] = rank
    return [c for c, _ in pooled.values()], [r for _, r in pooled.values()]


def subset_to(n_best: int, candidates: List[Dict[str, float]], ranks: List[Dict[int, int]],
              matrix: np.ndarray
              ) -> Tuple[List[Dict[str, float]], List[Dict[int, int]], np.ndarray]:
    """Restrict a matrix measured at a larger B to level `n_best`."""
    keep = [i for i, r in enumerate(ranks)
            if any(rank <= n_best for rank in r.values())]
    return ([candidates[i] for i in keep],
            [{f: rk for f, rk in ranks[i].items() if rk <= n_best} for i in keep],
            matrix[keep])


def ballots_for(matrix: np.ndarray, alpha: float, weighting: str
                ) -> Tuple[List[np.ndarray], List[float]]:
    """One fold = one voter."""
    ballots, weights = [], []
    for k in range(matrix.shape[1]):
        metrics, ratios = matrix[:, k, 0], matrix[:, k, 1]
        if weighting == "split":
            ballots += [rank_positions(metrics), rank_positions(ratios)]
            weights += [alpha, 1.0 - alpha]
        else:
            ballots.append(rank_positions(
                [combine_fitness(m, c, alpha) for m, c in zip(metrics, ratios)]))
            weights.append(1.0)
    return ballots, weights


class TestScorer:
    """The deployment model every test column is scored with."""

    def __init__(self, logs_root: str) -> None:
        self._cache: Dict[Any, Dict[str, Any]] = {}
        self._logs_root = logs_root

    def get(self, task: str, dataset: str, model_name: str, random_state: int) -> Dict[str, Any]:
        key = (task, dataset, model_name, int(random_state))
        if key in self._cache:
            return self._cache[key]
        group = _dataset_group(task, dataset)
        acfg = _analytics_cfg(task, model_name)
        loader = build_loaders(group["loader"]["name"], **(group["loader"].get("kwargs") or {}))
        build_model = _build_model_for(task)
        kwargs = dict(acfg["model"].get("kwargs") or {})
        kwargs["random_state"] = int(random_state)
        report_metrics = acfg["metrics"]["extra"]
        logs_dir = os.path.join(self._logs_root, task)

        if task == "forecasting":
            folds = list(loader.forward_chain(dataset))
            kwargs["horizon"] = loader.frc_h
            X_train_last, X_val_last, X_test = folds[-1]
            model = fit_deployment_model(
                build_model, model_name, kwargs, np.concatenate([X_train_last, X_val_last]),
                dataset=dataset, random_state=int(random_state),
                splitter_name="forward_chain", logs_dir=logs_dir)
            y_test, _ = model.predict(X_test)
            entry = dict(task=task, model=model, X_test=np.asarray(X_test), y_test=y_test,
                         classes=None, needs_score=False, report_metrics=report_metrics)
            entry["baseline"] = evaluate_metrics(report_metrics, y_test, model.predict(X_test)[1],
                                                 None, classes=None, analytics=task)
            self._cache[key] = entry
            return entry

        X_train, y_train = loader.load_train(dataset)
        X_test, y_test = loader.load_test(dataset)
        splitter_name = type(build_splitter(group["split"])).__name__
        if task == "clustering":
            kwargs["n_clusters"] = len(np.unique(y_train))
            model = fit_deployment_model(build_model, model_name, kwargs, X_train,
                                         dataset=dataset, random_state=int(random_state),
                                         splitter_name=splitter_name, logs_dir=logs_dir)
            classes: Any = kwargs["n_clusters"]
            needs_score = False
            baseline = evaluate_metrics(report_metrics, y_test, model.predict(X_test), None,
                                        classes=classes, analytics=task)
        else:
            model = fit_deployment_model(build_model, model_name, kwargs, X_train, y_train,
                                         dataset=dataset, random_state=int(random_state),
                                         splitter_name=splitter_name, logs_dir=logs_dir)
            needs_score = any(get_metric(m).needs_score for m in report_metrics)
            if task == "regression":
                classes = None
                baseline = evaluate_metrics(report_metrics, y_test, model.predict(X_test), None,
                                            classes=None, analytics=task)
            else:
                classes = model.get_classes()
                pred, score = model.predict_both(X_test, need_score=needs_score)
                baseline = evaluate_metrics(report_metrics, y_test, pred, score,
                                            classes=classes, analytics=task)
        entry = dict(task=task, model=model, X_test=X_test, y_test=y_test, classes=classes,
                     needs_score=needs_score, report_metrics=report_metrics, baseline=baseline)
        self._cache[key] = entry
        return entry

    def score(self, entry: Dict[str, Any], backend, params: Dict[str, Any]
              ) -> Tuple[Dict[str, float], BatchCR]:
        """(test metrics, test CR aggregates) for one pipeline on the fixed test set."""
        task, model = entry["task"], entry["model"]
        if task == "forecasting":
            reconstruction, cr = compress_and_decompress_batch_cr([entry["X_test"]], backend, params)
            _, y_pred = model.predict(reconstruction[0])
            score = None
        elif task == "regression":
            X_test = entry["X_test"]
            reconstruction, crs = np.empty_like(X_test), []
            for d in range(X_test.shape[2]):
                rec_dim, cr_dim = compress_and_decompress_batch_cr(X_test[..., d], backend, params)
                reconstruction[..., d] = rec_dim
                crs.append(cr_dim)
            cr = BatchCR.combine(crs)
            y_pred, score = model.predict(reconstruction), None
        else:
            reconstruction, cr = compress_and_decompress_batch_cr(entry["X_test"], backend, params)
            if task == "clustering":
                y_pred, score = model.predict(reconstruction), None
            else:
                y_pred, score = model.predict_both(reconstruction, need_score=entry["needs_score"])
        metrics = evaluate_metrics(entry["report_metrics"], entry["y_test"], y_pred, score,
                                   classes=entry["classes"], analytics=task)
        return metrics, cr


def optimizer_slug(n_best: int, aggregation: str, weighting: str) -> str:
    """The results-tree name this configuration writes under."""
    slug = f"rank_agg_b{int(n_best)}_{aggregation}"
    if aggregation in SCORE_AGGREGATIONS or weighting == "split":
        return slug
    return f"{slug}_scal"


def _matrix_path(logs_root: str, task: str, dataset: str, model: str, compressor: str,
                 alpha: float, budget: int, random_state: int | None = None) -> Path:
    base = (Path(logs_root) / task / "_rank_aggregation" / dataset / model / compressor
            / f"alpha_{alpha:g}" / f"budget_{budget}")
    if random_state is not None:
        base = base / f"rs{random_state}"
    return base / MATRIX_NAME


def _write_matrix(path: Path, candidates: List[Dict[str, float]], ranks: List[Dict[int, int]],
                  matrix: np.ndarray, param_columns: List[str], n_best_measured: int) -> None:
    rows = []
    for c, (candidate, rank_map) in enumerate(zip(candidates, ranks)):
        for k in range(matrix.shape[1]):
            row = {"candidate": c, "fold": k + 1,
                   "rank_in_fold": rank_map.get(k + 1, ""),
                   "n_best_measured": n_best_measured,
                   "task_metric": float(matrix[c, k, 0]), "avg_cr": float(matrix[c, k, 1])}
            row.update({name: float(candidate[name]) for name in param_columns})
            rows.append(row)
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(path, index=False)


def _read_matrix(path: Path
                 ) -> Tuple[List[Dict[str, float]], List[Dict[int, int]], np.ndarray, int]:
    frame = pd.read_csv(path)
    meta = ("candidate", "fold", "rank_in_fold", "n_best_measured", "task_metric", "avg_cr")
    param_columns = [c for c in frame.columns if c not in meta]
    n_candidates = int(frame["candidate"].max()) + 1
    n_folds = int(frame["fold"].max())
    matrix = np.empty((n_candidates, n_folds, 2), dtype=float)
    candidates: List[Dict[str, float]] = [dict() for _ in range(n_candidates)]
    ranks: List[Dict[int, int]] = [dict() for _ in range(n_candidates)]
    for _, row in frame.iterrows():
        c, k = int(row["candidate"]), int(row["fold"])
        matrix[c, k - 1] = (row["task_metric"], row["avg_cr"])
        candidates[c] = {name: float(row[name]) for name in param_columns}
        if pd.notna(row["rank_in_fold"]) and str(row["rank_in_fold"]).strip() != "":
            ranks[c][k] = int(row["rank_in_fold"])
    return candidates, ranks, matrix, int(frame["n_best_measured"].iloc[0])


def _trace_dir(logs_root: str, task: str, source_optimizer: str, dataset: str, model: str,
               compressor: str, alpha: float, budget: int, random_state: int | None = None) -> Path:
    base = (Path(logs_root) / task / _SOURCE_LOG_SUBDIR[source_optimizer] / dataset / model
            / compressor / f"alpha_{alpha:g}" / f"budget_{budget}")
    if random_state is None:
        return base
    seeded = base / f"rs{random_state}"
    if any(seeded.glob("fold_*/evaluations.csv")):
        return seeded
    if _flat_trace_seed(base) == random_state:
        return base
    return seeded


def _flat_trace_seed(base: Path) -> int | None:
    for meta in sorted(base.glob("fold_*/metadata.json")):
        try:
            return int(json.loads(meta.read_text())["run_metadata"]["random_state"])
        except (OSError, ValueError, KeyError, TypeError):
            return None
    return None


_ROW_KEYS = ("dataset", "compressor", "optimizer", "model", "random_state",
             "alpha", "fold", "n_evaluations")


def _seed_already_written(out_csv: Path, random_state: int) -> bool:
    if not out_csv.exists():
        return False
    try:
        seeds = pd.read_csv(out_csv, usecols=["random_state"])["random_state"]
    except (ValueError, OSError, pd.errors.EmptyDataError, pd.errors.ParserError):
        return False
    return bool((seeds == random_state).any())


def _merge_write(out_csv: Path, pieces: List[pd.DataFrame], column_order) -> None:
    new = pd.concat(pieces, ignore_index=True)
    if out_csv.exists():
        prev = pd.read_csv(out_csv)
        keys = [c for c in _ROW_KEYS if c in prev.columns and c in new.columns]
        prev = prev[~prev.set_index(keys).index.isin(new.set_index(keys).index)]
        new = pd.concat([prev, new], ignore_index=True)
    new = new[[c for c in column_order if c in new.columns]]
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    new.to_csv(out_csv, index=False)


def process(path: Path, args, deployment: TestScorer,
            voters_cache: Dict[Any, FoldVoters], task: str | None = None) -> Dict[str, Any]:
    """One source results CSV -> one consensus per random_state it contains."""
    frame = pd.read_csv(path)
    folds = frame[frame["fold"] >= 1]
    if folds.empty or "best_params" not in frame.columns:
        return {"status": "skipped", "reason": "no fold rows"}
    first = folds.iloc[0]
    task = task or path.parts[1]
    slug = optimizer_slug(args.n_best, args.aggregation, args.weighting)
    out_csv = (Path(args.results_root) / task / str(first["compressor"]) / slug
               / str(first["model"]) / f"budget_{int(first['n_evaluations'])}"
               / f"alpha_{float(first['alpha']):g}" / f"{first['dataset']}.csv".lower())

    seeds = sorted(int(v) for v in folds["random_state"].dropna().unique())
    summaries, pieces = [], []
    for seed in seeds:
        summary, rows = _process_seed(frame[frame["random_state"] == seed], path, args,
                                      deployment, voters_cache, seed, out_csv, task)
        summaries.append((seed, summary))
        if rows is not None:
            pieces.append(rows)
    if pieces and not args.dry_run:
        _merge_write(out_csv, pieces, frame.columns)

    written = [(s, r) for s, r in summaries if r["status"] == "written"]
    live = [(s, r) for s, r in summaries if r["status"] in ("written", "dry-run")]
    if not live:
        reason = "; ".join(f"rs{s}: {r.get('reason')}" for s, r in summaries)
        return {"status": "skipped", "reason": reason, "out": out_csv}
    combined = {"status": "written" if written else "dry-run", "out": out_csv,
                "candidates": live[0][1].get("candidates"),
                "evaluations": sum(int(r.get("evaluations") or 0) for _, r in live),
                "elapsed": sum(float(r.get("elapsed") or 0.0) for _, r in live),
                "reused": all(r.get("reused") for _, r in live),
                "changed": any(bool(r.get("changed")) for _, r in live)}
    if len(seeds) > 1:
        combined["reason"] = (
            f"{combined['evaluations']} evals, {combined['elapsed']:.0f}s, "
            + ", ".join(f"rs{s}:" + ("CHANGED" if r.get("changed") else
                                     "kept" if r["status"] == "written" else
                                     "dry" if r["status"] == "dry-run" else
                                     str(r.get("reason")))
                        for s, r in summaries))
    return combined


def _process_seed(frame: pd.DataFrame, path: Path, args, deployment: TestScorer,
                  voters_cache: Dict[Any, FoldVoters], random_state: int,
                  out_csv: Path, task: str | None = None) -> tuple[Dict[str, Any], pd.DataFrame | None]:
    folds = frame[frame["fold"] >= 1]
    if folds.empty or "best_params" not in frame.columns:
        return {"status": "skipped", "reason": "no fold rows"}, None
    first = folds.iloc[0]
    task = task or path.parts[1]
    dataset, model = str(first["dataset"]), str(first["model"])
    compressor, alpha = str(first["compressor"]), float(first["alpha"])
    budget = int(first["n_evaluations"])
    primary = str(first["primary_metric"]).strip().lower()

    slug = optimizer_slug(args.n_best, args.aggregation, args.weighting)
    if args.skip_done and _seed_already_written(out_csv, random_state):
        return {"status": "skipped", "reason": "already written", "out": out_csv}, None

    trace_dir = _trace_dir(args.logs_dir, task, args.source_optimizer, dataset, model,
                           compressor, alpha, budget, random_state)
    matrix_path = _matrix_path(args.logs_dir, task, dataset, model, compressor, alpha, budget,
                               random_state if trace_dir.name.startswith("rs") else None)
    trace_paths = sorted(trace_dir.glob("fold_*/evaluations.csv"),
                         key=lambda p: int(p.parent.name.split("_")[1]))
    if not trace_paths:
        return {"status": "skipped", "reason": f"no traces under {trace_dir}"}, None

    measured = 0
    if matrix_path.exists() and not args.remeasure:
        measured = int(pd.read_csv(matrix_path, usecols=["n_best_measured"], nrows=1).iloc[0, 0])
    reusable = measured >= max(args.n_best, args.measure_n_best)
    measure_at = max(args.n_best, args.measure_n_best, measured if args.remeasure else 0)

    if args.dry_run:
        candidates, _ = top_candidates(trace_paths, measure_at if not reusable else args.n_best)
        return {"status": "dry-run", "candidates": len(candidates), "reused": reusable,
                "evaluations": 0 if reusable else len(candidates) * len(trace_paths),
                "note": f"stored matrix covers B<={measured}" if measured else "no stored matrix",
                "out": out_csv}, None

    backend = _backend_for(compressor)
    elapsed, n_evals = 0.0, 0
    if reusable:
        candidates, ranks, matrix, measured = _read_matrix(matrix_path)
    else:
        if args.aggregate_only:
            return {"status": "skipped",
                    "reason": f"stored matrix covers B<={measured}, need B={args.n_best}"}, None
        candidates, ranks = top_candidates(trace_paths, measure_at)
        key = (task, dataset, model, random_state, compressor)
        if key not in voters_cache:
            voters_cache[key] = FoldVoters(task, dataset, model, random_state, compressor,
                                           alpha, primary, args.logs_dir)
        voters = voters_cache[key]
        if len(voters) != len(trace_paths):
            return {"status": "skipped",
                    "reason": f"{len(trace_paths)} traces but {len(voters)} folds rebuilt"}, None
        backend = voters.backend
        started = time.perf_counter()
        matrix = voters.cross_evaluate(candidates)
        elapsed = time.perf_counter() - started
        n_evals = int(matrix.shape[0] * matrix.shape[1])
        _write_matrix(matrix_path, candidates, ranks, matrix,
                      _param_columns(pd.read_csv(trace_paths[0])), measure_at)
        measured = measure_at

    if measured > args.n_best:
        candidates, ranks, matrix = subset_to(args.n_best, candidates, ranks, matrix)

    if args.aggregation in SCORE_AGGREGATIONS:
        scores = mean_fitness_scores(matrix, alpha,
                                     normalize=args.aggregation == "mean_normalized")
        order, prefiltered = list(np.argsort(-scores, kind="mergesort")), False
    else:
        ballots, weights = ballots_for(matrix, alpha, args.weighting)
        order, scores, prefiltered = consensus_order(
            ballots, weights, args.aggregation, kemeny_max_items=args.kemeny_max_items)
    winner_index = order[0]
    winner = candidates[winner_index]

    winners = [(c, fold) for c, rank_map in enumerate(ranks)
               for fold, rank in rank_map.items() if rank == 1]
    argmax_index = max(
        winners, key=lambda cf: combine_fitness(matrix[cf[0], cf[1] - 1, 0],
                                                matrix[cf[0], cf[1] - 1, 1], alpha))[0]

    entry = deployment.get(task, dataset, model, random_state)
    winner_params = backend.params_from_vector(winner)
    test_metrics, test_cr_stats = deployment.score(entry, backend, winner_params)
    test_cr = test_cr_stats.mean_cr

    updates = {"best_params": json.dumps(winner_params, default=float),
               "optimizer": slug,
               "test_avg_cr": round(float(test_cr), 4),
               "test_pooled_cr": round(float(test_cr_stats.pooled_cr), 4),
               "optimization_time": round(elapsed, 2)}
    for name, value in test_metrics.items():
        updates[f"test_{name}"] = round(float(value), 4)

    out = frame.copy()
    for column, value in updates.items():
        out[column] = value
    for position, (_, row) in enumerate(out[out["fold"] >= 1].iterrows()):
        fold_index = int(row["fold"]) - 1
        metric_value, ratio = matrix[winner_index, fold_index]
        out.loc[row.name, "val_avg_cr"] = round(float(ratio), 4)
        out.loc[row.name, f"val_{primary}"] = round(float(metric_value), 4)
    for column in [c for c in out.columns
                   if c.startswith("val_") and c not in ("val_avg_cr", f"val_{primary}")]:
        out[column] = np.nan

    baseline_primary = first.get(f"baseline_{primary}")
    if "acc_delta" in out.columns and baseline_primary is not None and float(baseline_primary) != 0:
        ba, ta = float(baseline_primary), float(test_metrics[primary])
        out["acc_delta"] = round(ba - ta, 4)
        out["acc_impact_%"] = round((ba - ta) / ba * 100.0, 2)
    if "baseline_cr" in out.columns:
        out["cr_improvement_x"] = round(float(test_cr) / float(first["baseline_cr"]), 2)

    numeric = out[out["fold"] >= 1].select_dtypes(include=[np.number]).columns
    for column in numeric:
        if column in ("fold", "random_state", "n_evaluations", "selection_evaluations"):
            continue
        out.loc[out["fold"] == 0, column] = round(
            float(pd.to_numeric(out.loc[out["fold"] >= 1, column], errors="coerce").mean()), 2)

    out = out[[c for c in frame.columns]]

    trace_out = matrix_path.parent / slug
    trace_out.mkdir(parents=True, exist_ok=True)
    (trace_out / "consensus.json").write_text(json.dumps({
        "source": {"optimizer": args.source_optimizer, "trace_dir": str(trace_dir),
                   "results_csv": str(path)},
        "design": "B (cross-fold): every candidate scored on every fold's validation split",
        "optimizer_slug": slug, "matrix": str(matrix_path), "matrix_reused": bool(reusable),
        "n_best_per_fold": args.n_best, "n_candidates": len(candidates),
        "n_folds": int(matrix.shape[1]), "selection_evaluations": n_evals,
        "elapsed_sec": round(elapsed, 2),
        "aggregation": args.aggregation, "weighting": args.weighting,
        "kemeny_prefiltered": bool(prefiltered),
        "consensus_order": [int(i) for i in order],
        "consensus_scores": [float(s) for s in scores],
        "n_best_measured": int(measured),
        "candidate_ranks": [{str(f): int(r) for f, r in rank_map.items()} for rank_map in ranks],
        "winner_vector": {k: float(v) for k, v in winner.items()},
        "winner_params": winner_params,
        "best_val_pick_vector": {k: float(v) for k, v in candidates[argmax_index].items()},
        "selection_changed": bool(winner_index != argmax_index),
    }, indent=2, default=float))

    return {"status": "written", "out": out_csv, "candidates": len(candidates),
            "evaluations": n_evals, "elapsed": elapsed, "reused": reusable,
            "changed": bool(winner_index != argmax_index)}, out


CONSENSUS_DEFAULTS = dict(n_best=1, measure_n_best=1, aggregation="mean_fitness",
                          weighting="split")


def consensus_for_results_csv(path, *, results_root="results", logs_dir=".logs",
                              source_optimizer="genetic", task=None, skip_done=True,
                              remeasure=False, aggregate_only=False, dry_run=False,
                              kemeny_max_items=8, **overrides):
    """Run stage 2 for ONE source results CSV."""
    settings = dict(CONSENSUS_DEFAULTS)
    settings.update(overrides)
    args = argparse.Namespace(
        results_root=str(results_root), logs_dir=str(logs_dir),
        source_optimizer=source_optimizer, skip_done=skip_done, remeasure=remeasure,
        aggregate_only=aggregate_only, dry_run=dry_run,
        kemeny_max_items=kemeny_max_items, **settings)
    path = Path(path)
    if task is None:
        task = path.resolve().relative_to(Path(results_root).resolve()).parts[0]
    return process(path, args, TestScorer(str(logs_dir)), {}, task=task)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--tasks", nargs="*", default=list(TASKS), choices=list(TASKS))
    ap.add_argument("--results-root", default="results")
    ap.add_argument("--logs-dir", default=".logs")
    ap.add_argument("--compressor", default="laconic")
    ap.add_argument("--source-optimizer", default="genetic", choices=sorted(_SOURCE_LOG_SUBDIR))
    ap.add_argument("--n-best", type=int, default=4,
                    help="Best distinct pipelines per fold to aggregate over (default 4).")
    ap.add_argument("--measure-n-best", type=int, default=0,
                    help="Measure the matrix at this B; smaller B are then free.")
    ap.add_argument("--aggregation", default="borda",
                    choices=("borda", "kemeny", "markov", "mean_fitness", "mean_normalized"),
                    help="Aggregation strategy.")
    ap.add_argument("--weighting", default="split", choices=("split", "scalarized"))
    ap.add_argument("--kemeny-max-items", type=int, default=8)
    ap.add_argument("--datasets", default=None, help="comma-separated subset")
    ap.add_argument("--budgets", default=None, help="comma-separated budgets (default: all)")
    ap.add_argument("--alphas", default=None, help="comma-separated alphas (default: all)")
    ap.add_argument("--aggregate-only", action="store_true",
                    help="Aggregate from stored cross_evaluation.csv matrices only.")
    ap.add_argument("--remeasure", action="store_true",
                    help="recompute the cross-evaluation matrix even if one is stored")
    ap.add_argument("--skip-done", action="store_true",
                    help="skip files whose output already exists - makes a run resumable")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args(argv)

    wanted = {d.strip() for d in args.datasets.split(",")} if args.datasets else None
    budgets = {int(b) for b in args.budgets.split(",")} if args.budgets else None
    alphas = {float(a) for a in args.alphas.split(",")} if args.alphas else None

    files: List[Path] = []
    for task in args.tasks:
        root = Path(args.results_root) / task / args.compressor / args.source_optimizer
        for path in sorted(root.rglob("*.csv")):
            if budgets is not None and not any(
                    p.startswith("budget_") and int(p.split("_", 1)[1]) in budgets for p in path.parts):
                continue
            if alphas is not None and not any(
                    p.startswith("alpha_") and float(p.split("_", 1)[1]) in alphas for p in path.parts):
                continue
            if wanted is not None:
                head = pd.read_csv(path, usecols=["dataset"], nrows=1)
                if head.empty or head["dataset"].iloc[0] not in wanted:
                    continue
            files.append(path)
    if not files:
        raise SystemExit(f"No {args.compressor}/{args.source_optimizer} results matched.")

    print(f"source        : {args.compressor}/{args.source_optimizer}")
    print(f"tasks         : {', '.join(args.tasks)}")
    print(f"files         : {len(files)}")
    slug = optimizer_slug(args.n_best, args.aggregation, args.weighting)
    print(f"B (per fold)  : {args.n_best}   aggregation: {args.aggregation}/{args.weighting}")
    print(f"writes to     : results/<task>/{args.compressor}/{slug}/...")
    print(f"mode          : {'aggregate-only' if args.aggregate_only else 'measure + aggregate'}"
          f"{' (forcing remeasure)' if args.remeasure else ''}")
    print()

    deployment = TestScorer(args.logs_dir)
    voters_cache: Dict[Any, FoldVoters] = {}
    totals = {"written": 0, "skipped": 0, "dry-run": 0, "changed": 0, "evaluations": 0}
    for index, path in enumerate(files, start=1):
        result = process(path, args, deployment, voters_cache)
        totals[result["status"]] = totals.get(result["status"], 0) + 1
        totals["evaluations"] += int(result.get("evaluations", 0) or 0)
        totals["changed"] += int(bool(result.get("changed")))
        detail = result.get("reason") or (
            f"{result.get('candidates')} candidates, {result.get('evaluations')} evals"
            f"{' (matrix reusable)' if result.get('reused') else ''}"
            if args.dry_run else
            f"{result.get('candidates')} candidates, "
            f"{'matrix reused' if result.get('reused') else f'{result.get("evaluations")} evals, {result.get("elapsed", 0):.0f}s'}, "
            f"{'CHANGED' if result.get('changed') else 'kept'} the best_val pick")
        print(f"[{index:3d}/{len(files)}] {path.relative_to(args.results_root)}  "
              f"{result['status']}: {detail}")

    print()
    print(f"written {totals['written']}, skipped {totals['skipped']}, "
          f"{totals['evaluations']} cross-evaluations, "
          f"selection changed the pick in {totals['changed']} file(s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
