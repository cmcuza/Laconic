"""Anomaly detection runner: one pipeline search per series of a TSB-AD-U corpus."""
from __future__ import annotations
import os, json, time
from typing import Any, Dict, List, Optional
import numpy as np
import pandas as pd

from data.loaders import build_loaders
from models.anomaly import build_model
from optimizer.utils import build_optimizer, bounds_list_to_tuple
from compression.backend import build_backend
from compression.utils import compute_average_params, reconstruct_array, zstd_baseline_cr
from evals.anomaly_metrics import AD_Q_MEASURES, TSB_AD_METRIC_KEYS, anomaly_q, evaluate_anomaly_tsb
from tracking import mlflow_tracking as tracking
from .objectives import AnomalyObjective


UNIQUE_KEYS = ["dataset", "compressor", "optimizer", "model", "random_state",
               "alpha", "fold", "n_evaluations", "q_measure"]


def run(cfg) -> pd.DataFrame:
    """
    Anomaly detection runner. One series is one unit of work, as in TSB-AD's
    Run_Detector_U.py: the optimizer searches a pipeline for each series on its
    validation range, and the row reports the chosen pipeline on the test range.

    cfg (ExperimentConfig) fields used:
      - dataset, loader_name, loader_kwargs (root, file_list, split_manifest)
      - model_name, model_kwargs, metrics (primary, extra, q_measure)
      - compressor, compressor_bounds, compressor_space, compressor_methods, alpha
      - optimizer, optimizer_kwargs
      - random_state, out_dir, logs_dir
    """
    os.makedirs(cfg.out_dir, exist_ok=True)
    os.makedirs(cfg.logs_dir, exist_ok=True)

    loader = build_loaders(cfg.loader_name, **cfg.loader_kwargs)
    series_names = loader.series_names()

    backend = build_backend(cfg.compressor,
                            bounds_list_to_tuple(cfg.compressor_bounds),
                            {"methods": cfg.compressor_methods,
                             "space_definition": cfg.compressor_space})
    space_definition = cfg.compressor_space or {}
    n_evaluations = build_optimizer(cfg.optimizer, cfg.optimizer_kwargs).total_budget

    q_measure = cfg.metrics.get("q_measure", "AUC-PR")
    if q_measure not in AD_Q_MEASURES:
        raise ValueError(f"metrics.q_measure must be one of {list(AD_Q_MEASURES)}, got '{q_measure}'")
    primary_metric = cfg.metrics["primary"]
    report_metrics = cfg.metrics["extra"]
    missing = set([primary_metric, *report_metrics]) - set(TSB_AD_METRIC_KEYS) - {q_measure}
    if missing:
        raise ValueError(f"metrics {sorted(missing)} are not computed under q_measure '{q_measure}'")

    run_ctx = tracking.start_experiment_run(
        cfg, splitter_name="none", primary_metric=primary_metric, n_evaluations=n_evaluations
    )

    out_csv = _output_path(cfg, n_evaluations)
    done = _completed_series(out_csv, cfg, n_evaluations, q_measure)
    if done:
        print(f"Resuming {cfg.dataset}: {len(done)}/{len(series_names)} series already done")

    rows: List[Dict[str, Any]] = []
    try:
        for fold, name in enumerate(series_names, start=1):
            if name in done:
                continue

            x, y, train_index = loader.load_series(name)
            val_end, test_start = loader.load_split(name)
            val, test = slice(0, val_end), slice(test_start, None)

            # The VUS-PR window is computed once on the uncompressed series.
            from TSB_AD.utils.slidingWindows import find_length_rank
            sliding_window = int(find_length_rank(x[:, 0].reshape(-1, 1), rank=1))

            detector = build_model(cfg.model_name, cfg.model_kwargs or {})
            # A new optimizer per series, so each search starts from the same seed.
            optimizer = build_optimizer(cfg.optimizer, cfg.optimizer_kwargs)
            print(f"[{fold}/{len(series_names)}] {name}  n={len(x)}  window={sliding_window}")

            row = {
                "dataset": cfg.dataset, "fold": fold, "model": cfg.model_name,
                "compressor": cfg.compressor, "optimizer": cfg.optimizer,
                "random_state": cfg.random_state, "alpha": cfg.alpha,
                "primary_metric": primary_metric, "q_measure": q_measure,
                "n_evaluations": n_evaluations,
                "mlflow_run_id": run_ctx.run_id, "git_commit": run_ctx.git_commit,
                "series": name, "series_length": int(len(x)),
                "n_anomaly_points": int(y.sum()), "train_index": int(train_index),
                "sliding_window": sliding_window,
                "val_end": int(val_end), "test_start": int(test_start),
            }

            # The detector on the uncompressed series.
            baseline_metrics, baseline_scores, error = _score(detector, x, y, q_measure, sliding_window)
            if error is not None:
                print(f"    skipped, detector failed on the uncompressed series: {error}")
                row.update(best_params=json.dumps({}), status=f"baseline_failed: {error}")
                rows.append(row)
                _write_rows(out_csv, rows, cfg, n_evaluations, q_measure, with_mean=False)
                continue
            baseline_cr = zstd_baseline_cr(x)
            q_pre_val = _q(y, baseline_scores, val, q_measure, sliding_window)
            q_pre_test = _q(y, baseline_scores, test, q_measure, sliding_window)

            objective = AnomalyObjective(
                x_clean=x, y=y, detector=detector, backend=backend, alpha=cfg.alpha,
                val_end=val_end, q_measure=q_measure, sliding_window=sliding_window,
            )
            t0 = time.time()
            best_vec = optimizer.maximize(
                objective,
                search_space=backend.bounds(),
                space_definition=space_definition,
                log_dir=cfg.logs_dir,
                run_metadata=tracking.tag_run_metadata(
                    {
                        "task": cfg.task,
                        "dataset": cfg.dataset,
                        "series": name,
                        "fold": fold,
                        "model": cfg.model_name,
                        "compressor": cfg.compressor,
                        "optimizer": cfg.optimizer,
                        "seeded_trace_dirs": cfg.seeded_trace_dirs,
                        "random_state": cfg.random_state,
                        "primary_metric": primary_metric,
                        "splitter": "none",
                    },
                    run_ctx,
                ),
            )
            tracking.log_optimizer_learning_curve(run_ctx.run_id, fold, optimizer.last_run_dir)
            search_secs = time.time() - t0
            best_params = backend.params_from_vector(best_vec)

            x_rec, test_avg_cr = reconstruct_array(x, backend, best_params)
            test_metrics, test_scores, error = _score(detector, x_rec, y, q_measure, sliding_window)
            if error is not None:
                print(f"    detector failed on the chosen pipeline: {error}")
                test_metrics = {k: float("nan") for k in report_metrics}

            row.update({
                "best_params": json.dumps(best_params),
                "optimization_time": round(search_secs, 2),
                "baseline_cr": round(float(baseline_cr), 4),
                "test_avg_cr": round(float(test_avg_cr), 4),
                "q_pre_val": _round(q_pre_val),
                "q_pre_test": _round(q_pre_test),
                "q_post_val": _round(_q(y, test_scores, val, q_measure, sliding_window)),
                "q_post_test": _round(_q(y, test_scores, test, q_measure, sliding_window)),
                "aucpr_pre_val": _round(_q(y, baseline_scores, val, "AUC-PR")),
                "aucpr_pre_test": _round(_q(y, baseline_scores, test, "AUC-PR")),
                "aucpr_post_val": _round(_q(y, test_scores, val, "AUC-PR")),
                "aucpr_post_test": _round(_q(y, test_scores, test, "AUC-PR")),
                "status": "ok" if error is None else f"detector_failed: {error}",
            })
            # Whole-series metrics, on the uncompressed series and on the reconstruction.
            for metric_name in report_metrics:
                row[f"baseline_{metric_name}"] = round(float(baseline_metrics[metric_name]), 4)
                row[f"test_{metric_name}"] = round(float(test_metrics[metric_name]), 4)
            base_q, test_q = baseline_metrics[primary_metric], test_metrics[primary_metric]
            row["q_delta"] = round(float(base_q - test_q), 4)
            row["q_impact_%"] = (None if base_q == 0 or not np.isfinite(test_q)
                                 else round((base_q - test_q) / base_q * 100.0, 2))
            row["cr_improvement_x"] = round(float(test_avg_cr) / float(baseline_cr), 2)

            rows.append(row)
            tracking.log_fold_row(row, fold)
            # Rows are written after every series, so a killed run can resume.
            _write_rows(out_csv, rows, cfg, n_evaluations, q_measure, with_mean=False)

        if not rows:
            print(f"Nothing to do for {cfg.dataset}: all series are done")
            return pd.read_csv(out_csv) if os.path.exists(out_csv) else pd.DataFrame()

        df = _write_rows(out_csv, rows, cfg, n_evaluations, q_measure, with_mean=True,
                         run_ctx=run_ctx, primary_metric=primary_metric)
        tracking.log_new_rows_artifact(df)
        return df
    finally:
        tracking.end_run()


def _q(y, scores, index_range: slice, measure: str, sliding_window: Optional[int] = None):
    """Q on one index range, or None when the range has no scores or only one class."""
    if scores is None:
        return None
    yy, ss = y[index_range], scores[index_range]
    if len(yy) == 0 or not (0 < int(yy.sum()) < len(yy)):
        return None
    return anomaly_q(yy, ss, measure, sliding_window)


def _round(value, digits: int = 6):
    return None if value is None else round(float(value), digits)


def _score(detector, series, y, q_measure, sliding_window):
    """(metrics, scores, None) on success, (None, None, reason) when the detector fails."""
    try:
        scores = detector.score(series)
        metrics = evaluate_anomaly_tsb(y, scores)
        if q_measure not in metrics:
            metrics[q_measure] = anomaly_q(y, scores, q_measure, sliding_window)
        return metrics, scores, None
    except Exception as err:                      # noqa: BLE001
        return None, None, f"{type(err).__name__}: {str(err)[:160]}"


def _output_path(cfg, n_evaluations) -> str:
    """results/anomaly/<compressor>/<optimizer>/<model>/budget_<N>/alpha_<a>/<dataset>.csv"""
    out_dir = os.path.join(cfg.out_dir, f"budget_{n_evaluations}", f"alpha_{cfg.alpha:g}")
    os.makedirs(out_dir, exist_ok=True)
    return os.path.join(out_dir, f"{cfg.dataset}.csv".lower())


def _completed_series(out_csv: str, cfg, n_evaluations, q_measure) -> set:
    """Series already in out_csv under the same run identity."""
    if not os.path.exists(out_csv):
        return set()
    prev = pd.read_csv(out_csv)
    match = (
        (prev["dataset"] == cfg.dataset)
        & (prev["compressor"] == cfg.compressor)
        & (prev["optimizer"] == cfg.optimizer)
        & (prev["model"] == cfg.model_name)
        & (prev["random_state"] == cfg.random_state)
        & (np.isclose(prev["alpha"], cfg.alpha))
        & (prev["n_evaluations"] == n_evaluations)
        & (prev["q_measure"] == q_measure)
        & (prev["fold"] != 0)
    )
    return set(prev.loc[match, "series"].astype(str))


def _write_rows(out_csv, rows, cfg, n_evaluations, q_measure, *, with_mean,
                run_ctx=None, primary_metric=None) -> pd.DataFrame:
    """Merge `rows` into out_csv, replacing rows with the same UNIQUE_KEYS."""
    df = pd.DataFrame(rows)

    if with_mean:
        # fold=0 is the mean over the series the detector could score.
        ok = df[df["status"] == "ok"]
        source = ok if len(ok) else df
        mean_row = source.drop(columns=["fold", "best_params"]).mean(numeric_only=True).round(4)
        mean_row["dataset"] = cfg.dataset
        mean_row["fold"] = 0
        mean_row["best_params"] = json.dumps(compute_average_params(ok) if len(ok) else {})
        mean_row["status"] = f"ok={len(ok)}/{len(df)}"
        mean_row["model"] = cfg.model_name
        mean_row["compressor"] = cfg.compressor
        mean_row["optimizer"] = cfg.optimizer
        mean_row["random_state"] = cfg.random_state
        mean_row["alpha"] = cfg.alpha
        mean_row["primary_metric"] = primary_metric
        mean_row["q_measure"] = q_measure
        mean_row["n_evaluations"] = n_evaluations
        mean_row["series"] = f"MEAN({len(df)} series)"
        if run_ctx is not None:
            mean_row["mlflow_run_id"] = run_ctx.run_id
            mean_row["git_commit"] = run_ctx.git_commit
        tracking.log_summary_row(mean_row.to_dict())
        df = pd.concat([df, pd.DataFrame([mean_row])], ignore_index=True)

    if os.path.exists(out_csv):
        prev = pd.read_csv(out_csv)
        prev = prev[~prev.set_index(UNIQUE_KEYS).index.isin(df.set_index(UNIQUE_KEYS).index)]
        df = pd.concat([prev, df], ignore_index=True)
    df.to_csv(out_csv, index=False)
    return df
