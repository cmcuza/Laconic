from __future__ import annotations
import os, json, time
from typing import Any, Dict, List, Tuple
import numpy as np
import pandas as pd

# plumbing
from data.loaders import build_loaders, build_splitter
from models.clustering import build_model          # returns a fresh model instance
from optimizer.utils import build_optimizer, bounds_list_to_tuple
from compression.backend import build_backend
from compression.utils import (compute_average_params, compress_and_decompress_batch,
                               compress_and_decompress_batch_cr, zstd_baseline_cr_stats)
from .deployment_model import fit_deployment_model
# Optimizers whose contract is objective(c) -> (task_metric, avg_cr) rather
# than the scalarized float every other optimizer gets. Listed once here so
# adding a third cannot silently miss one of the four runners.
_COMPONENT_OBJECTIVE_OPTIMIZERS = ("preference",)

from .objectives import ClusteringObjective, PreferenceObjective
from tracking import mlflow_tracking as tracking

from evals.metrics import evaluate_metrics, get_metric


def run(cfg) -> pd.DataFrame:
    """
    Generic clustering runner.

    cfg (ExperimentConfig) fields used:
      - dataset, loader_name, loader_kwargs
      - model_name, model_kwargs
      - compressor, comp_bounds, comp_params, alpha, cr_scale
      - optimizer, optimizer_kwargs
      - split (e.g., StratifiedShuffleSplit)
      - random_state, out_dir
    """
    os.makedirs(cfg.out_dir, exist_ok=True)
    os.makedirs(cfg.logs_dir, exist_ok=True)

    # ---- data ----
    loader = build_loaders(cfg.loader_name, **cfg.loader_kwargs)
    X_train, y_train = loader.load_train(cfg.dataset)
    X_test,  y_test  = loader.load_test(cfg.dataset)

    # ---- splitter ----
    splitter = build_splitter(cfg.split)

    # ---- compressor + optimizer ----
    backend = build_backend(cfg.compressor,
                            bounds_list_to_tuple(cfg.compressor_bounds),
                            {
                                "methods": cfg.compressor_methods,
                                "space_definition": cfg.compressor_space,
                            })
    space_definition = cfg.compressor_space
    optimizer = build_optimizer(cfg.optimizer, cfg.optimizer_kwargs)
    # Total objective evaluations this optimizer config will spend. Keeps
    # re-runs at a different budget from silently overwriting/mixing with
    # results at the old budget.
    n_evaluations = optimizer.total_budget

    rows: List[Dict[str, Any]] = []
    fold = 0

    primary_metric = cfg.metrics["primary"]
    report_metrics = cfg.metrics["extra"]

    run_ctx = tracking.start_experiment_run(
        cfg, splitter_name=type(splitter).__name__, primary_metric=primary_metric, n_evaluations=n_evaluations
    )

    # ---- baseline CR (zstd) and baseline accuracy on raw ----
    baseline_stats = zstd_baseline_cr_stats(X_test)
    baseline_cr = baseline_stats.mean_cr

    # One model for every fold's test columns - see experiments/deployment_model.py.
    # n_clusters comes from the FULL training labels, not a fold's val split, for
    # the same reason the fit does: it must not depend on which fold was picked.
    deploy_kwargs = dict(cfg.model_kwargs)
    deploy_n_clusters = len(np.unique(y_train))
    deploy_kwargs["n_clusters"] = deploy_n_clusters
    deploy_model = fit_deployment_model(
        build_model, cfg.model_name, deploy_kwargs, X_train,
        dataset=cfg.dataset, random_state=cfg.random_state,
        splitter_name=type(splitter).__name__, logs_dir=cfg.logs_dir)
    baseline_metrics = evaluate_metrics(
        report_metrics, y_test, deploy_model.predict(X_test), None,
        classes=deploy_n_clusters, analytics="clustering")

    try:
        for tr_idx, val_idx in splitter.split(X_train, y_train):
            fold += 1
            X_tr, X_val = X_train[tr_idx], X_train[val_idx]
            y_tr, y_val = y_train[tr_idx], y_train[val_idx]
            n_clusters = len(np.unique(y_val))
            cfg.model_kwargs['n_clusters'] = n_clusters
            print(f"Training {cfg.model_name} on {cfg.dataset} fold {fold} with rs {cfg.random_state}")
            model = build_model(cfg.model_name, cfg.model_kwargs)
            model.fit_cached(
                X_tr,
                dataset=cfg.dataset,
                random_state=cfg.random_state,
                fold=fold,
                splitter_name=type(splitter).__name__,
                train_indices=tr_idx,
                out_dir=cfg.logs_dir,
            )
            print(f"Predicting on test data with {cfg.model_name}")

            spec = get_metric(primary_metric)

            # ---- objective (tunes on validation reconstructions) ----
            raw_objective = ClusteringObjective(
                model=model,
                X_val=X_val, y_val=y_val,
                backend=backend,
                alpha=cfg.alpha,
                spec=spec,                     # <-- pass the MetricSpec
                classes=n_clusters,
            )
            objective = PreferenceObjective(inner=raw_objective) if cfg.optimizer in _COMPONENT_OBJECTIVE_OPTIMIZERS else raw_objective

            print("Maximizing Objective")
            # ---- optimize ----
            t0 = time.time()
            maximize_kwargs = {
                "search_space": backend.bounds(),
                "space_definition": space_definition,
                "log_dir": cfg.logs_dir,
                "run_metadata": tracking.tag_run_metadata(
                    {
                        "task": cfg.task,
                        "dataset": cfg.dataset,
                        "fold": fold,
                        "model": cfg.model_name,
                        "compressor": cfg.compressor,
                        "optimizer": cfg.optimizer,
                        "seeded_trace_dirs": cfg.seeded_trace_dirs,
                        "random_state": cfg.random_state,
                        "primary_metric": primary_metric,
                        "splitter": type(splitter).__name__,
                    },
                    run_ctx,
                ),
            }
            best_vec = optimizer.maximize(objective, **maximize_kwargs)
            tracking.log_optimizer_learning_curve(
                run_ctx.run_id, fold, optimizer.last_run_dir
            )
            tune_secs = time.time() - t0
            best_params = backend.params_from_vector(best_vec)

            # ---- evaluate on val and test with best params ----
            Xval_rec, val_avg_cr = compress_and_decompress_batch(X_val, backend, best_params)
            Xtest_rec, test_cr_stats = compress_and_decompress_batch_cr(X_test, backend, best_params)
            test_avg_cr = test_cr_stats.mean_cr

            y_val_pred  = model.predict(Xval_rec)
            y_test_pred = deploy_model.predict(Xtest_rec)
            val_metrics  = evaluate_metrics(report_metrics, y_val,  y_val_pred,  None,
                                            classes=n_clusters, analytics="clustering")
            test_metrics = evaluate_metrics(report_metrics, y_test, y_test_pred, None,
                                            classes=deploy_n_clusters, analytics="clustering")

            row = {
                "dataset": cfg.dataset,
                "fold": fold,
                "model": cfg.model_name,
                "compressor": cfg.compressor,
                "optimizer": cfg.optimizer,
                "random_state": cfg.random_state,
                "alpha": cfg.alpha,
                "primary_metric": primary_metric,
                "n_evaluations": n_evaluations,
                "mlflow_run_id": run_ctx.run_id,
                "git_commit": run_ctx.git_commit,

                "best_params": json.dumps(best_params),
                "optimization_time": round(tune_secs, 2),

                "baseline_cr": round(float(baseline_cr), 4),
                "val_avg_cr": round(float(val_avg_cr), 4),
                "test_avg_cr": round(float(test_avg_cr), 4),
                # Pooled (length-weighted harmonic) CR alongside the historical
                # mean-of-ratios. Reported only, never optimized on: the objective
                # still scores on the mean so stored results stay comparable.
                "test_pooled_cr": round(float(test_cr_stats.pooled_cr), 4),
                "baseline_pooled_cr": round(float(baseline_stats.pooled_cr), 4),

                "elicited_w_mean": getattr(optimizer, "last_w_mean", None),
                "elicited_n_comparisons": getattr(optimizer, "last_n_comparisons", None),
            }

            # flatten metrics into columns: e.g., baseline_accuracy, val_f1_macro, test_roc_auc_macro
            for name, val in baseline_metrics.items():
                row[f"baseline_{name}"] = round(float(val), 4)
            for name, val in val_metrics.items():
                row[f"val_{name}"] = round(float(val), 4)
            for name, val in test_metrics.items():
                row[f"test_{name}"] = round(float(val), 4)

            # keep convenience deltas if accuracy exists
            if "accuracy" in baseline_metrics and "test_accuracy" in {f"test_{k}": v for k, v in test_metrics.items()}:
                ba = baseline_metrics["accuracy"]
                ta = test_metrics["accuracy"]
                row["acc_delta"] = None if ba == 0 else round((ba - ta), 4)
                row["acc_impact_%"] = None if ba == 0 else round((ba - ta) / ba * 100.0, 2)

            row["cr_improvement_x"] = round(float(test_avg_cr) / float(baseline_cr), 2)
            rows.append(row)
            tracking.log_fold_row(row, fold)

        # ---- write CSV (append if exists) ----
        df = pd.DataFrame(rows)

        # ---- compute mean across folds and print/store ----
        mean_row = df.drop(columns=["fold", "best_params"]).mean(numeric_only=True).round(2)
        mean_row["dataset"] = cfg.dataset
        mean_row["fold"] = 0
        # Analyze the best parameters across folds
        # Compute the average of the best_params across folds
        mean_row["best_params"] = json.dumps(compute_average_params(df))

        mean_row["model"] = cfg.model_name
        mean_row["compressor"] = cfg.compressor
        mean_row["optimizer"] = cfg.optimizer
        mean_row["random_state"] = cfg.random_state
        mean_row["alpha"] = cfg.alpha
        mean_row["primary_metric"] = primary_metric
        mean_row["n_evaluations"] = n_evaluations
        mean_row["mlflow_run_id"] = run_ctx.run_id
        mean_row["git_commit"] = run_ctx.git_commit

        # Optionally, you can add the mean row to the DataFrame
        df = pd.concat([df, pd.DataFrame([mean_row])], ignore_index=True)

        tracking.log_summary_row(mean_row.to_dict())
        tracking.log_new_rows_artifact(df)

        # Budget and alpha each get their own subfolder segment so re-running
        # the same (compressor, optimizer, model) at a different evaluation
        # budget or a different task-vs-compression tradeoff can never silently
        # overwrite or mix with results at the old budget/alpha.
        out_dir = os.path.join(cfg.out_dir, f"budget_{n_evaluations}")
        out_dir = os.path.join(out_dir, f"alpha_{cfg.alpha:g}")
        os.makedirs(out_dir, exist_ok=True)
        out_csv = os.path.join(out_dir, f"{cfg.dataset}.csv".lower())
        unique_keys = ["dataset", "compressor", "optimizer", "model", "random_state", "alpha", "fold", "n_evaluations"]
        if os.path.exists(out_csv):
            prev = pd.read_csv(out_csv)
            # Re-running the same budget cleanly replaces matching rows instead of
            # accumulating duplicates that a later drop_duplicates(keep="first")
            # could resolve in favor of the stale row.
            prev = prev[~prev.set_index(unique_keys).index.isin(df.set_index(unique_keys).index)]
            df = pd.concat([prev, df], ignore_index=True)
        df.to_csv(out_csv, index=False)
        return df
    finally:
        tracking.end_run()
