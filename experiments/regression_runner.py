from __future__ import annotations
import os, json, time
from typing import Any, Dict, List, Tuple
import numpy as np
import pandas as pd

from data.loaders import build_loaders, build_splitter
from models.regression import build_model
from optimizer.utils import build_optimizer, bounds_list_to_tuple
from compression.backend import build_backend
from compression.utils import (compute_average_params, compress_and_decompress_batch,
                               compress_and_decompress_batch_cr, zstd_baseline_cr_stats, BatchCR)
from .deployment_model import fit_deployment_model

from .objectives import RegressionObjective
from tracking import mlflow_tracking as tracking

from evals.metrics import evaluate_metrics, get_metric


def run(cfg) -> pd.DataFrame:
    """Generic regression runner."""
    os.makedirs(cfg.out_dir, exist_ok=True)
    os.makedirs(cfg.logs_dir, exist_ok=True)

    loader = build_loaders(cfg.loader_name, **cfg.loader_kwargs)
    X_train, y_train = loader.load_train(cfg.dataset)
    X_test,  y_test  = loader.load_test(cfg.dataset)

    cfg.split["shuffle"] = True
    cfg.split["random_state"] = cfg.random_state

    splitter = build_splitter(cfg.split)

    backend = build_backend(cfg.compressor,
                            bounds_list_to_tuple(cfg.compressor_bounds),
                            {
                                "methods": cfg.compressor_methods,
                                "space_definition": cfg.compressor_space,
                            })

    space_definition = cfg.compressor_space
    optimizer = build_optimizer(cfg.optimizer, cfg.optimizer_kwargs)
    n_evaluations = optimizer.total_budget

    rows: List[Dict[str, Any]] = []


    primary_metric = cfg.metrics["primary"]
    report_metrics = cfg.metrics["extra"]

    run_ctx = tracking.start_experiment_run(
        cfg, splitter_name=type(splitter).__name__, primary_metric=primary_metric, n_evaluations=n_evaluations
    )

    dimensions = X_test.shape[2]
    baseline_cr_parts = []
    for d in range(dimensions):
        baseline_cr_parts.append(zstd_baseline_cr_stats(X_test[:, :, d]))

    baseline_stats = BatchCR.combine(baseline_cr_parts)
    baseline_cr = baseline_stats.mean_cr

    deploy_model = fit_deployment_model(
        build_model, cfg.model_name, cfg.model_kwargs, X_train, y_train,
        dataset=cfg.dataset, random_state=cfg.random_state,
        splitter_name=type(splitter).__name__, logs_dir=cfg.logs_dir)
    baseline_metrics = evaluate_metrics(
        report_metrics, y_test, deploy_model.predict(X_test), None,
        classes=None, analytics="regression")

    try:
        for fold, (tr_idx, val_idx) in enumerate(splitter.split(X_train[..., 0])):
            X_tr, X_val = X_train[tr_idx, ...], X_train[val_idx, ...]
            y_tr, y_val = y_train[tr_idx], y_train[val_idx]


            print(f"Training {cfg.model_name} on {cfg.dataset} fold {fold + 1} with rs {cfg.random_state}")
            model = build_model(cfg.model_name, cfg.model_kwargs)
            model.fit_cached(
                X_tr, y_tr,
                dataset=cfg.dataset,
                random_state=cfg.random_state,
                fold=fold+1,
                splitter_name=type(splitter).__name__,
                train_indices=tr_idx,
                out_dir=cfg.logs_dir,
            )

            spec = get_metric(primary_metric)

            objective = RegressionObjective(
                model=model,
                X_val=X_val, y_val=y_val,
                backend=backend,
                alpha=cfg.alpha,
                spec=spec,
                classes=None,
            )

            print("Maximizing Objective")
            t0 = time.time()
            maximize_kwargs = {
                "search_space": backend.bounds(),
                "space_definition": space_definition,
                "log_dir": cfg.logs_dir,
                "run_metadata": tracking.tag_run_metadata(
                    {
                        "task": cfg.task,
                        "dataset": cfg.dataset,
                        "fold": fold + 1,
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
                run_ctx.run_id, fold + 1, optimizer.last_run_dir
            )
            tune_secs = time.time() - t0
            best_params = backend.params_from_vector(best_vec)

            Xval_rec, val_avg_cr = np.empty_like(X_val), []
            Xtest_rec, test_cr_parts = np.empty_like(X_test), []
            for d in range(X_val.shape[2]):
                Xval_rec_dim, val_avg_cr_dim = compress_and_decompress_batch(X_val[..., d], backend, best_params)
                Xtest_rec_dim, test_cr_dim = compress_and_decompress_batch_cr(X_test[..., d], backend, best_params)
                Xval_rec[..., d] = Xval_rec_dim
                Xtest_rec[..., d] = Xtest_rec_dim
                val_avg_cr.append(val_avg_cr_dim)
                test_cr_parts.append(test_cr_dim)

            val_avg_cr = np.mean(val_avg_cr)
            test_cr_stats = BatchCR.combine(test_cr_parts)
            test_avg_cr = test_cr_stats.mean_cr

            y_val_pred  = model.predict(Xval_rec)
            y_test_pred = deploy_model.predict(Xtest_rec)
            val_metrics  = evaluate_metrics(report_metrics, y_val,  y_val_pred,  None, classes=None, analytics=cfg.task)
            test_metrics = evaluate_metrics(report_metrics, y_test, y_test_pred, None, classes=None, analytics=cfg.task)

            row = {
                "dataset": cfg.dataset,
                "fold": fold + 1,
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
                "test_pooled_cr": round(float(test_cr_stats.pooled_cr), 4),
                "baseline_pooled_cr": round(float(baseline_stats.pooled_cr), 4),

                "elicited_w_mean": getattr(optimizer, "last_w_mean", None),
                "elicited_n_comparisons": getattr(optimizer, "last_n_comparisons", None),
            }


            for name, val in baseline_metrics.items():
                row[f"baseline_{name}"] = round(float(val), 4)
            for name, val in val_metrics.items():
                row[f"val_{name}"] = round(float(val), 4)
            for name, val in test_metrics.items():
                row[f"test_{name}"] = round(float(val), 4)

            ba = baseline_metrics[primary_metric]
            ta = test_metrics[primary_metric]
            row[f"{primary_metric}_delta"] = None if ba == 0 else round((ba - ta), 4)
            row[f"{primary_metric}_impact_%"] = None if ba == 0 else round((ba - ta) / ba * 100.0, 2)

            row["cr_improvement_x"] = round(float(test_avg_cr) / float(baseline_cr), 2)
            rows.append(row)
            tracking.log_fold_row(row, fold + 1)

        df = pd.DataFrame(rows)

        mean_row = df.drop(columns=["fold", "best_params"]).mean(numeric_only=True).round(2)
        mean_row["dataset"] = cfg.dataset
        mean_row["fold"] = 0
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

        df = pd.concat([df, pd.DataFrame([mean_row])], ignore_index=True)

        tracking.log_summary_row(mean_row.to_dict())
        tracking.log_new_rows_artifact(df)

        out_dir = os.path.join(cfg.out_dir, f"budget_{n_evaluations}")
        out_dir = os.path.join(out_dir, f"alpha_{cfg.alpha:g}")
        os.makedirs(out_dir, exist_ok=True)
        out_csv = os.path.join(out_dir, f"{cfg.dataset}.csv".lower())
        unique_keys = ["dataset", "compressor", "optimizer", "model", "random_state", "alpha", "fold", "n_evaluations"]
        if os.path.exists(out_csv):
            prev = pd.read_csv(out_csv)
            prev = prev[~prev.set_index(unique_keys).index.isin(df.set_index(unique_keys).index)]
            df = pd.concat([prev, df], ignore_index=True)
        df.to_csv(out_csv, index=False)
        return df
    finally:
        tracking.end_run()
