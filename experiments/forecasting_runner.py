from __future__ import annotations
import os, json, time
from typing import Any, Dict, List
import numpy as np
import pandas as pd
from collections import Counter

from data.loaders import build_loaders
from models.forecasting import build_model
from optimizer.utils import build_optimizer, bounds_list_to_tuple
from compression.backend import build_backend
from compression.utils import (compute_average_params, compress_and_decompress_batch,
                               compress_and_decompress_batch_cr, zstd_baseline_cr_stats)
from evals.metrics import get_metric, evaluate_metrics
from .deployment_model import DEPLOYMENT_FOLD

from .objectives import ForecastingObjective
from tracking import mlflow_tracking as tracking

def run(cfg) -> pd.DataFrame:
    os.makedirs(cfg.out_dir, exist_ok=True)
    os.makedirs(cfg.logs_dir, exist_ok=True)

    primary_metric = cfg.metrics["primary"]
    report_metrics = cfg.metrics["extra"]
    rows: List[Dict[str, Any]] = []

    loader = build_loaders(cfg.loader_name, **cfg.loader_kwargs)


    backend = build_backend(cfg.compressor,
                            bounds_list_to_tuple(cfg.compressor_bounds),
                            {
                                "methods": cfg.compressor_methods,
                                "space_definition": cfg.compressor_space,
                            })
    space_definition = cfg.compressor_space
    optimizer = build_optimizer(cfg.optimizer, cfg.optimizer_kwargs)
    n_evaluations = optimizer.total_budget

    run_ctx = tracking.start_experiment_run(
        cfg, splitter_name="forward_chain", primary_metric=primary_metric, n_evaluations=n_evaluations
    )


    folds = list(loader.forward_chain(cfg.dataset))
    if not folds:
        raise RuntimeError(f"forward_chain yielded no folds for {cfg.dataset}.")
    cfg.model_kwargs["horizon"] = loader.frc_h
    fixed_fold = len(folds)
    X_train_last, X_val_last, X_test_fixed = folds[-1]
    X_fit_fixed = np.concatenate([X_train_last, X_val_last])

    print(f"Fixed test window for {cfg.dataset}: fold {fixed_fold} "
          f"({X_test_fixed.size} points), shared by all {len(folds)} folds; "
          f"deployment model fit on {X_fit_fixed.size} train+val points")
    test_model = build_model(cfg.model_name, cfg.model_kwargs)
    test_model.fit_cached(
        X_fit_fixed,
        dataset=cfg.dataset,
        random_state=cfg.random_state,
        fold=DEPLOYMENT_FOLD,
        splitter_name="forward_chain",
        train_indices=np.arange(X_fit_fixed.size),
        out_dir=cfg.logs_dir,
    )
    baseline_stats = zstd_baseline_cr_stats(X_test_fixed[:, np.newaxis])
    baseline_cr = baseline_stats.mean_cr
    test_y_true, test_y_pred = test_model.predict(X_test_fixed)
    test_baseline_metrics = evaluate_metrics(
        report_metrics, y_true=test_y_true, y_pred=test_y_pred, analytics=cfg.task)

    fold = 0
    try:
        for X_train, X_val, _X_test_unused in folds:
            fold += 1

            print(f"Forecasting with {cfg.model_name} on {cfg.dataset} at fold {fold} with rs {cfg.random_state}")
            model = build_model(cfg.model_name, cfg.model_kwargs)
            model.fit_cached(
                X_train,
                dataset=cfg.dataset,
                random_state=cfg.random_state,
                fold=fold,
                splitter_name="forward_chain",
                train_indices=np.arange(X_train.size),
                out_dir=cfg.logs_dir,
            )

            val_y_true, val_y_pred = model.predict(X_val)
            val_baseline_metrics = evaluate_metrics(report_metrics, y_true=val_y_true, y_pred=val_y_pred, analytics=cfg.task)

            spec = get_metric(primary_metric)
            objective = ForecastingObjective(model=model,
                                             X_val=X_val,
                                             y_val=val_y_true,
                                             backend=backend,
                                             alpha=cfg.alpha,
                                             spec=spec)

            print("Maximize Objective")
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
                        "splitter": "forward_chain",
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
            
            Xval_rec, val_avg_cr = compress_and_decompress_batch([X_val], backend, best_params)
            Xtest_rec, test_cr_stats = compress_and_decompress_batch_cr([X_test_fixed], backend, best_params)
            test_avg_cr = test_cr_stats.mean_cr

            _,  comp_val_y_pred  = model.predict(Xval_rec[0])
            _, comp_test_y_pred = test_model.predict(Xtest_rec[0])
            val_metrics  = evaluate_metrics(report_metrics, y_true = val_y_true,  y_pred=comp_val_y_pred, analytics=cfg.task)
            test_metrics = evaluate_metrics(report_metrics, y_true=test_y_true, y_pred=comp_test_y_pred, analytics=cfg.task)

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
                "test_pooled_cr": round(float(test_cr_stats.pooled_cr), 4),
                "baseline_pooled_cr": round(float(baseline_stats.pooled_cr), 4),

                "elicited_w_mean": getattr(optimizer, "last_w_mean", None),
                "elicited_n_comparisons": getattr(optimizer, "last_n_comparisons", None),
            }


            for name, val in val_baseline_metrics.items():
                row[f"val_baseline_{name}"] = round(float(val), 4)
            for name, val in test_baseline_metrics.items():
                row[f"baseline_{name}"] = round(float(val), 4)
            for name, val in val_metrics.items():
                row[f"val_{name}"] = round(float(val), 4)
            for name, val in test_metrics.items():
                row[f"test_{name}"] = round(float(val), 4)

            ba = test_baseline_metrics[primary_metric]
            ta = test_metrics[primary_metric]
            row[f"{primary_metric}_delta"] = None if ba == 0 else round((ba - ta), 4)
            row[f"{primary_metric}_impact_%"] = None if ba == 0 else round((ba - ta) / ba * 100.0, 2)

            row["cr_improvement_x"] = round(float(test_avg_cr) / float(baseline_cr), 2)
            rows.append(row)
            tracking.log_fold_row(row, fold)

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
