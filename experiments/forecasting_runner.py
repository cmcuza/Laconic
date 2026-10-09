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
# Optimizers whose contract is objective(c) -> (task_metric, avg_cr) rather
# than the scalarized float every other optimizer gets. Listed once here so
# adding a third cannot silently miss one of the four runners.
_COMPONENT_OBJECTIVE_OPTIMIZERS = ("preference",)

from .objectives import ForecastingObjective, PreferenceObjective
from tracking import mlflow_tracking as tracking

def run(cfg) -> pd.DataFrame:
    os.makedirs(cfg.out_dir, exist_ok=True)
    os.makedirs(cfg.logs_dir, exist_ok=True)

    primary_metric = cfg.metrics["primary"]
    report_metrics = cfg.metrics["extra"]
    rows: List[Dict[str, Any]] = []

    loader = build_loaders(cfg.loader_name, **cfg.loader_kwargs)


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

    run_ctx = tracking.start_experiment_run(
        cfg, splitter_name="forward_chain", primary_metric=primary_metric, n_evaluations=n_evaluations
    )


    # ---- one fixed test window for every fold ----------------------------
    # forward_chain gives each fold its own test window, and `best_val` then
    # picks a different fold per method - so two methods on the same dataset
    # were being compared on different data, against different baselines. Every
    # fold therefore reports on ONE window: the last partition's, scored by the
    # model trained on that partition's own train prefix.
    #
    # The last partition and not fold 1's: it is the genuine held-out future
    # (the final 20% of the series) and its train prefix ends immediately
    # before it. Scoring it with an earlier fold's model would leave a gap of
    # up to half the series between train and test, penalising a pipeline for
    # something unrelated to compression.
    #
    # Folds still differ - each searches on its own train/val split, so the
    # val_* columns and the optimizer traces are per fold exactly as before.
    # Only the test-side columns are now shared, which is what makes methods
    # comparable. scripts/fix_forecasting_test_window.py applied this same
    # rule retroactively to results produced before this change; the two agree
    # by construction, so re-running a repaired experiment reproduces it.
    folds = list(loader.forward_chain(cfg.dataset))
    if not folds:
        raise RuntimeError(f"forward_chain yielded no folds for {cfg.dataset}.")
    cfg.model_kwargs["horizon"] = loader.frc_h
    fixed_fold = len(folds)
    X_train_last, X_val_last, X_test_fixed = folds[-1]
    # train + val, matching the other three tasks: their deployment model is fit
    # on ALL the training data (the union of every fold's train and val split),
    # so forecasting fits on everything before the test window rather than
    # holding its last fold's validation split out for no reason. Contiguous by
    # construction - forward_chain lays out [train | val | test] in order.
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
            # Not reusable from test_model any more: that one is fit on
            # train+val, this one on train only, so the search still validates
            # on data its own model never saw.
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

            # objective on validation reconstructions
            spec = get_metric(primary_metric)
            raw_objective = ForecastingObjective(model=model,
                                             X_val=X_val,
                                             y_val=val_y_true,
                                             backend=backend,
                                             alpha=cfg.alpha,
                                             spec=spec)
            objective = PreferenceObjective(inner=raw_objective) if cfg.optimizer in _COMPONENT_OBJECTIVE_OPTIMIZERS else raw_objective

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
            
            # ---- evaluate on val and test with best params ----
            Xval_rec, val_avg_cr = compress_and_decompress_batch([X_val], backend, best_params)
            # This fold's pipeline, deployed on the one fixed window.
            # Single-element batch, so pooled == mean identically here; the column
            # is still written so the schema is uniform across tasks.
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


            # flatten metrics into columns: e.g., baseline_accuracy, val_f1_macro, test_roc_auc_macro
            for name, val in val_baseline_metrics.items():
                row[f"val_baseline_{name}"] = round(float(val), 4)
            for name, val in test_baseline_metrics.items():
                row[f"baseline_{name}"] = round(float(val), 4)
            for name, val in val_metrics.items():
                row[f"val_{name}"] = round(float(val), 4)
            for name, val in test_metrics.items():
                row[f"test_{name}"] = round(float(val), 4)

            # keep convenience deltas if accuracy exists
            ba = test_baseline_metrics[primary_metric]
            ta = test_metrics[primary_metric]
            row[f"{primary_metric}_delta"] = None if ba == 0 else round((ba - ta), 4)
            row[f"{primary_metric}_impact_%"] = None if ba == 0 else round((ba - ta) / ba * 100.0, 2)

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
