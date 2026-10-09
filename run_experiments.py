import os
os.environ["NUMBA_DISABLE_CUDA"] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ.setdefault("MLFLOW_ENABLE_ASYNC_TRACE_LOGGING", "false")
import argparse, yaml, os, glob
import mlflow
from mlflow.entities import SpanType

import random
import numpy as np
from typing import Any, Dict, List

from experiment_config import (
    ExperimentConfig,
    resolve_experiment_alpha,
    resolve_genetic_percentages_for_budget,
)
from tracking import mlflow_tracking as tracking


def _set_seeds(seed: int | None):
    if seed is None: return
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
    except ImportError:
        return
    else:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)


_PERCENTAGE_SIZED_OPTIMIZERS = ("genetic",)


def _apply_budget_override(optimizer_name: str, kwargs: Dict[str, Any], budget: int) -> None:
    if optimizer_name in ("bosmp", "adaedge"):
        init_points = int(kwargs["init_points"])
        kwargs["n_iter"] = max(1, budget - init_points)
        realized = init_points + kwargs["n_iter"]
        if realized != budget:
            print(f"[budget] {optimizer_name} can't go below its init_points={init_points} warm-start; "
                  f"using n_iter={kwargs['n_iter']} -> {realized} evaluations")
    elif optimizer_name in _PERCENTAGE_SIZED_OPTIMIZERS:
        resolved_kwargs, realized = resolve_genetic_percentages_for_budget(kwargs, budget)
        kwargs.update(resolved_kwargs)
        if realized != budget:
            print(
                "[budget] {} cannot hit {} exactly with "
                "pop_size={}/elitism={}; using gens={} -> {} evaluations".format(
                    optimizer_name,
                    budget,
                    kwargs["pop_size"],
                    kwargs["elitism"],
                    kwargs["gens"],
                    realized,
                )
            )
    else:
        raise ValueError(
            f"--budget has no translation for optimizer '{optimizer_name}'. "
            "Add a branch here (step 4 of the add-optimizer checklist) instead of running at an unknown budget."
        )


def load_yaml(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)

def list_model_yamls(analytics_dir: str) -> List[str]:
    return sorted(glob.glob(os.path.join(analytics_dir, "*.yaml")))


def _dispatch_experiment(cfg: ExperimentConfig):
    if cfg.task == "classification":
        from experiments.classification_runner import run
    elif cfg.task == "clustering":
        from experiments.clustering_runner import run
    elif cfg.task == "forecasting":
        from experiments.forecasting_runner import run
    elif cfg.task == "regression":
        from experiments.regression_runner import run
    else:
        raise ValueError(f"No runner for analytics task '{cfg.task}'")
    return run(cfg)


_CONSENSUS_SOURCE_OPTIMIZERS = ("genetic", "adaedge", "bosmp")


def _run_consensus(cfg: ExperimentConfig, args, model_name: str, df) -> None:
    if args.optimizer not in _CONSENSUS_SOURCE_OPTIMIZERS:
        return
    import sys
    from pathlib import Path as _Path
    from experiments.result_paths import result_csv_path

    scripts_dir = str(_Path(__file__).resolve().parent / "scripts")
    if scripts_dir not in sys.path:
        sys.path.insert(0, scripts_dir)
    from rank_aggregation_selection import consensus_for_results_csv

    budget = int(df["n_evaluations"].iloc[0])
    path = result_csv_path(args.out_dir, args.analytics, args.compression,
                           args.optimizer, model_name, budget, cfg.alpha, cfg.dataset)
    print(f"\n[stage 2] consensus selection for {path}")
    try:
        print("[stage 2]", consensus_for_results_csv(
            path, results_root=args.out_dir, logs_dir=args.logs,
            source_optimizer=args.optimizer))
    except Exception as exc:                                    # noqa: BLE001
        print(f"[stage 2] FAILED ({type(exc).__name__}: {exc}). The search is safe on disk; "
              f"resume selection with:\n  python scripts/rank_aggregation_selection.py "
              f"--tasks {args.analytics} --compressor {args.compression} "
              f"--source-optimizer {args.optimizer} --datasets {cfg.dataset} "
              f"--n-best 1 --measure-n-best 1 --aggregation mean_fitness --skip-done")


@mlflow.trace(name="laconic_experiment", span_type=SpanType.CHAIN)
def _dispatch_experiment_traced(cfg: ExperimentConfig):
    mlflow.update_current_trace(
        tags={
            "laconic.task": cfg.task,
            "laconic.dataset": cfg.dataset,
            "laconic.model": cfg.model_name,
            "laconic.optimizer": cfg.optimizer,
            "laconic.compressor": cfg.compressor,
        }
    )
    return _dispatch_experiment(cfg)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--analytics", required=True, choices=["classification", "clustering", "forecasting", "regression"])
    ap.add_argument("--compression", required=True, choices=["laconic", "tersets_reduced", "sz", "mixpiece", "serfxor", "adaedge"])
    ap.add_argument("--optimizer", required=True, choices=["genetic", "bosmp", "adaedge"])
    ap.add_argument("--model", required=False)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--cfg_root", default="cfg")
    ap.add_argument("--out_dir", default="results")
    ap.add_argument("--logs", default=".logs")
    ap.add_argument("--seeded-trace-dirs", dest="seeded_trace_dirs", action="store_true",
                    help="Write traces under an rs<N> segment so seeds do not overwrite each other.")
    ap.add_argument("--random_state", type=int, required=True)
    ap.add_argument(
        "--alpha", type=float, default=None,
        help="Fitness weight on the task metric (default: optimizer YAML).",
    )
    ap.add_argument(
        "--budget", type=int, default=None,
        help="Total evaluation budget (default: optimizer YAML; required for genetic).",
    )
    ap.add_argument("--no_mlflow", action="store_true", help="Disable MLflow experiment tracking for this invocation")
    ap.add_argument("--no-consensus", dest="no_consensus", action="store_true",
                    help="Skip stage 2 (cross-fold consensus selection).")
    args = ap.parse_args()

    tracking.init_tracking(enabled=not args.no_mlflow)
    if not args.no_mlflow:
        mlflow.autolog()

    analytics_dir   = os.path.join(args.cfg_root, "analytics", args.analytics)
    compression_yml = os.path.join(args.cfg_root, "compression", f"{args.compression}.yaml")
    optimizer_yml   = os.path.join(args.cfg_root, "optimizer",  f"{args.optimizer}.yaml")
    dataset_yml    = os.path.join(args.cfg_root, "datasets", args.analytics, f"{(args.dataset or 'default')}.yaml")

    compression_configuration  = load_yaml(compression_yml)
    optimization_configuration   = load_yaml(optimizer_yml)
    data_configuration  = load_yaml(dataset_yml)

    model_paths = []
    if args.model:
        mp = os.path.join(analytics_dir, f"{args.model}.yaml")
        if not os.path.exists(mp):
            raise FileNotFoundError(f"Model config not found: {mp}")
        model_paths = [mp]
    else:
        model_paths = list_model_yamls(analytics_dir)
        if not model_paths:
            raise FileNotFoundError(f"No model yamls in {analytics_dir}")

    data_configuration["split"]["random_state"] = args.random_state
    optimization_configuration["kwargs"]["random_state"] = args.random_state
    compression_configuration["random_state"] = args.random_state
    experiment_alpha = resolve_experiment_alpha(
        args.alpha, optimization_configuration["kwargs"]
    )
    if optimization_configuration["name"] in _PERCENTAGE_SIZED_OPTIMIZERS:
        if args.budget is None:
            ap.error(
                f"--budget is required for {optimization_configuration['name']} because "
                "pop_size and elitism are configured as percentages"
            )
        _apply_budget_override(
            optimization_configuration["name"], optimization_configuration["kwargs"], args.budget
        )
    elif args.budget is not None:
        _apply_budget_override(
            optimization_configuration["name"],
            optimization_configuration["kwargs"],
            args.budget,
        )
    _set_seeds(args.random_state)

    if args.analytics == "forecasting" and args.optimizer in _PERCENTAGE_SIZED_OPTIMIZERS:
        optimization_configuration["kwargs"]["n_workers"] = 1

    for mp in model_paths:
        model_cfg = load_yaml(mp)
        model_cfg["model"]["kwargs"]['random_state'] = args.random_state
        for ds_name in data_configuration["datasets"]:
            loader_name   = data_configuration["loader"]["name"]
            loader_kwargs = data_configuration["loader"]["kwargs"]
            split         = data_configuration["split"]

            cfg = ExperimentConfig(
                task=args.analytics,
                model_name=model_cfg["model"]["name"],
                model_kwargs=model_cfg["model"]["kwargs"],
                metrics=model_cfg["metrics"],

                dataset=ds_name,
                loader_name=loader_name,
                loader_kwargs=loader_kwargs,
                split=split,

                compressor=compression_configuration["name"],
                compressor_bounds=compression_configuration["bounds"],
                compressor_space=compression_configuration["space"],
                compressor_methods=compression_configuration["methods"],

                optimizer=optimization_configuration["name"],
                optimizer_kwargs=optimization_configuration["kwargs"],
                alpha=experiment_alpha,

                random_state=args.random_state,
                out_dir=os.path.join(args.out_dir, args.analytics, args.compression, args.optimizer, model_cfg["model"]["name"]),
                logs_dir=os.path.join(args.logs, args.analytics),
                mlflow_enabled=not args.no_mlflow,
                seeded_trace_dirs=args.seeded_trace_dirs,
            )

            dispatch = _dispatch_experiment if args.no_mlflow else _dispatch_experiment_traced
            df = dispatch(cfg)
            print(df.tail())

            if not args.no_consensus:
                _run_consensus(cfg, args, model_cfg["model"]["name"], df)
