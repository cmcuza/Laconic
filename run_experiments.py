import os
# numba is used for CPU JIT only (models/regression.py, models/clustering.py);
# CUDA is disabled here so a GPU present on the machine can't change results.
os.environ["NUMBA_DISABLE_CUDA"] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = ""
# A CLI invocation emits only one root trace per dataset. Export it inline so
# MLflow cannot lazily create its async exporter thread during Python 3.12
# interpreter shutdown. An explicit environment setting still takes priority.
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
from optimizer.oracle import build_oracle



def _set_seeds(seed: int | None):
    if seed is None: return
    random.seed(seed)
    np.random.seed(seed)
    try:
        # Imported only after mlflow.autolog() is enabled at the entry point.
        import torch
    except ImportError:
        return
    else:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)


# Optimizers whose YAML pop_size/elitism are PERCENTAGES of the evaluation
# budget, so there is no budget-free reading of their config and --budget is
# mandatory. See CLAUDE.md "What defines an experiment".
_PERCENTAGE_SIZED_OPTIMIZERS = ("genetic", "preference")


def _apply_budget_override(optimizer_name: str, kwargs: Dict[str, Any], budget: int) -> None:
    """Translate a single "total evaluations" number into each optimizer
    family's own kwargs, so one --budget value means the same thing across
    every optimizer being compared in an experiment (alpha/random_state
    already work this way - see CLAUDE.md's "Key invariants").

    random: n_iter *is* total_budget. successive_halving/adaptive_halving:
    total_budget is a native constructor kwarg that caps the loop directly,
    regardless of init_points/n_iter. bosmp: total_budget = init_points + n_iter, so n_iter
    is solved keeping init_points (the warm-start count) fixed - a budget
    below init_points can't be honored (n_iter floors at 1) and is printed
    rather than silently absorbed. genetic/preference: YAML pop_size and
    elitism are percentages of the evaluation budget; gens and tournament_k
    remain absolute algorithm parameters. The standard 100/200/500 schedules
    are exact, while arbitrary budgets may differ after integer rounding.
    """
    if optimizer_name == "random":
        kwargs["n_iter"] = budget
    elif optimizer_name in ("successive_halving", "adaptive_halving"):
        kwargs["total_budget"] = budget
    elif optimizer_name in ("bosmp", "adaedge"):
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


# Source optimizers stage 2 can collapse - the ones that actually search. A run
# with any other optimizer has no K fold-winners to vote over.
_CONSENSUS_SOURCE_OPTIMIZERS = ("genetic", "adaedge", "bosmp")


def _run_consensus(cfg: ExperimentConfig, args, model_name: str, df) -> None:
    """Stage 2 for the cell just written, so ONE command produces a full result.

    The figures read `rank_agg_b1_mean_fitness`, which only stage 2 writes;
    without this a finished search leaves `results/` holding K fold-winners and
    no deployable pipeline, and every figure comes out empty. Scoped to the CSV
    this invocation produced and idempotent - seeds already voted on are skipped
    - so it costs nothing on a re-run.

    Deliberately NOT a fail-fast site, unlike the rest of the pipeline: the
    search is finished and its CSV is safely on disk, so aborting here would
    discard hours of compute over a selection step that re-runs in minutes. It
    reports loudly and prints the command to resume instead.
    """
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
    """One root trace per concrete model/dataset experiment invocation."""
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
    ap.add_argument("--compression", required=True, choices=["tersets_mab", "tersets_reduced", "sz", "mixpiece", "serfxor", "adaedge"])
    ap.add_argument("--optimizer", required=True, choices=["random", "successive_halving", "adaptive_halving", "genetic", "bosmp", "adaedge", "preference"])
    ap.add_argument("--model", required=False)       # name without .yaml; if omitted, run ALL models for the analytics
    ap.add_argument("--dataset", required=True)     # a dataset GROUP yaml in cfg/datasets/<task>/
    ap.add_argument("--cfg_root", default="cfg")
    ap.add_argument("--out_dir", default="results")
    ap.add_argument("--logs", default=".logs")
    ap.add_argument("--seeded-trace-dirs", dest="seeded_trace_dirs", action="store_true",
                    help="Write optimizer traces under an rs<N> segment "
                         "(.../budget_<N>/rs<N>/fold_<n>/) so runs differing only by "
                         "--random_state do not overwrite each other. Transitional: off "
                         "by default until the existing rs32 traces are migrated.")
    ap.add_argument("--hidden_weight", type=float, default=None, help="Override the optimizer's hidden_weight (the ground truth the elicitation is benchmarked against). Defaults to --alpha when omitted.")
    ap.add_argument("--random_state", type=int, required=True)  # identity dimension - never defaulted silently
    ap.add_argument(
        "--alpha", type=float, default=None,
        help="Override the optimizer's fitness weight (alpha * task_metric + (1-alpha) * "
             "compression term). Defaults to cfg/optimizer/<name>.yaml's kwargs.alpha when omitted.",
    )
    ap.add_argument(
        "--budget", type=int, default=None,
        help="Override the optimizer's total evaluation budget, translated to each optimizer "
             "family's own kwargs (see _apply_budget_override). Defaults to cfg/optimizer/<name>.yaml's "
             "own budget (n_iter / pop_size+gens*(...) / init_points+n_iter) when omitted.",
    )
    ap.add_argument(
        "--beta", type=float, default=None,
        help="Only used by --optimizer preference: override the Bradley-Terry sharpness "
             "constant. beta is calibrated OFFLINE per task with optimizer.preference."
             "calibrate_beta and held fixed for the run (never recalibrated live -- see that "
             "module's D10); the right value depends on the task's front geometry, so this "
             "override exists to avoid one global constant standing in for every task. "
             "Defaults to cfg/optimizer/preference.yaml's kwargs.beta when omitted.",
    )
    ap.add_argument(
        "--oracle", default="deterministic_oracle", choices=["deterministic_oracle", "stochastic_oracle"],
        help="Only used by --optimizer preference: which cfg/oracle/<name>.yaml to build the "
             "pairwise-comparison oracle from. hidden_weight is always the resolved --alpha (the "
             "ground truth the elicitation is benchmarked against), never read from the yaml.",
    )
    ap.add_argument("--no_mlflow", action="store_true", help="Disable MLflow experiment tracking for this invocation")
    ap.add_argument("--no-consensus", dest="no_consensus", action="store_true",
                    help="Skip stage 2 (cross-fold consensus selection) after the search. "
                         "The figures read the tree it writes, so a run without it is only "
                         "half a result - see 'The two stages of a result' in CLAUDE.md.")
    args = ap.parse_args()

    tracking.init_tracking(enabled=not args.no_mlflow)
    if not args.no_mlflow:
        # Enable all installed framework integrations before runner/model imports.
        mlflow.autolog()

    # Paths
    analytics_dir   = os.path.join(args.cfg_root, "analytics", args.analytics)
    compression_yml = os.path.join(args.cfg_root, "compression", f"{args.compression}.yaml")
    optimizer_yml   = os.path.join(args.cfg_root, "optimizer",  f"{args.optimizer}.yaml")
    dataset_yml    = os.path.join(args.cfg_root, "datasets", args.analytics, f"{(args.dataset or 'default')}.yaml")

    # Load configs
    compression_configuration  = load_yaml(compression_yml)   # contains bounds + static compressor config
    optimization_configuration   = load_yaml(optimizer_yml)     # contains optimizer + kwargs
    data_configuration  = load_yaml(dataset_yml)      # contains dataset list + loader + split

    # Choose model yamls
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

    # Setup a random state initial seed
    data_configuration["split"]["random_state"] = args.random_state
    optimization_configuration["kwargs"]["random_state"] = args.random_state
    compression_configuration["random_state"] = args.random_state
    # alpha is a first-class experiment dimension. A CLI/suite value wins over
    # the optimizer YAML; the YAML value is only a backwards-compatible
    # fallback when --alpha is omitted.
    experiment_alpha = resolve_experiment_alpha(
        args.alpha, optimization_configuration["kwargs"]
    )
    if optimization_configuration["name"] == "preference":
        if args.beta is not None:
            optimization_configuration["kwargs"]["beta"] = args.beta
        # hidden_weight always comes from the resolved --alpha (the ground
        # truth the elicitation is benchmarked against), never from the yaml.
        oracle_configuration = load_yaml(os.path.join(args.cfg_root, "oracle", f"{args.oracle}.yaml"))
        oracle_kwargs = {**oracle_configuration["kwargs"], "hidden_weight": experiment_alpha}
        optimization_configuration["kwargs"]["oracle"] = build_oracle(oracle_configuration["name"], oracle_kwargs)
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

    # Iterate: models x dataset
    for mp in model_paths:
        model_cfg = load_yaml(mp)   # model_name, model_kwargs, metrics
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

                compressor=compression_configuration["name"],             # matches registry key
                compressor_bounds=compression_configuration["bounds"],          # dict of parameter:[lo,hi]
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

            # Hand to the generic runner (runs all folds, writes CSV, returns
            # df). Keep --no_mlflow a true opt-out by bypassing the decorated
            # function instead of creating a trace in the default experiment.
            dispatch = _dispatch_experiment if args.no_mlflow else _dispatch_experiment_traced
            df = dispatch(cfg)
            print(df.tail())

            if not args.no_consensus:
                _run_consensus(cfg, args, model_cfg["model"]["name"], df)
