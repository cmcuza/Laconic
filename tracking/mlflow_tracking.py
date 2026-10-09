"""MLflow-backed experiment tracking for Laconic.

Every call to `experiments/<task>_runner.py::run(cfg)` opens one MLflow run
(one per invocation, as before), but it now nests inside a small, persistent
hierarchy instead of sitting alone in a per-task experiment:

    experiment "laconic-main"
    +-- run "task=<task>"                      (created once, reused forever)
        +-- run "model=<model_name>"           (created once, reused forever)
            +-- run "<dataset>_rs<random_state>"   (one per invocation, as before)

The task- and model-level runs are *container* runs: get_or_create_container_run
finds them by tag if they already exist (created by some earlier, separate
`run_experiments.py` process) or creates them fresh. Because two runs created
in separate Python processes can't use MLflow's native `nested=True` (that
relies on an in-process "active run" stack), nesting is instead recorded
explicitly via the `mlflow.parentRunId` tag - the same tag MLflow's own
`nested=True` sets, so the UI's run tree groups them identically.

This gives every result row the same identity/config columns as before (dataset,
compressor, optimizer, model, random_state, alpha, n_evaluations, git state),
plus a place to browse: `analysis/*.py` (rewritten to call log_figure_to_run)
attaches its rollup figures to the task/model container runs and one shared
`summary` run (get_or_create_summary_run), so opening the experiment in MLflow
and drilling into task -> model -> dataset shows the same figures you'd
otherwise only find by hand in results/figures/.

Disabled by passing `cfg.mlflow_enabled = False` (wired to `--no_mlflow` in
run_experiments.py) — every function below then becomes a no-op.
"""
from __future__ import annotations

import csv
import json
import os
import platform
import re
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

import mlflow
from mlflow.entities import Metric

# Row/mean-row fields that are numeric but already captured elsewhere
# (fold -> the MLflow step, random_state -> a run param) — logging them again
# as metrics would just be noise.
_NON_METRIC_KEYS = {"fold", "random_state"}

# Single shared experiment every task/model/dataset run nests inside.
_DEFAULT_EXPERIMENT = "laconic-main"
_DEFAULT_TRACKING_URI = "sqlite:///mlflow.db"
_OVERVIEW_RUN_NAME = "00-results-overview"


@dataclass
class RunContext:
    run_id: Optional[str]
    git_commit: Optional[str]


def init_tracking(enabled: bool = True, experiment_name: str = _DEFAULT_EXPERIMENT) -> None:
    """Configure the project tracking store and select its shared experiment."""
    if not enabled:
        return
    tracking_uri = os.environ.get("MLFLOW_TRACKING_URI", _DEFAULT_TRACKING_URI)
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_name)


def get_git_info() -> tuple[str, bool]:
    """(commit_hash, is_dirty). Uncommitted changes can't be pinned by a hash alone."""
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"]).decode().strip()
    status = subprocess.check_output(["git", "status", "--porcelain"]).decode()
    return commit, bool(status.strip())


def get_library_versions() -> Dict[str, str]:
    """Mirrors the version fields captured by models/base.py::_expected_meta."""
    import numpy
    import sklearn
    import aeon

    return {
        "python": platform.python_version(),
        "numpy": numpy.__version__,
        "scikit_learn": sklearn.__version__,
        "aeon": aeon.__version__,
    }


def _sanitize_metric_name(name: str) -> str:
    name = name.replace("%", "pct")
    return re.sub(r"[^a-zA-Z0-9_\-./ ]", "_", name)


def _escape_tag_value(value: str) -> str:
    return str(value).replace("\\", "\\\\").replace("'", "\\'")


def _resolve_experiment_id(experiment_name: str) -> str:
    mlflow.set_experiment(experiment_name)
    exp = mlflow.get_experiment_by_name(experiment_name)
    return exp.experiment_id


def _find_or_create_run(
    experiment_id: str,
    tags: Dict[str, str],
    run_name: str,
    parent_run_id: Optional[str] = None,
) -> str:
    """Finds a run in `experiment_id` whose tags match `tags` exactly, else
    creates one (nested under `parent_run_id` via the `mlflow.parentRunId` tag
    - see module docstring for why that's used instead of `nested=True`)."""
    filter_string = " and ".join(
        f"tags.`{k}` = '{_escape_tag_value(v)}'" for k, v in tags.items()
    )
    existing = mlflow.search_runs(
        experiment_ids=[experiment_id],
        filter_string=filter_string,
        max_results=1,
        output_format="list",
    )
    if existing:
        return existing[0].info.run_id

    with mlflow.start_run(experiment_id=experiment_id, run_name=run_name) as run:
        for k, v in tags.items():
            mlflow.set_tag(k, v)
        if parent_run_id is not None:
            mlflow.set_tag("mlflow.parentRunId", parent_run_id)
        return run.info.run_id


def get_or_create_container_run(
    task: str, model: Optional[str] = None, experiment_name: str = _DEFAULT_EXPERIMENT
) -> str:
    """Returns the run_id of the task-level container run (model=None) or the
    model-level one nested under it (model given). These persist across
    separate `run_experiments.py` invocations - unlike the per-invocation leaf
    run from start_experiment_run - so rollup figures/metrics logged to them
    (by the analysis scripts, usually from a later, separate process) accumulate
    in one place instead of each invocation getting its own disconnected copy.
    """
    experiment_id = _resolve_experiment_id(experiment_name)

    task_run_id = _find_or_create_run(
        experiment_id, {"laconic.level": "task", "laconic.task": task}, run_name=f"task={task}"
    )
    if model is None:
        return task_run_id

    return _find_or_create_run(
        experiment_id,
        {"laconic.level": "model", "laconic.task": task, "laconic.model": model},
        run_name=f"model={model}",
        parent_run_id=task_run_id,
    )


def get_or_create_summary_run(experiment_name: str = _DEFAULT_EXPERIMENT) -> str:
    """The one run at the top of the experiment meant to hold whole-sweep
    rollup figures (e.g. the combined cross-task trade-off panel) - the
    "observe the most important figures per experiment directly in MLflow"
    entry point."""
    experiment_id = _resolve_experiment_id(experiment_name)
    run_id = _find_or_create_run(
        experiment_id,
        {"laconic.level": "summary"},
        run_name=_OVERVIEW_RUN_NAME,
    )
    client = mlflow.MlflowClient()
    # Existing stores used the less prominent name "summary". Updating
    # the standard run-name tag is idempotent and preserves the run ID
    # and every artifact already attached to it.
    client.set_tag(run_id, "mlflow.runName", _OVERVIEW_RUN_NAME)
    client.set_tag(
        run_id,
        "mlflow.note.content",
        "Start here for Laconic results. Open Artifacts for cross-task "
        "figures and tasks/<task>/optimizer_results for learning curves "
        "and optimizer comparison tables.",
    )
    return run_id




def _optional_float(value: Any) -> Optional[float]:
    """Parse one evaluations.csv cell; empty/non-finite cells are valid data
    (RunLogger serializes non-finite rewards as empty), not errors."""
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if parsed == parsed else None


def log_optimizer_learning_curve(
    run_id: Optional[str],
    fold: int,
    optimizer_run_dir: Optional[str],
) -> bool:
    """Publish one fold's structured optimizer history as native MLflow data.

    Optimizers already write a common evaluations.csv schema. This keeps
    those files as the source of truth while making reward and incumbent
    curves directly chartable in the MLflow UI. It also attaches the raw CSV,
    metadata, and summary to the exact dataset-level run for deeper debugging.
    """
    if run_id is None or not optimizer_run_dir:
        return False  # --no_mlflow, or the optimizer ran with logging opted out

    run_dir = Path(optimizer_run_dir)
    evaluations_path = run_dir / "evaluations.csv"
    if not evaluations_path.exists():
        return False

    client = mlflow.MlflowClient()
    metric_names = {
        "reward": f"optimizer/fold_{fold}/reward",
        "best": f"optimizer/fold_{fold}/best_so_far",
        "elapsed": f"optimizer/fold_{fold}/objective_seconds",
    }
    existing_steps = {
        name: {metric.step for metric in client.get_metric_history(run_id, key)}
        for name, key in metric_names.items()
    }

    metric_batch = []
    incumbent = float("-inf")
    timestamp_ms = int(time.time() * 1000)
    with evaluations_path.open("r", encoding="utf-8", newline="") as handle:
        for row_index, row in enumerate(csv.DictReader(handle), 1):
            step = int(float(row.get("evaluation") or row_index))
            reward = _optional_float(row.get("reward"))
            best_after = _optional_float(row.get("global_best_after"))
            elapsed = _optional_float(row.get("objective_elapsed_sec"))

            if reward is not None:
                incumbent = max(incumbent, reward)
                if step not in existing_steps["reward"]:
                    metric_batch.append(
                        Metric(metric_names["reward"], reward, timestamp_ms, step)
                    )
            if best_after is not None:
                incumbent = max(incumbent, best_after)
            if incumbent != float("-inf") and step not in existing_steps["best"]:
                metric_batch.append(
                    Metric(metric_names["best"], incumbent, timestamp_ms, step)
                )
            if elapsed is not None and step not in existing_steps["elapsed"]:
                metric_batch.append(
                    Metric(metric_names["elapsed"], elapsed, timestamp_ms, step)
                )

    # Tracking servers cap batch sizes; 750 metrics leaves room below the
    # common 1,000-entity request limit while keeping backfills efficient.
    for start in range(0, len(metric_batch), 750):
        client.log_batch(run_id, metrics=metric_batch[start:start + 750])

    artifact_path = f"optimizer/fold_{fold}"
    for filename in ("evaluations.csv", "summary.json", "metadata.json", "arm_summary.csv"):
        local_path = run_dir / filename
        if local_path.exists():
            client.log_artifact(run_id, str(local_path), artifact_path=artifact_path)
    return True


def sync_optimizer_learning_curves(log_root: str = ".logs") -> Dict[str, int]:
    """Backfill native MLflow curves/artifacts from every structured log on disk."""
    stats = {"metadata_files": 0, "synced": 0, "skipped": 0}
    for metadata_path in Path(log_root).rglob("metadata.json"):
        # Model-cache metadata has no optimizer history and no mlflow_run_id.
        if not (metadata_path.parent / "evaluations.csv").exists():
            continue
        stats["metadata_files"] += 1
        payload = json.loads(metadata_path.read_text(encoding="utf-8"))
        run_metadata = payload["run_metadata"]
        run_id = run_metadata.get("mlflow_run_id")
        fold = run_metadata.get("fold")
        if run_id is None or fold is None:
            # Traces from --no_mlflow runs carry no run id; nothing to sync to.
            stats["skipped"] += 1
            continue
        if log_optimizer_learning_curve(run_id, int(fold), str(metadata_path.parent)):
            stats["synced"] += 1
        else:
            stats["skipped"] += 1
    return stats


def log_figure_to_run(run_id: Optional[str], local_path: str, artifact_path: Optional[str] = None) -> None:
    """Attaches a file already written to disk (a figure, a CSV, ...) as an
    MLflow artifact on `run_id`. Safe to call from any process at any time -
    reopens the run via `run_id=` rather than assuming it's the active run,
    since container runs are typically populated later, by the analysis
    scripts running as a separate process from whichever run created them."""
    if run_id is None:
        return  # --no_mlflow
    if not os.path.exists(local_path):
        raise FileNotFoundError(f"Cannot log missing file to MLflow: {local_path}")
    with mlflow.start_run(run_id=run_id):
        mlflow.log_artifact(local_path, artifact_path=artifact_path)


def start_experiment_run(
    cfg, splitter_name: str, primary_metric: str, n_evaluations: int | None = None
) -> RunContext:
    """Starts one MLflow run for one runner invocation, nested under that
    invocation's task/model container runs (see module docstring); logs
    identity/config params exactly as before."""
    if not cfg.mlflow_enabled:
        return RunContext(run_id=None, git_commit=None)

    task = cfg.task
    experiment_id = _resolve_experiment_id(_DEFAULT_EXPERIMENT)
    model_run_id = get_or_create_container_run(task, cfg.model_name)

    run_name = f"{cfg.dataset}_rs{cfg.random_state}"
    active = mlflow.start_run(experiment_id=experiment_id, run_name=run_name)
    mlflow.set_tag("mlflow.parentRunId", model_run_id)
    mlflow.set_tag("laconic.level", "dataset")
    mlflow.set_tag("laconic.task", task)
    mlflow.set_tag("laconic.model", cfg.model_name)
    mlflow.set_tag("laconic.dataset", cfg.dataset)

    commit, dirty = get_git_info()
    params: Dict[str, Any] = {
        "task": task,
        "dataset": cfg.dataset,
        "compressor": cfg.compressor,
        "optimizer": cfg.optimizer,
        "model": cfg.model_name,
        "random_state": cfg.random_state,
        "alpha": cfg.alpha,
        "primary_metric": primary_metric,
        "splitter_name": splitter_name,
        "n_evaluations": n_evaluations,
        "git_commit": commit,
        "git_dirty": dirty,
    }
    params.update(get_library_versions())
    mlflow.log_params({k: v for k, v in params.items() if v is not None})
    mlflow.set_tag("logs_dir", cfg.logs_dir)

    for name, payload in (
        ("config/model_kwargs.json", cfg.model_kwargs),
        ("config/optimizer_kwargs.json", cfg.optimizer_kwargs),
        ("config/compressor_bounds.json", cfg.compressor_bounds),
        ("config/compressor_space.json", cfg.compressor_space),
    ):
        mlflow.log_dict(payload, name)

    return RunContext(run_id=active.info.run_id, git_commit=commit)


def tag_run_metadata(run_metadata: Dict[str, Any], run_ctx: RunContext) -> Dict[str, Any]:
    """Adds the MLflow run id to the dict already threaded into optimizer logs
    (random.py/bomab.py/run_logger.py all serialize whatever run_metadata they're
    handed into .logs/.../metadata.json), linking that trace back to this run."""
    if run_ctx.run_id is None:
        return run_metadata
    return {**run_metadata, "mlflow_run_id": run_ctx.run_id}


def _row_metrics(row: Dict[str, Any], prefix: str = "") -> Dict[str, float]:
    metrics: Dict[str, float] = {}
    for key, value in row.items():
        if key in _NON_METRIC_KEYS or isinstance(value, bool):
            continue
        try:
            fval = float(value)
        except (TypeError, ValueError):
            continue
        if fval != fval:  # NaN
            continue
        name = _sanitize_metric_name(key)
        if prefix and not name.startswith(prefix):
            name = f"{prefix}{name}"
        metrics[name] = fval
    return metrics


def _log_best_params(best_params: Any, artifact_path: str) -> None:
    if not best_params:
        return
    payload = json.loads(best_params) if isinstance(best_params, str) else dict(best_params)
    mlflow.log_dict(payload, artifact_path)


def log_fold_row(row: Dict[str, Any], fold: int) -> None:
    """Logs one fold's row (the exact dict already built for the CSV) as step-indexed metrics."""
    if mlflow.active_run() is None:
        return
    metrics = _row_metrics(row)
    if metrics:
        mlflow.log_metrics(metrics, step=fold)
    _log_best_params(row.get("best_params"), f"best_params/fold_{fold}.json")


def log_summary_row(mean_row: Dict[str, Any]) -> None:
    """Logs the fold-0 mean row as final (non-stepped) `mean_*` metrics."""
    if mlflow.active_run() is None:
        return
    metrics = _row_metrics(mean_row, prefix="mean_")
    if metrics:
        mlflow.log_metrics(metrics)
    _log_best_params(mean_row.get("best_params"), "best_params/mean.json")


def log_new_rows_artifact(df) -> None:
    """Snapshots exactly the rows this invocation produced (folds + mean row),
    before they're concatenated with any pre-existing results/*.csv history."""
    if mlflow.active_run() is None:
        return
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "results_rows.csv")
        df.to_csv(path, index=False)
        mlflow.log_artifact(path)


def end_run() -> None:
    if mlflow.active_run() is not None:
        # Called from each runner's finally block, where the active exception
        # remains visible through sys.exc_info().
        status = "FAILED" if sys.exc_info()[0] is not None else "FINISHED"
        mlflow.end_run(status=status)
