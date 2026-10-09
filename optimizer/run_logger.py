"""Optimizer-agnostic structured run logger.

This is a lightweight counterpart to the logging plumbing baked into
``BOMABOptimizer``/``SuccessiveHalvingOptimizer``/``RandomOptimizer``. Those
loggers assume the TerseTS three-part arm structure
``(logical, coefficient, indices)`` and two named error parameters, which does
not apply to single-parameter baselines (``sz``, ``mixpiece``, ``serfxor``)
driven by generic optimizers such as :class:`optimizer.bosmp.BOSimple`.

:class:`RunLogger` reproduces the same on-disk layout (``metadata.json``,
``events.jsonl``, ``evaluations.jsonl`` / ``evaluations.csv``, ``summary.json``)
over an *arbitrary* float search space, so downstream log-parsing tooling keeps
working without any optimizer-specific assumptions.
"""

from __future__ import annotations

import csv
import json
import math
import os
import re
import time
from datetime import datetime, timezone
from typing import Any, Dict, List


def _now_utc_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sanitize_for_path(value: Any) -> str:
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value)).strip("._")
    return safe or "unknown"


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if hasattr(value, "item") and not isinstance(value, (str, bytes)):
        try:
            value = value.item()
        except Exception:
            pass
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, (int, str, bool)) or value is None:
        return value
    return str(value)


def _csv_safe(value: Any) -> Any:
    value = _json_safe(value)
    if isinstance(value, (dict, list)):
        return json.dumps(value, sort_keys=True)
    return value


class RunLogger:
    """Generic JSONL+CSV+metadata logger for an arbitrary float search space.

    Every method is a no-op when ``log_dir`` is empty (the explicit opt-out for
    programmatic use, e.g. scripts/verify_optimizer_budgets.py). A write failure
    is never swallowed: a trace that silently stops mid-run is worse than a
    failed run, so I/O errors propagate and abort the experiment.
    """

    def __init__(
        self,
        *,
        log_dir: str | None,
        run_metadata: Dict[str, Any] | None,
        optimizer_name: str,
        optimizer_meta: Dict[str, Any],
        search_space: Dict[str, Any],
        space_definition: Dict[str, Any] | None,
        param_names: List[str],
        total_budget: int,
        log_subdir: str,
        verbose: int = 1,
        extra_eval_fields: List[str] | None = None,
    ):
        self.disabled = True
        self.run_id = None
        self.run_dir = None
        # Wall time this logger has spent writing, accumulated across every
        # public call. The execution-time study (docs/EXECUTION_TIME_STUDY_PLAN.md
        # §6.2's T_log) needs the trace-logging tax measured PER CELL and in the
        # SAME pass as the search it is part of - otherwise the only way to get
        # it is a second, `quiet` run of the whole matrix, which doubles the
        # compute for a component that is two perf_counter calls away. Stays 0.0
        # when log_dir is None, since every method below returns immediately.
        self.total_logging_sec = 0.0
        self._param_names = list(param_names)
        # Generic evaluation schema: bookkeeping columns + any optimizer-specific
        # per-evaluation columns + one column per parameter (raw and model space).
        #
        # `extra_eval_fields` exists because the scalarized `reward` is lossy:
        # `alpha * m + (1 - alpha) * (1 - 1/CR)` cannot be inverted back to
        # (m, CR), so a trace that logs only `reward` cannot be re-analysed on
        # the two objectives separately - recovering m requires recompressing
        # every logged pipeline. An optimizer that already computes both (any of
        # the ones taking PreferenceObjective's (task_metric, avg_cr) contract)
        # can declare them here and log them for free. Optimizers that do not
        # pass it keep their existing schema byte for byte.
        self._eval_fields = [
            "run_id",
            "evaluation",
            "timestamp_utc",
            "phase",
            "proposal_source",
            "reward",
            "objective_elapsed_sec",
            "global_best_before",
            "global_best_after",
            "is_new_global_best",
        ]
        self._extra_eval_fields = list(extra_eval_fields or [])
        self._eval_fields += self._extra_eval_fields
        self._eval_fields += list(self._param_names)
        self._eval_fields += [f"{name}_model_space" for name in self._param_names]
        self._header_written = False

        if log_dir is None or str(log_dir).strip() == "":
            return

        # File logging was requested, so the full run identity is required -
        # "unknown_dataset"-style placeholder path segments would silently
        # detach a trace from the run it belongs to.
        if not isinstance(run_metadata, dict):
            raise ValueError("run_metadata is required when log_dir is set.")
        metadata = _json_safe(run_metadata)
        dataset_name = _sanitize_for_path(metadata["dataset"])
        model_name = _sanitize_for_path(metadata["model"])
        compressor_name = _sanitize_for_path(metadata["compressor"])
        alpha_name = f"alpha_{optimizer_meta['alpha']:g}"
        fold_value = metadata["fold"]
        fold_name = f"fold_{_sanitize_for_path(fold_value)}"
        timestamp_for_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
        run_id = f"{dataset_name}_{model_name}_{compressor_name}_{alpha_name}_{fold_name}_rs{optimizer_meta['random_state']}_{timestamp_for_id}"

        subdir = log_subdir or "_opt"
        # Model AND compressor are both part of the path (not just
        # metadata.json) so two models sharing (dataset, optimizer, budget,
        # fold) - e.g. classification's proximity_forest + tsfresh - or two
        # single-parameter baselines (sz/mixpiece/serfxor) tuned by the same
        # optimizer, can never clobber each other's logs. alpha is also a
        # path segment (like budget) so re-running at a different task-vs-
        # compression tradeoff lands in a sibling directory instead of
        # overwriting the previous trace.
        # TRANSITIONAL (2026-09-06): the path has never carried the seed, so two
        # runs differing only by random_state overwrite each other's trace. Opting
        # in adds an rs<N> segment; it is opt-in rather than unconditional because
        # run_suite.py spawns one process per cell, so flipping this while a sweep
        # is in flight would split that sweep's traces across two layouts and every
        # reader globs one shape (scripts/rank_aggregation_selection.py::_trace_dir)
        # - the missing half comes back as "no traces", not an error. Retire it by
        # migrating the existing rs32 traces into rs32/ and making this the default.
        seed_segment = ()
        if metadata.get("seeded_trace_dirs"):
            seed_segment = (f"rs{_sanitize_for_path(optimizer_meta['random_state'])}",)
        run_dir = os.path.join(str(log_dir), subdir, dataset_name, model_name, compressor_name, alpha_name, f"budget_{total_budget}", *seed_segment, fold_name)
        os.makedirs(run_dir, exist_ok=True)
        self.run_id = run_id
        self.run_dir = run_dir
        self._start_monotonic = time.perf_counter()
        self._paths = {
            "metadata": os.path.join(run_dir, "metadata.json"),
            "events": os.path.join(run_dir, "events.jsonl"),
            "evaluations_jsonl": os.path.join(run_dir, "evaluations.jsonl"),
            "evaluations_csv": os.path.join(run_dir, "evaluations.csv"),
            "summary": os.path.join(run_dir, "summary.json"),
        }
        for path in self._paths.values():
            if os.path.exists(path):
                os.remove(path)
        self.disabled = False
        self._write_json(
            "metadata",
            {
                "run_id": run_id,
                "started_at": _now_utc_iso(),
                "cache": {
                    "path_schema": f"{subdir}/dataset_name/model_name/compressor_name/alpha_x/budget_x/fold_x",
                    "dataset": dataset_name,
                    "model": model_name,
                    "compressor": compressor_name,
                    "fold": fold_value,
                    "cache_key": {
                        "task": metadata["task"],
                        "dataset": metadata["dataset"],
                        "fold": metadata["fold"],
                        "model": metadata["model"],
                        "compressor": metadata["compressor"],
                        "optimizer": metadata["optimizer"],
                        "random_state": optimizer_meta["random_state"],
                    },
                },
                "run_metadata": metadata,
                "optimizer": {"name": optimizer_name, "total_budget": total_budget, **optimizer_meta},
                "search_space": search_space,
                "space_definition": space_definition,
                "parameters": self._param_names,
                "files": self._paths,
            },
        )
        self.log_event({"event": "optimization_started", "total_budget": total_budget})
        if verbose > 0:
            print(f"[{optimizer_name}] logging optimization trace to {run_dir}")

    # -- low level writers -------------------------------------------------

    def _write_json(self, path_key: str, payload: Dict[str, Any]) -> None:
        if self.disabled:
            return
        start = time.perf_counter()
        with open(self._paths[path_key], "w", encoding="utf-8") as f:
            json.dump(_json_safe(payload), f, indent=2, sort_keys=True)
            f.write("\n")
        self.total_logging_sec += time.perf_counter() - start

    def _append_jsonl(self, path_key: str, payload: Dict[str, Any]) -> None:
        if self.disabled:
            return
        start = time.perf_counter()
        with open(self._paths[path_key], "a", encoding="utf-8") as f:
            f.write(json.dumps(_json_safe(payload), sort_keys=True) + "\n")
        self.total_logging_sec += time.perf_counter() - start

    # -- public API --------------------------------------------------------

    def log_event(self, payload: Dict[str, Any]) -> None:
        if self.disabled:
            return
        record = {
            "run_id": self.run_id,
            "timestamp_utc": _now_utc_iso(),
            "elapsed_since_start_sec": round(time.perf_counter() - self._start_monotonic, 6),
        }
        record.update(payload)
        self._append_jsonl("events", record)

    def log_evaluation(
        self,
        *,
        evaluation: int,
        params_raw: Dict[str, float],
        params_model: Dict[str, float],
        reward: float,
        elapsed_sec: float,
        global_best_before: float,
        global_best_after: float,
        is_new_global_best: bool,
        phase: str = "unknown",
        proposal_source: str = "unknown",
        extra: Dict[str, Any] | None = None,
    ) -> None:
        if self.disabled:
            return
        row = {
            "run_id": self.run_id,
            "evaluation": evaluation,
            "timestamp_utc": _now_utc_iso(),
            "phase": phase,
            "proposal_source": proposal_source,
            "reward": reward,
            "objective_elapsed_sec": round(elapsed_sec, 6),
            "global_best_before": global_best_before if math.isfinite(global_best_before) else None,
            "global_best_after": global_best_after if math.isfinite(global_best_after) else None,
            "is_new_global_best": is_new_global_best,
        }
        for name in self._extra_eval_fields:
            row[name] = (extra or {}).get(name)
        for name in self._param_names:
            row[name] = params_raw.get(name)
            row[f"{name}_model_space"] = params_model.get(name)
        self._append_jsonl("evaluations_jsonl", row)   # accumulates its own time
        start = time.perf_counter()
        file_exists = os.path.exists(self._paths["evaluations_csv"])
        with open(self._paths["evaluations_csv"], "a", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=self._eval_fields, extrasaction="ignore")
            if not self._header_written and not file_exists:
                writer.writeheader()
            self._header_written = True
            writer.writerow({key: _csv_safe(row.get(key)) for key in self._eval_fields})
        self.total_logging_sec += time.perf_counter() - start

    def write_summary(
        self,
        *,
        evaluations: int,
        total_budget: int,
        best_reward: float,
        best_params: Dict[str, Any] | None,
        status: str,
        error: str | None = None,
    ) -> None:
        if self.disabled:
            return
        self._write_json(
            "summary",
            {
                "run_id": self.run_id,
                "status": status,
                "completed_at": _now_utc_iso(),
                "elapsed_sec": round(time.perf_counter() - self._start_monotonic, 6),
                "evaluations": evaluations,
                "total_budget": total_budget,
                "best_reward": best_reward if math.isfinite(best_reward) else None,
                "best_params": best_params,
                "error": error,
            },
        )
