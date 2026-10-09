"""Optimizer-agnostic structured run logger."""

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
    """Generic JSONL+CSV+metadata logger for an arbitrary float search space."""

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
        self.total_logging_sec = 0.0
        self._param_names = list(param_names)
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
        self._append_jsonl("evaluations_jsonl", row)
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
