"""Shared trace-logging for the arm-structured optimizers.

``random`` and ``bomab`` (and through it ``successive_halving`` /
``adaptive_halving``) write the same optimizer trace: ``metadata.json``,
``events.jsonl``, ``evaluations.{jsonl,csv}``, ``arms.csv``,
``arm_summary.csv``. The buffered-IO helpers that produce those files used to
exist as two byte-identical copies, one per module - 604 duplicated lines,
which is how the CSV field lists drifted apart in the original repo. They live
here once instead.

Everything in this module is the *trace format*, not the search: it reads only
``self.log_flush_every`` and the ``log_run`` dict handed to it, so any optimizer
that builds a ``log_run`` via ``_start_log_run`` can mix it in. What stays in
each optimizer is what actually differs - ``_start_log_run`` (run identity and
directory layout) and ``_log_evaluation`` (which columns that optimizer fills).

``optimizer/run_logger.py::RunLogger`` is the *other*, newer logger, used by
genetic and bosmp for generic (non-arm-structured) search spaces. The two are
not interchangeable: this one emits the per-arm bandit columns.
"""
from __future__ import annotations

import csv
import json
import math
import os
import re
import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Tuple

import numpy as np


def _now_utc_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sanitize_for_path(value: Any) -> str:
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value)).strip("._")
    return safe or "unknown"


def _json_safe(value: Any) -> Any:
    """Convert common scientific-Python values into JSON-native objects."""
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        value = float(value)
    if isinstance(value, np.ndarray):
        return _json_safe(value.tolist())
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


def _arm_label(arm: Tuple[int, int, int], methods: List[str]) -> str:
    return f"L{arm[0]}:{methods[arm[0]]}|C{arm[1]}:{methods[arm[1]]}|I{arm[2]}:{methods[arm[2]]}"


def _arm_components(arm: Tuple[int, int, int], methods: List[str]) -> Dict[str, Any]:
    return {
        "arm": list(arm),
        "arm_label": _arm_label(arm, methods),
        "logical_method_index": int(arm[0]),
        "coefficient_method_index": int(arm[1]),
        "indices_method_index": int(arm[2]),
        "logical_method_name": methods[arm[0]],
        "coefficient_method_name": methods[arm[1]],
        "indices_method_name": methods[arm[2]],
    }

EVALUATION_CSV_FIELDS = [
    "run_id",
    "evaluation",
    "timestamp_utc",
    "phase",
    "proposal_source",
    "arm_label",
    "logical_method_index",
    "coefficient_method_index",
    "indices_method_index",
    "logical_method_name",
    "coefficient_method_name",
    "indices_method_name",
    "logical_method_error",
    "coefficient_method_error",
    "logical_method_error_model_space",
    "coefficient_method_error_model_space",
    "reward",
    "objective_elapsed_sec",
    "arm_observations_before",
    "arm_observations_after",
    "arm_best_before",
    "arm_best_after",
    "global_best_before",
    "global_best_after",
    "is_new_global_best",
    "ucb_score",
    "ucb_exploit",
    "ucb_explore",
    "ucb_rank",
    "fallback_reason",
]


ARM_CSV_FIELDS = [
    "run_id",
    "arm_label",
    "logical_method_index",
    "coefficient_method_index",
    "indices_method_index",
    "logical_method_name",
    "coefficient_method_name",
    "indices_method_name",
    "logical_method_error_min",
    "logical_method_error_max",
    "coefficient_method_error_min",
    "coefficient_method_error_max",
    "logical_method_error_model_min",
    "logical_method_error_model_max",
    "coefficient_method_error_model_min",
    "coefficient_method_error_model_max",
    "num_observations",
    "best_reward",
    "mean_reward",
    "std_reward",
    "last_reward",
]


class ArmTraceLogging:
    """Buffered writers for the arm-structured optimizer trace.

    Mixed into ``RandomOptimizer`` and ``BOMABOptimizer``. Requires the host to
    provide ``self.log_flush_every`` and to pass the ``log_run`` dict built by
    its own ``_start_log_run``; a ``log_run`` of ``None`` is the sanctioned
    no-op path (logging disabled), not a swallowed error.
    """

    def _flush_jsonl_buffer(self, log_run: Dict[str, Any], path_key: str) -> None:
        buffer = log_run["buffers"]["jsonl"].get(path_key, [])
        if not buffer:
            return
        with open(log_run["paths"][path_key], "a", encoding="utf-8") as f:
            f.writelines(buffer)
        log_run["buffers"]["jsonl"][path_key] = []

    def _flush_csv_buffer(self, log_run: Dict[str, Any], path_key: str) -> None:
        csv_state = log_run["buffers"]["csv"].get(path_key)
        if not csv_state or not csv_state.get("rows"):
            return
        path = log_run["paths"][path_key]
        fieldnames = csv_state["fieldnames"]
        rows = csv_state["rows"]
        file_exists = os.path.exists(path)
        with open(path, "a", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
            header_written = bool(log_run["csv_headers_written"].get(path_key, False))
            if not header_written and not file_exists:
                writer.writeheader()
                log_run["csv_headers_written"][path_key] = True
            writer.writerows(rows)
        csv_state["rows"] = []

    def _flush_log_buffers(self, log_run: Dict[str, Any] | None, *, force: bool = False) -> None:
        if log_run is None:
            return
        jsonl_keys = list(log_run["buffers"]["jsonl"].keys())
        csv_keys = list(log_run["buffers"]["csv"].keys())
        for key in jsonl_keys:
            if force or len(log_run["buffers"]["jsonl"].get(key, [])) >= log_run["flush_every"]:
                self._flush_jsonl_buffer(log_run, key)
        for key in csv_keys:
            csv_state = log_run["buffers"]["csv"].get(key, {})
            if force or len(csv_state.get("rows", [])) >= log_run["flush_every"]:
                self._flush_csv_buffer(log_run, key)
    def _write_json(self, log_run: Dict[str, Any] | None, path_key: str, payload: Dict[str, Any]) -> None:
        if log_run is None:
            return
        with open(log_run["paths"][path_key], "w", encoding="utf-8") as f:
            json.dump(_json_safe(payload), f, indent=2, sort_keys=True)
            f.write("\n")
    def _append_jsonl(self, log_run: Dict[str, Any] | None, path_key: str, payload: Dict[str, Any]) -> None:
        if log_run is None:
            return
        log_run["buffers"]["jsonl"].setdefault(path_key, []).append(
            json.dumps(_json_safe(payload), sort_keys=True) + "\n"
        )
        self._flush_log_buffers(log_run)
    def _append_csv(
        self,
        log_run: Dict[str, Any] | None,
        path_key: str,
        row: Dict[str, Any],
        fieldnames: List[str],
    ) -> None:
        if log_run is None:
            return
        csv_buffers = log_run["buffers"]["csv"]
        if path_key not in csv_buffers:
            csv_buffers[path_key] = {"fieldnames": list(fieldnames), "rows": []}
        elif csv_buffers[path_key]["fieldnames"] != list(fieldnames):
            raise ValueError(
                f"CSV fieldnames mismatch for '{path_key}'. "
                f"Expected {csv_buffers[path_key]['fieldnames']}, got {fieldnames}."
            )
        csv_buffers[path_key]["rows"].append({key: _csv_safe(row.get(key)) for key in fieldnames})
        self._flush_log_buffers(log_run)
    def _log_event(self, log_run: Dict[str, Any] | None, payload: Dict[str, Any]) -> None:
        if log_run is None:
            return
        record = {
            "run_id": log_run["run_id"],
            "timestamp_utc": _now_utc_iso(),
            "elapsed_since_start_sec": round(time.perf_counter() - log_run["start_monotonic"], 6),
        }
        record.update(payload)
        self._append_jsonl(log_run, "events", record)

    def _arm_bounds_record(
        self,
        arm: Tuple[int, int, int],
        methods: List[str],
        search_space: Dict[str, Any],
        logical_scale: str,
        coeff_scale: str,
    ) -> Dict[str, Any]:
        arm_raw_bounds = self._arm_error_bounds(arm, methods, search_space)
        arm_model_bounds = self._arm_model_bounds(arm_raw_bounds, logical_scale, coeff_scale)
        record = _arm_components(arm, methods)
        record.update(
            {
                "logical_method_error_min": arm_raw_bounds["logical_method_error"][0],
                "logical_method_error_max": arm_raw_bounds["logical_method_error"][1],
                "coefficient_method_error_min": arm_raw_bounds["coefficient_method_error"][0],
                "coefficient_method_error_max": arm_raw_bounds["coefficient_method_error"][1],
                "logical_method_error_model_min": arm_model_bounds["logical_method_error"][0],
                "logical_method_error_model_max": arm_model_bounds["logical_method_error"][1],
                "coefficient_method_error_model_min": arm_model_bounds["coefficient_method_error"][0],
                "coefficient_method_error_model_max": arm_model_bounds["coefficient_method_error"][1],
            }
        )
        return record

    def _log_arm_inventory(
        self,
        log_run: Dict[str, Any] | None,
        pipeline_arms: List[Tuple[int, int, int]],
        methods: List[str],
        search_space: Dict[str, Any],
        logical_scale: str,
        coeff_scale: str,
    ) -> None:
        for arm in pipeline_arms:
            row = {"run_id": None if log_run is None else log_run["run_id"]}
            row.update(self._arm_bounds_record(arm, methods, search_space, logical_scale, coeff_scale))
            self._append_csv(log_run, "arms_csv", row, ARM_CSV_FIELDS)

    def _write_arm_summary(
        self,
        log_run: Dict[str, Any] | None,
        *,
        history: Dict[Tuple[int, int, int], Dict[str, Any]],
        pipeline_arms: List[Tuple[int, int, int]],
        methods: List[str],
        search_space: Dict[str, Any],
        logical_scale: str,
        coeff_scale: str,
    ) -> None:
        for arm in pipeline_arms:
            rewards = np.asarray(history[arm]["y"], dtype=float)
            row = {"run_id": None if log_run is None else log_run["run_id"]}
            row.update(self._arm_bounds_record(arm, methods, search_space, logical_scale, coeff_scale))
            row.update(
                {
                    "num_observations": int(rewards.size),
                    "best_reward": None if rewards.size == 0 else float(np.max(rewards)),
                    "mean_reward": None if rewards.size == 0 else float(np.mean(rewards)),
                    "std_reward": None if rewards.size == 0 else float(np.std(rewards)),
                    "last_reward": None if rewards.size == 0 else float(rewards[-1]),
                }
            )
            self._append_csv(log_run, "arm_summary_csv", row, ARM_CSV_FIELDS)
