"""Run a sweep of run_experiments.py invocations from one YAML manifest."""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, Iterator, List

import yaml

from run_experiments import _PERCENTAGE_SIZED_OPTIMIZERS

_THREAD_CAP_ENV_VARS = (
    "NUMBA_NUM_THREADS",
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
)


def _as_list(value: Any) -> List[Any]:
    return value if isinstance(value, list) else [value]


def iter_combos(manifest: Dict[str, Any]) -> Iterator[Dict[str, Any]]:
    """Yield every (task, compressor, optimizer, random_state, alpha, budget) combination."""
    random_states = manifest["random_states"]
    alphas = _as_list(manifest["alpha"]) if manifest.get("alpha") is not None else [None]
    budgets = _as_list(manifest["budget"]) if manifest.get("budget") is not None else [None]

    for task_entry in manifest["tasks"]:
        analytics = task_entry["analytics"]
        dataset = task_entry["dataset"]
        model = task_entry.get("model")
        for comp_entry in manifest["compressors"]:
            compression = comp_entry["compression"]
            for optimizer in comp_entry["optimizers"]:
                if optimizer in _PERCENTAGE_SIZED_OPTIMIZERS and budgets == [None]:
                    raise ValueError(
                        f"Suite manifests using {optimizer} must define budget because "
                        "its population settings are percentages."
                    )
                for random_state in random_states:
                    for alpha in alphas:
                        for budget in budgets:
                            yield {
                                "analytics": analytics,
                                "compression": compression,
                                "optimizer": optimizer,
                                "dataset": dataset,
                                "model": model,
                                "random_state": random_state,
                                "alpha": alpha,
                                "budget": budget,
                                "seeded_trace_dirs": bool(manifest.get("seeded_trace_dirs", False)),
                            }


def _build_command(combo: Dict[str, Any], python_bin: str) -> List[str]:
    cmd = [
        python_bin, "run_experiments.py",
        "--analytics", combo["analytics"],
        "--compression", combo["compression"],
        "--optimizer", combo["optimizer"],
        "--random_state", str(combo["random_state"]),
        "--dataset", combo["dataset"],
    ]
    if combo["alpha"] is not None:
        cmd += ["--alpha", str(combo["alpha"])]
    if combo["model"]:
        cmd += ["--model", combo["model"]]
    if combo["budget"] is not None:
        cmd += ["--budget", str(combo["budget"])]
    if combo.get("seeded_trace_dirs"):
        cmd += ["--seeded-trace-dirs"]
    return cmd


def run_one(combo: Dict[str, Any], python_bin: str, dry_run: bool) -> bool:
    cmd = _build_command(combo, python_bin)
    print("+", " ".join(cmd))
    if dry_run:
        return True
    env = os.environ.copy()
    env.update({var: "1" for var in _THREAD_CAP_ENV_VARS})
    return subprocess.run(cmd, env=env).returncode == 0


_CONSENSUS_SOURCE_OPTIMIZERS = ("genetic", "adaedge", "bosmp")

_CONSENSUS_TREE = "rank_agg_b1_mean_fitness"


def auto_consensus(python_bin: str, manifest: Dict[str, Any]) -> bool:
    """Stage 2: collapse each cell's K fold-winners to ONE deployed pipeline."""
    print("\n=== Stage 2: cross-fold consensus selection "
          "(results/<task>/<compressor>/rank_agg_b1_mean_fitness/) ===")
    pairs = sorted({(c["compression"], optimizer)
                    for c in manifest.get("compressors", [])
                    for optimizer in c.get("optimizers", [])
                    if optimizer in _CONSENSUS_SOURCE_OPTIMIZERS})
    tasks = sorted({t["analytics"] for t in manifest.get("tasks", [])})
    ok = True
    for compressor, source_optimizer in pairs:
        cmd = [python_bin, "scripts/rank_aggregation_selection.py",
               "--compressor", compressor,
               "--source-optimizer", source_optimizer,
               "--n-best", "1", "--measure-n-best", "1",
               "--aggregation", "mean_fitness", "--skip-done"]
        if tasks:
            cmd += ["--tasks", *tasks]
        print("   ", " ".join(cmd))
        ok &= subprocess.run(cmd).returncode == 0

    missing = [f"{task}/{compressor}" for task in tasks for compressor, _ in pairs
               if not (Path("results") / task / compressor / _CONSENSUS_TREE).is_dir()]
    if missing:
        print("\nStage 2 produced no consensus tree for: " + ", ".join(sorted(set(missing))))
        ok = False
    return ok


def auto_visualize() -> None:
    """Regenerate figures (requires the analysis package)."""
    print("\n=== Generating figures (results/figures/budget_<N>/alpha_<a>/ + MLflow) ===")
    from analysis.make_figures import run_figure_scripts

    run_figure_scripts()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("manifest", type=Path, help="Path to a cfg/suites/<name>.yaml manifest")
    ap.add_argument("--python-bin", default=sys.executable, help="Python interpreter to invoke run_experiments.py with")
    ap.add_argument("--dry-run", action="store_true", help="Print the run_experiments.py commands without executing them")
    ap.add_argument(
        "--continue-on-error", action="store_true",
        help="Keep going after a failed invocation.",
    )
    ap.add_argument("--no-visualize", action="store_true", help="Skip the auto_visualize step even if the manifest enables it")
    ap.add_argument("--no-consensus", action="store_true",
                    help="Skip stage 2 (cross-fold consensus selection).")
    args = ap.parse_args()

    manifest = yaml.safe_load(args.manifest.read_text())
    combos = list(iter_combos(manifest))
    print(f"Suite '{args.manifest.name}': {len(combos)} run_experiments.py invocation(s) queued.\n")

    failures: List[Dict[str, Any]] = []
    for i, combo in enumerate(combos, 1):
        print(f"[{i}/{len(combos)}] ", end="")
        if not run_one(combo, args.python_bin, args.dry_run):
            failures.append(combo)
            print(f"  FAILED: {combo}")
            if not args.continue_on_error:
                break

    if failures:
        print(f"\n{len(failures)} invocation(s) failed:")
        for f in failures:
            print(" ", f)

    should_finish = not args.dry_run and not failures
    if should_finish and not args.no_consensus and manifest.get("auto_consensus", True):
        if not auto_consensus(args.python_bin, manifest):
            print("\nStage 2 failed; skipping figures (they would read a stale consensus tree).")
            return 1

    should_visualize = (
        should_finish
        and not args.no_visualize
        and manifest.get("auto_visualize", True)
    )
    if should_visualize:
        auto_visualize()
    elif failures and not args.dry_run:
        print("\nSkipping auto_visualize because at least one invocation failed "
              "(pass --continue-on-error to run the rest of the sweep anyway).")

    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
