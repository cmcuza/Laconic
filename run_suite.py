"""Suite-runner: drives a full sweep from one YAML manifest instead of
hand-run commands or per-optimizer shell scripts with hardcoded, commented-
in/out task/model/dataset lists.

Every path a sweep produces already uses the canonical (task, compressor,
optimizer, model, dataset, random_state, alpha, n_evaluations) identity - see
project CLAUDE.md - so there is nothing to hand-rename or hand-move
afterward: adding a (compressor, optimizer) combination to the manifest is
enough for it to show up correctly in results/, .logs/, and MLflow.

After every run_experiments.py invocation in the manifest finishes (unless
--no-visualize or a failure occurred), this also calls analysis/make_figures.py,
which runs every figure script: they write the usual results/figures/*.pdf AND
log PNG copies to MLflow (see analysis/*.py's log_figures_to_mlflow) -
visualization as a side effect of the sweep finishing, not a step you remember
to run by hand after.

Usage:
    python run_suite.py cfg/suites/main_comparison.yaml
    python run_suite.py cfg/suites/main_comparison.yaml --dry-run
    python run_suite.py cfg/suites/main_comparison.yaml --continue-on-error

Manifest schema (see cfg/suites/main_comparison.yaml for a worked example):
    random_states: [32]              # list; sweep more than one seed by adding entries
    alpha: 0.75                       # float or list of floats to sweep
    budget: 100                       # int or list of ints to sweep; required for
                                      # genetic, optional for count-based optimizers
    tasks:
      - analytics: classification
        dataset: ucr_small           # cfg/datasets/<analytics>/<dataset>.yaml
        model: proximity_forest      # omitted -> every model configured for this task
    compressors:
      - compression: tersets_mab
        optimizers: [random, successive_halving, genetic]
      - compression: sz
        optimizers: [bosmp]
    auto_visualize: true              # default true; set false to skip the analysis step

An "experiment" in the sense used across this project is one (budget, alpha,
random_state) triple applied over the whole task x compressor x optimizer x
model matrix below - see CLAUDE.md's "What defines an experiment".
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, Iterator, List

import yaml

# The one list of optimizers whose YAML sizes are percentages of the budget.
# Imported rather than restated: a fourth hand-maintained copy of this set is
# exactly how the original repo's method registries drifted apart.
from run_experiments import _PERCENTAGE_SIZED_OPTIMIZERS

# Every run_experiments.py subprocess gets these forced to "1". Rocket's numba
# kernels are parallel=True and genetic's optimizer runs its own n_workers-sized
# process pool (cfg/optimizer/genetic.yaml), so left uncapped the two multiply
# and oversubscribe the CPU.
#
# These are a CPU-contention cap only - they are NOT what makes the worker pool
# safe. That is `mp_context="forkserver"` in optimizer/genetic.py: capping the
# thread counts here was measured to still leave ~25 threads in the parent (a
# JVM, among others) and a fork-based pool still deadlocked. Don't reintroduce a
# fork-based pool on the strength of these being set. See "Key invariants" in
# CLAUDE.md.
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
    """Cross-product of every (task, compressor+optimizer, random_state, alpha,
    budget) combination the manifest describes, in a stable, repeatable order."""
    # random_states must be stated by the manifest - it's a first-class identity
    # dimension with no other source of truth. Alpha may be omitted to use the
    # optimizer fallback. Budget may be omitted only for optimizers whose YAML
    # contains concrete counts; the percentage-sized optimizers require it
    # explicitly (genetic, preference).
    random_states = manifest["random_states"]
    alphas = _as_list(manifest["alpha"]) if manifest.get("alpha") is not None else [None]
    budgets = _as_list(manifest["budget"]) if manifest.get("budget") is not None else [None]

    for task_entry in manifest["tasks"]:
        analytics = task_entry["analytics"]
        dataset = task_entry["dataset"]
        model = task_entry.get("model")  # omitted -> every model configured for this task
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
                                # Transitional manifest key: rs<N> segment in the
                                # optimizer trace path (see optimizer/run_logger.py).
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


# Source trees stage 2 can collapse: the optimizers that actually search.
# Mirrors scripts/rank_aggregation_selection.py's --source-optimizer choices.
_CONSENSUS_SOURCE_OPTIMIZERS = ("genetic", "adaedge", "bosmp")

# The tree stage 2 writes and analysis/method_registry.py reads. Kept in step with
# scripts/rank_aggregation_selection.py's `rank_agg_b{n_best}_{aggregation}` slug
# for the settings auto_consensus passes (B=1, mean_fitness).
_CONSENSUS_TREE = "rank_agg_b1_mean_fitness"


def auto_consensus(python_bin: str, manifest: Dict[str, Any]) -> bool:
    """Stage 2: collapse each cell's K fold-winners to ONE deployed pipeline.

    This is a production stage, not a post-hoc fixup, and it has to run between
    the sweep and the figures: `analysis/method_registry.py` points every
    plotted method at the `rank_agg_b1_mean_fitness` tree, which only
    `scripts/rank_aggregation_selection.py` writes. Without this step a
    completed sweep produces empty figures.

    It cannot be an optimizer. The vote needs all K folds' pipelines and all K
    validation splits at once, and `maximize()` is called once per fold and
    never sees the others; it is a runner-level concern, not an optimizer one.

    Idempotent: `--skip-done` leaves every (cell, seed) that already has a
    consensus row untouched, so re-running a suite costs nothing, and a stored
    cross-evaluation matrix is reused rather than re-measured.
    """
    print("\n=== Stage 2: cross-fold consensus selection "
          "(results/<task>/<compressor>/rank_agg_b1_mean_fitness/) ===")
    # (compressor, optimizer) PAIRS, not optimizers alone: the results tree keys
    # both independently, `--compressor` defaults to tersets_mab, and a pair that
    # matches nothing is a silent no-op with exit 0 ("No <c>/<o> results matched").
    # Iterating optimizers alone therefore ran adaedge against tersets_mab and
    # skipped the real work without failing.
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

    # Fail-fast, like the rest of the pipeline. The script exits 0 when a pair
    # matches no results, so a typo'd manifest or a renamed tree would otherwise
    # reach the figures as "consensus ran fine" and plot nothing.
    missing = [f"{task}/{compressor}" for task in tasks for compressor, _ in pairs
               if not (Path("results") / task / compressor / _CONSENSUS_TREE).is_dir()]
    if missing:
        print("\nStage 2 produced no consensus tree for: " + ", ".join(sorted(set(missing))))
        ok = False
    return ok


def auto_visualize() -> None:
    """Regenerate every figure in-process: writes
    results/figures/budget_<N>/alpha_<a>/ and logs PNG copies to MLflow.

    Which scripts that means lives in analysis/make_figures.py, not here - a
    new figure script is added there once and every caller (this suite, a
    by-hand rebuild) picks it up. Each script discovers every (budget, alpha)
    experiment present under results/ (and .logs/) and regenerates that
    experiment's figure set, so a sweep at a new budget produces a new sibling
    directory with no manual intervention.
    """
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
        help="Keep running remaining combinations after a failed invocation instead of stopping the suite",
    )
    ap.add_argument("--no-visualize", action="store_true", help="Skip the auto_visualize step even if the manifest enables it")
    ap.add_argument("--no-consensus", action="store_true",
                    help="Skip stage 2 (cross-fold consensus selection). The figures read the "
                         "tree it writes, so they will be stale or empty without it.")
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
