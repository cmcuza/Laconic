"""One clustering cell, end to end, from `run_experiments.py` to a deployable pipeline.

This is the test that says the repo is self-contained: ONE command has to produce
everything a figure reads. A result takes two stages (see "The two stages of a
result" in CLAUDE.md) - the per-fold search, and the cross-fold consensus that
collapses K fold-winners to the single pipeline that would actually be deployed -
and `analysis/method_registry.py` points every plotted method at the second one.
Until stage 2 was wired into `run_experiments.py`, a finished search left
`results/` with no deployable pipeline and every figure came out empty. Nothing
below mocks either stage: it runs the real optimizer over the real dataset and
asserts on the CSVs that land on disk.

Scope is deliberately one dataset (Coffee), one seed (32), budget 50. That is the
smallest thing that still exercises every seam - five folds, so the vote has
something to vote over; a real compressor, so the CR columns are real - and it
keeps the test to minutes rather than hours. Coffee is the cheapest clustering
dataset (28 train series of length 286) and KShape is the cheapest model.

The run writes into a tmpdir, never the repo's `results/`. Datasets are selected
by GROUP yaml, not by name, so the one-dataset form is the documented trick from
`.claude/skills/run-experiment/SKILL.md`: copy `cfg/` to a scratch dir and trim
the group's `datasets:` list, never edit the repo's own cfg.
"""
from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest
import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent

TASK = "clustering"
COMPRESSOR = "tersets_mab"
OPTIMIZER = "genetic"
MODEL_FLAG, MODEL_DIR = "kshape", "KShape"
DATASET, GROUP = "Coffee", "ucr_small"
SEED, ALPHA, BUDGET = 32, 0.75, 50
METRIC = "test_ari"

CONSENSUS_TREE = "rank_agg_b1_mean_fitness"
# Five folds plus the fold-0 mean row the runners append last.
EXPECTED_ROWS = 6
RUN_TIMEOUT_SEC = 3600


def _scratch_cfg(tmp_path: Path) -> Path:
    """A copy of `cfg/` whose clustering group holds exactly one dataset."""
    cfg = tmp_path / "cfg"
    shutil.copytree(PROJECT_ROOT / "cfg", cfg)
    group = cfg / "datasets" / TASK / f"{GROUP}.yaml"
    spec = yaml.safe_load(group.read_text())
    spec["datasets"] = [DATASET]
    group.write_text(yaml.safe_dump(spec, sort_keys=False))
    return cfg


def _run(tmp_path: Path, out_dir: Path, logs: Path) -> subprocess.CompletedProcess:
    cmd = [
        sys.executable, "run_experiments.py",
        "--analytics", TASK, "--compression", COMPRESSOR, "--optimizer", OPTIMIZER,
        "--dataset", GROUP, "--model", MODEL_FLAG,
        "--random_state", str(SEED), "--alpha", str(ALPHA), "--budget", str(BUDGET),
        "--cfg_root", str(_scratch_cfg(tmp_path)),
        "--out_dir", str(out_dir), "--logs", str(logs),
        "--no_mlflow",          # a true opt-out; keeps the test off the tracking store
    ]
    return subprocess.run(cmd, cwd=PROJECT_ROOT, capture_output=True, text=True,
                          timeout=RUN_TIMEOUT_SEC)


def _source_csv(out_dir: Path) -> Path:
    return (out_dir / TASK / COMPRESSOR / OPTIMIZER / MODEL_DIR
            / f"budget_{BUDGET}" / f"alpha_{ALPHA:g}" / f"{DATASET.lower()}.csv")


def _consensus_csv(out_dir: Path) -> Path:
    return (out_dir / TASK / COMPRESSOR / CONSENSUS_TREE / MODEL_DIR
            / f"budget_{BUDGET}" / f"alpha_{ALPHA:g}" / f"{DATASET.lower()}.csv")


@pytest.fixture(scope="module")
def completed(tmp_path_factory) -> dict:
    """Run the cell once; every assertion below reads this one run."""
    tmp_path = tmp_path_factory.mktemp("e2e")
    out_dir, logs = tmp_path / "results", tmp_path / "logs"
    proc = _run(tmp_path, out_dir, logs)
    if proc.returncode != 0:
        pytest.fail(f"run_experiments.py exited {proc.returncode}\n"
                    f"--- stdout tail ---\n{proc.stdout[-4000:]}\n"
                    f"--- stderr tail ---\n{proc.stderr[-4000:]}")
    return {"tmp_path": tmp_path, "out_dir": out_dir, "logs": logs, "proc": proc}


def test_search_writes_one_row_per_fold(completed):
    """Stage 1: the search produced a complete cell."""
    frame = pd.read_csv(_source_csv(completed["out_dir"]))
    assert len(frame) == EXPECTED_ROWS, f"expected {EXPECTED_ROWS} rows, got {len(frame)}"
    assert set(frame["fold"]) == set(range(EXPECTED_ROWS))
    assert set(frame["random_state"]) == {SEED}
    # The budget is a tracked dimension and must be spent exactly, not approximately.
    assert set(frame["n_evaluations"]) == {BUDGET}


def test_consensus_tree_exists(completed):
    """Stage 2 ran as part of the SAME command - the self-containment claim."""
    path = _consensus_csv(completed["out_dir"])
    assert path.is_file(), (
        f"{path} is missing: the search finished but no deployable pipeline was "
        "selected, so every figure would be empty. Stage 2 did not run.")


def test_consensus_deploys_exactly_one_pipeline(completed):
    """The vote returns ONE pipeline, so every fold row carries identical test_*.

    This is the invariant that distinguishes a real consensus from a per-fold
    rewrite: if these vary, stage 2 did not collapse the folds and `best_val`
    would silently disagree with the consensus value on the same cell.
    """
    folds = pd.read_csv(_consensus_csv(completed["out_dir"])).query("fold >= 1")
    assert not folds.empty
    assert folds["best_params"].nunique() == 1, "more than one deployed pipeline"
    for column in (METRIC, "test_avg_cr", "test_pooled_cr"):
        assert folds[column].nunique() == 1, f"{column} varies across folds"
    # ... while the per-fold validation columns still differ: each fold searched
    # its own split, and only the test side is shared.
    assert folds["val_avg_cr"].nunique() > 1 or len(folds) == 1


def test_consensus_pipeline_is_a_real_tersets_pipeline(completed):
    folds = pd.read_csv(_consensus_csv(completed["out_dir"])).query("fold >= 1")
    params = json.loads(folds["best_params"].iloc[0])
    for stage in ("logical_method", "coefficient_method", "indices_method"):
        assert isinstance(params.get(stage), str) and params[stage], f"missing {stage}"
    for bound in ("logical_method_error", "coefficient_method_error"):
        assert float(params[bound]) > 0.0


def test_both_cr_aggregations_are_present_and_ordered(completed):
    """`test_pooled_cr` is a length-weighted harmonic mean of the same ratios.

    A harmonic mean can never exceed the arithmetic one, so this catches a column
    filled from a different compression - the check the removed backfill script
    used as its validation gate, kept here where it costs nothing.
    """
    for path in (_source_csv(completed["out_dir"]), _consensus_csv(completed["out_dir"])):
        folds = pd.read_csv(path).query("fold >= 1")
        for column in ("test_avg_cr", "test_pooled_cr", "baseline_pooled_cr"):
            assert column in folds.columns, f"{column} missing from {path}"
            assert folds[column].notna().all(), f"{column} has gaps in {path}"
        assert (folds["test_pooled_cr"] <= folds["test_avg_cr"] + 1e-6).all(), (
            f"pooled CR exceeds the arithmetic mean in {path}, which is impossible "
            "for a length-weighted harmonic mean of the same per-series ratios")
        assert (folds["test_avg_cr"] > 1.0).all(), "compression ratio below 1"


def test_metric_is_in_range(completed):
    """ARI is folded onto [0, 1] by `evals/metrics.py::_ari`, not left at [-1, 1]."""
    folds = pd.read_csv(_consensus_csv(completed["out_dir"])).query("fold >= 1")
    for column in (METRIC, f"baseline_{METRIC.removeprefix('test_')}"):
        assert folds[column].between(0.0, 1.0).all(), f"{column} outside [0, 1]"


def test_rerunning_reproduces_the_same_numbers(completed, tmp_path_factory):
    """Same seed, same command, same answer - into a clean results/ and logs/.

    The seed is the whole basis for comparing methods, so a run that does not
    reproduce makes every cross-method claim unfalsifiable. The model cache is
    also fresh here, so this covers the cache path as well as the search.
    """
    second = tmp_path_factory.mktemp("e2e_repeat")
    proc = _run(second, second / "results", second / "logs")
    assert proc.returncode == 0, proc.stderr[-4000:]

    for path_of in (_source_csv, _consensus_csv):
        first_frame = pd.read_csv(path_of(completed["out_dir"])).query("fold >= 1")
        repeat_frame = pd.read_csv(path_of(second / "results")).query("fold >= 1")
        shared = [c for c in (METRIC, "test_avg_cr", "test_pooled_cr", "best_params")
                  if c in first_frame.columns]
        pd.testing.assert_frame_equal(
            first_frame[shared].reset_index(drop=True),
            repeat_frame[shared].reset_index(drop=True),
            check_exact=False, rtol=0.0, atol=0.0,
            obj=f"{path_of.__name__} differs between two identical runs")
