"""Cross-fold consensus selection (the assessment's Design B), computed offline.

`docs/RANK_AGGREGATION_ASSESSMENT.md` is the argument. In one paragraph: the
deployed pipeline is currently the `best_val` argmax over K folds, and that
argmax is dominated by which fold drew the easy validation split (measured:
the reward spread *between* folds is ~15x the spread between a fold's own top
candidates). This script replaces that rule with a vote - it takes the B best
distinct pipelines each fold's search found, scores every one of them on
*every* fold's validation split, and returns the rank-aggregation consensus.

Why a script and not an optimizer
---------------------------------
The vote needs all K folds' pipelines and all K validation splits at once, and
`maximize()` sees one fold. `optimizer/rank_aggregation.py` exists for the
inline case and can only do the weaker within-fold variant (Design B'); this
script does the real thing. It also needs no search re-run at all: the genetic
traces already on disk hold every pipeline the GA evaluated, with full
parameter vectors, so the only new cost is the cross-evaluation.

    new evaluations = B * K * K   per (task, compressor, model, dataset, alpha)

with K = 5 here. B*K of those (candidate, fold) pairs - one per fold per rank
slot - were ALREADY evaluated by the search, and their `reward` is in the
trace; this script re-measures them anyway. The reason is only that the trace
stores the scalarized `reward` and not the (task_metric, avg_cr) pair the
ballots need. Recovering it is possible - `m = (reward - (1-alpha)(1-1/CR)) /
alpha` after recomputing CR from the logged parameters - and cheap, because
compression is 0.4% (classification/Coffee) to 3.7% (clustering/Plane) of an
evaluation, the rest being model inference. That optimisation is NOT
implemented here; it would cut the measurement by B*K of B*K*K, i.e. ~20% at
B=10.

Note the home-fold scores are kept in the vote deliberately, and re-measuring
does not change their value or their bias - it is the same computation on the
same split. Each candidate carries exactly one home-fold ballot out of K, so
the optimism is symmetric across candidates and largely cancels in a rank
aggregation. Dropping the home fold instead would break the vote: each voter
would be missing the B candidates that live on it, and rank aggregation needs
every voter to rank every candidate.

What it writes
--------------
A full results tree under its own optimizer name, as if it had run:

    results/<task>/<compressor>/rank_aggregation/<model>/budget_<N>/alpha_<a>/<ds>.csv

The source run's columns are carried over unchanged wherever they are
unaffected - `baseline_*` (already the repaired deployment-model values),
`baseline_cr`, `n_evaluations`, `primary_metric`. Rewritten: `best_params`
(the consensus pipeline, identical on every fold row - the vote produces ONE
deployable pipeline, not one per fold), `val_*`/`val_avg_cr` (that pipeline
measured on each fold's own validation split, from the cross-evaluation
matrix), and `test_*`/`test_avg_cr` (identical across folds: one pipeline, one
deployment model, one test set). `n_evaluations` deliberately stays the
source's search budget so this lands in the same `budget_<N>` directory as the
run it is compared against - the cross-evaluation cost is recorded separately
as `selection_evaluations` and in the trace, never folded into the budget.

Usage
-----
    ~/anaconda3/envs/laconic_nts/bin/python scripts/rank_aggregation_selection.py --dry-run
    ~/anaconda3/envs/laconic_nts/bin/python scripts/rank_aggregation_selection.py \
        --tasks classification --budgets 100 --n-best 4 --skip-done

Shards cleanly by `--tasks`/`--datasets` the way scripts/fix_deployment_model.py
does: each shard writes a disjoint set of output files and shares only the
read-only model cache.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))
if str(Path(__file__).resolve().parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parent))

# Reused rather than re-derived: these already know how to find a dataset's
# group yaml, its analytics yaml, and a compressor's backend. `Deployment` is
# NOT reused - it is classification/clustering-only (its own TASKS tuple), and
# this script needs all four - so the test side is built directly on the shared
# experiments/deployment_model.py helper the runners themselves call.
from fix_deployment_model import _analytics_cfg, _backend_for, _dataset_group

from experiments.deployment_model import DEPLOYMENT_FOLD, fit_deployment_model

from compression.utils import (compress_and_decompress_batch,
                               compress_and_decompress_batch_cr, BatchCR)
from data.loaders import build_loaders, build_splitter
from evals.metrics import combine_fitness, evaluate_metrics, get_metric
from models.base import _CachedModelMixin
from experiments.objectives import (
    ClassificationObjective,
    ClusteringObjective,
    ForecastingObjective,
    RegressionObjective,
)
from evals.rank_aggregation import (SCORE_AGGREGATIONS, consensus_order,
                                    mean_fitness_scores, rank_positions)

TASKS = ("classification", "clustering", "regression", "forecasting")
# The long-format cross-evaluation matrix: the expensive artifact, written once
# per (source run, B) and re-read by every aggregation afterwards.
MATRIX_NAME = "cross_evaluation.csv"
# Trace directory of the source optimizer, per optimizer name. Any optimizer
# using RunLogger works as a source - the consensus rule cares only that the
# trace lists evaluated pipelines with a reward, not which search produced them.
# That makes "does consensus selection help the BASELINES too?" answerable, which
# decides whether this is a Laconic result or a methodology result.
_SOURCE_LOG_SUBDIR = {"genetic": "_genetic_opt", "adaedge": "_adaedge_opt",
                      "bosmp": "_bosmp_opt"}
# The parameter columns a trace carries, per compressor family. Read from the
# trace header rather than assumed, so a reshaped search space still works.
_NON_PARAM_COLUMNS = {
    "run_id", "evaluation", "timestamp_utc", "phase", "proposal_source", "reward",
    "objective_elapsed_sec", "global_best_before", "global_best_after", "is_new_global_best",
}


def _param_columns(frame: pd.DataFrame) -> List[str]:
    return [c for c in frame.columns
            if c not in _NON_PARAM_COLUMNS and not c.endswith("_model_space")]


def _build_model_for(task: str):
    if task == "classification":
        from models.classification import build_model
    elif task == "clustering":
        from models.clustering import build_model
    elif task == "regression":
        from models.regression import build_model
    else:
        from models.forecasting import build_model
    return build_model


class FoldVoters:
    """The K per-fold objectives a candidate is scored against.

    Each is the *same* objective class the search itself used, constructed the
    same way, so a cross-evaluation measurement is directly comparable to the
    number in the trace - not a re-implementation that could drift from it.
    Models come from the run's own per-fold cache (`fold=1..K`), so nothing is
    refitted."""

    def __init__(self, task: str, dataset: str, model_name: str, random_state: int,
                 compressor: str, alpha: float, primary_metric: str, logs_root: str):
        self.task, self.dataset, self.alpha = task, dataset, float(alpha)
        group = _dataset_group(task, dataset)
        acfg = _analytics_cfg(task, model_name)
        loader = build_loaders(group["loader"]["name"], **(group["loader"].get("kwargs") or {}))
        build_model = _build_model_for(task)
        kwargs = dict(acfg["model"].get("kwargs") or {})
        kwargs["random_state"] = int(random_state)
        spec = get_metric(primary_metric)
        logs_dir = os.path.join(logs_root, task)
        self.backend = _backend_for(compressor)
        self.objectives: List[Any] = []

        if task == "forecasting":
            # The runner injects the horizon from the loader rather than the
            # analytics yaml (experiments/forecasting_runner.py:75), and it is
            # part of the model-cache kwargs hash - omit it and every fold model
            # misses the cache and is refit with a different horizon. It is only
            # populated once the loader has READ the series, so forward_chain
            # has to run first; reading it earlier silently yields None.
            folds = list(loader.forward_chain(dataset))
            kwargs["horizon"] = loader.frc_h
            for fold, (X_tr, X_val, _unused_test) in enumerate(folds, start=1):
                model = build_model(model_name, kwargs)
                model.fit_cached(X_tr, dataset=dataset, random_state=int(random_state),
                                 fold=fold, splitter_name="forward_chain",
                                 train_indices=np.arange(len(X_tr)), out_dir=logs_dir)
                y_val, _ = model.predict(X_val)
                self.objectives.append(ForecastingObjective(
                    model=model, X_val=X_val, y_val=y_val,
                    backend=self.backend, alpha=self.alpha, spec=spec))
            return

        X_train, y_train = loader.load_train(dataset)
        # THE fold splits, reproduced exactly. build_splitter passes no
        # random_state (data/loaders.py), so StratifiedShuffleSplit/KFold draw
        # from the GLOBAL numpy RNG - which run_experiments.py seeds with
        # _set_seeds(random_state) before the runner. Seeding here the same way,
        # immediately before the split, reproduces the original folds; verified
        # against the stored train_idx_sha1 below. Without this the splits are
        # arbitrary, every per-fold model cache misses, models get silently
        # retrained on the WRONG data, and the whole cross-evaluation measures
        # something other than what the search saw.
        np.random.seed(int(random_state))
        splitter = build_splitter(group["split"])
        splitter_name = type(splitter).__name__
        for fold, (tr_idx, val_idx) in enumerate(splitter.split(X_train, y_train), start=1):
            self._verify_split(logs_dir, dataset, model_name, random_state, fold, tr_idx)
            fold_kwargs = dict(kwargs)
            if task == "clustering":
                fold_kwargs["n_clusters"] = len(np.unique(y_train))
            model = build_model(model_name, fold_kwargs)
            fit_args = (X_train[tr_idx],) if task == "clustering" else (X_train[tr_idx], y_train[tr_idx])
            model.fit_cached(*fit_args, dataset=dataset, random_state=int(random_state),
                             fold=fold, splitter_name=splitter_name,
                             train_indices=tr_idx, out_dir=logs_dir)
            common = dict(model=model, X_val=X_train[val_idx], y_val=y_train[val_idx],
                          backend=self.backend, alpha=self.alpha, spec=spec)
            if task == "classification":
                self.objectives.append(ClassificationObjective(classes=model.get_classes(), **common))
            elif task == "clustering":
                self.objectives.append(ClusteringObjective(classes=len(np.unique(y_train)), **common))
            else:
                self.objectives.append(RegressionObjective(**common))

    @staticmethod
    def _verify_split(logs_dir: str, dataset: str, model_name: str, random_state: int,
                      fold: int, train_indices: np.ndarray) -> None:
        """Fail loudly if the reconstructed fold is not the original one.

        The model cache records `train_idx_sha1` for exactly this purpose. A
        mismatch means the split drifted, and `fit_cached` would merely print
        "retraining" and carry on - producing a cross-evaluation against models
        the search never used. That is a wrong answer, not a slow path, so it
        raises. (It measured as a real failure once: an unseeded first draft
        mismatched all 25 clustering folds and silently refit every one.)"""
        meta_path = (Path(_CachedModelMixin._cache_dir(logs_dir, dataset, model_name,
                                                       int(random_state), fold)) / "meta.json")
        if not meta_path.exists():
            return  # nothing cached to check against; fit_cached will populate it
        stored = json.loads(meta_path.read_text()).get("train_idx_sha1")
        actual = hashlib.sha1(np.asarray(train_indices, dtype=np.int64).tobytes()).hexdigest()
        if stored and stored != actual:
            raise SystemExit(
                f"{dataset} fold {fold}: reconstructed train split does not match the one the "
                f"search used (cached {stored[:12]}..., rebuilt {actual[:12]}...). Refusing to "
                "cross-evaluate against a model trained on different data."
            )

    def __len__(self) -> int:
        return len(self.objectives)

    def cross_evaluate(self, candidates: List[Dict[str, float]]) -> np.ndarray:
        """(n_candidates, K, 2) of (task_metric, avg_cr). B*K*K objective calls."""
        out = np.empty((len(candidates), len(self.objectives), 2), dtype=float)
        for c, candidate in enumerate(candidates):
            for k, objective in enumerate(self.objectives):
                out[c, k] = objective._compute_components(candidate)
        return out


def top_candidates(trace_paths: List[Path], n_best: int
                   ) -> Tuple[List[Dict[str, float]], List[Dict[int, int]]]:
    """The `n_best` best DISTINCT pipelines from each fold's trace, pooled, each
    carrying its RANK in every fold's list.

    Distinct by parameter vector, not by reward: elitism re-enters the same
    candidate every generation, so a raw top-B can be one pipeline repeated.
    Duplicates *across* folds are collapsed too - two folds converging on the
    same pipeline is one candidate with one set of ballots, not two - but the
    ranks are kept per fold rather than reduced to a single "home", because
    that is what makes a stored measurement reusable at a SMALLER B: a
    candidate belongs to level B if it ranked <= B in *any* fold, and the
    best_val comparison needs to know which candidate was rank 1 in which fold.
    Without the ranks, a matrix measured at B=10 could not answer B=4 and every
    B would have to be measured from scratch."""
    pooled: Dict[Tuple, Tuple[Dict[str, float], Dict[int, int]]] = {}
    for fold, path in enumerate(trace_paths, start=1):
        frame = pd.read_csv(path)
        columns = _param_columns(frame)
        ranked = (frame.sort_values("reward", ascending=False)
                       .drop_duplicates(subset=columns).head(n_best))
        for rank, (_, row) in enumerate(ranked.iterrows(), start=1):
            candidate = {c: float(row[c]) for c in columns}
            key = tuple(round(candidate[c], 10) for c in columns)
            if key not in pooled:
                pooled[key] = (candidate, {})
            pooled[key][1][fold] = rank
    return [c for c, _ in pooled.values()], [r for _, r in pooled.values()]


def subset_to(n_best: int, candidates: List[Dict[str, float]], ranks: List[Dict[int, int]],
              matrix: np.ndarray
              ) -> Tuple[List[Dict[str, float]], List[Dict[int, int]], np.ndarray]:
    """The level-`n_best` view of a matrix measured at a larger B: keep every
    candidate that ranked <= n_best in at least one fold, and truncate the rank
    maps accordingly. Identical to what measuring at `n_best` would have
    produced - so measure once at the largest B you want, then aggregate at any
    smaller one for free."""
    keep = [i for i, r in enumerate(ranks)
            if any(rank <= n_best for rank in r.values())]
    return ([candidates[i] for i in keep],
            [{f: rk for f, rk in ranks[i].items() if rk <= n_best} for i in keep],
            matrix[keep])


def ballots_for(matrix: np.ndarray, alpha: float, weighting: str
                ) -> Tuple[List[np.ndarray], List[float]]:
    """One fold = one voter. `split` casts two ballots per voter (task metric
    weighted alpha, CR weighted 1-alpha); `scalarized` casts one, ranked by
    combine_fitness. See the assessment's section 6."""
    ballots, weights = [], []
    for k in range(matrix.shape[1]):
        metrics, ratios = matrix[:, k, 0], matrix[:, k, 1]
        if weighting == "split":
            ballots += [rank_positions(metrics), rank_positions(ratios)]
            weights += [alpha, 1.0 - alpha]
        else:
            ballots.append(rank_positions(
                [combine_fitness(m, c, alpha) for m, c in zip(metrics, ratios)]))
            weights.append(1.0)
    return ballots, weights


class TestScorer:
    """The ONE model every test column is scored with, per (task, dataset, model,
    random_state) - `experiments/deployment_model.py`'s rule, for all four tasks.

    Cached across files: every alpha of a dataset shares one deployment model,
    and the model itself comes from the `fold=0` cache the repair scripts and
    runners already populated, so nothing is refitted."""

    def __init__(self, logs_root: str) -> None:
        self._cache: Dict[Any, Dict[str, Any]] = {}
        self._logs_root = logs_root

    def get(self, task: str, dataset: str, model_name: str, random_state: int) -> Dict[str, Any]:
        key = (task, dataset, model_name, int(random_state))
        if key in self._cache:
            return self._cache[key]
        group = _dataset_group(task, dataset)
        acfg = _analytics_cfg(task, model_name)
        loader = build_loaders(group["loader"]["name"], **(group["loader"].get("kwargs") or {}))
        build_model = _build_model_for(task)
        kwargs = dict(acfg["model"].get("kwargs") or {})
        kwargs["random_state"] = int(random_state)
        report_metrics = acfg["metrics"]["extra"]
        logs_dir = os.path.join(self._logs_root, task)

        if task == "forecasting":
            # The fixed test window: forward_chain's LAST fold, with the model
            # fit on that fold's train+val prefix. Same rule as
            # experiments/forecasting_runner.py and fix_forecasting_test_window.py.
            folds = list(loader.forward_chain(dataset))
            kwargs["horizon"] = loader.frc_h   # after forward_chain; see FoldVoters
            X_train_last, X_val_last, X_test = folds[-1]
            model = fit_deployment_model(
                build_model, model_name, kwargs, np.concatenate([X_train_last, X_val_last]),
                dataset=dataset, random_state=int(random_state),
                splitter_name="forward_chain", logs_dir=logs_dir)
            y_test, _ = model.predict(X_test)
            entry = dict(task=task, model=model, X_test=np.asarray(X_test), y_test=y_test,
                         classes=None, needs_score=False, report_metrics=report_metrics)
            entry["baseline"] = evaluate_metrics(report_metrics, y_test, model.predict(X_test)[1],
                                                 None, classes=None, analytics=task)
            self._cache[key] = entry
            return entry

        X_train, y_train = loader.load_train(dataset)
        X_test, y_test = loader.load_test(dataset)
        splitter_name = type(build_splitter(group["split"])).__name__
        if task == "clustering":
            kwargs["n_clusters"] = len(np.unique(y_train))
            model = fit_deployment_model(build_model, model_name, kwargs, X_train,
                                         dataset=dataset, random_state=int(random_state),
                                         splitter_name=splitter_name, logs_dir=logs_dir)
            classes: Any = kwargs["n_clusters"]
            needs_score = False
            baseline = evaluate_metrics(report_metrics, y_test, model.predict(X_test), None,
                                        classes=classes, analytics=task)
        else:
            model = fit_deployment_model(build_model, model_name, kwargs, X_train, y_train,
                                         dataset=dataset, random_state=int(random_state),
                                         splitter_name=splitter_name, logs_dir=logs_dir)
            needs_score = any(get_metric(m).needs_score for m in report_metrics)
            if task == "regression":
                classes = None
                baseline = evaluate_metrics(report_metrics, y_test, model.predict(X_test), None,
                                            classes=None, analytics=task)
            else:
                classes = model.get_classes()
                pred, score = model.predict_both(X_test, need_score=needs_score)
                baseline = evaluate_metrics(report_metrics, y_test, pred, score,
                                            classes=classes, analytics=task)
        entry = dict(task=task, model=model, X_test=X_test, y_test=y_test, classes=classes,
                     needs_score=needs_score, report_metrics=report_metrics, baseline=baseline)
        self._cache[key] = entry
        return entry

    def score(self, entry: Dict[str, Any], backend, params: Dict[str, Any]
              ) -> Tuple[Dict[str, float], BatchCR]:
        """(test metrics, test CR aggregates) for one pipeline on the fixed test set.

        Returns BOTH CR aggregates so the consensus rows carry `test_pooled_cr`
        exactly like the runners do - otherwise re-running stage 2 for a seed
        would blank the backfilled column for the rows it rewrites.
        `params` must already be backend-converted (params_from_vector)."""
        task, model = entry["task"], entry["model"]
        if task == "forecasting":
            reconstruction, cr = compress_and_decompress_batch_cr([entry["X_test"]], backend, params)
            _, y_pred = model.predict(reconstruction[0])
            score = None
        elif task == "regression":
            # X_test is (n_samples, length, channels) - compress_and_decompress_batch
            # only handles the 2D (n_samples, length) shape every other task's X has.
            # RegressionObjective._compute_components (used by FoldVoters, so the
            # cross-evaluation matrix is already correct) and regression_runner.py
            # both compress/decompress per channel and average CR across them;
            # mirrored here so test-scoring uses the same reconstruction.
            X_test = entry["X_test"]
            reconstruction, crs = np.empty_like(X_test), []
            for d in range(X_test.shape[2]):
                rec_dim, cr_dim = compress_and_decompress_batch_cr(X_test[..., d], backend, params)
                reconstruction[..., d] = rec_dim
                crs.append(cr_dim)
            # combine, not mean-of-pooled: see compression/utils.py::BatchCR
            cr = BatchCR.combine(crs)
            y_pred, score = model.predict(reconstruction), None
        else:
            reconstruction, cr = compress_and_decompress_batch_cr(entry["X_test"], backend, params)
            if task == "clustering":
                y_pred, score = model.predict(reconstruction), None
            else:
                y_pred, score = model.predict_both(reconstruction, need_score=entry["needs_score"])
        metrics = evaluate_metrics(entry["report_metrics"], entry["y_test"], y_pred, score,
                                   classes=entry["classes"], analytics=task)
        return metrics, cr


def optimizer_slug(n_best: int, aggregation: str, weighting: str) -> str:
    """The results-tree name this configuration writes under.

    Every knob that changes the ANSWER is in the name, so two aggregations can
    never overwrite each other and a results row always corresponds to a real,
    reconstructible configuration. This is also what lets the analysis scripts
    read it with no changes at all: `analysis/method_registry.py` already maps a
    plot label to a (compressor, optimizer) pair, so "Borda vs Kemeny" is just
    two registry entries - not a new column, a new loader, or a flag threaded
    through make_figures.py."""
    slug = f"rank_agg_b{int(n_best)}_{aggregation}"
    # weighting shapes BALLOTS only; the score-based aggregators never build any,
    # so tagging them with it would imply a distinction that does not exist.
    if aggregation in SCORE_AGGREGATIONS or weighting == "split":
        return slug
    return f"{slug}_scal"


def _matrix_path(logs_root: str, task: str, dataset: str, model: str, compressor: str,
                 alpha: float, budget: int, random_state: int | None = None) -> Path:
    """One matrix per source run - NOT keyed by B or by aggregation.

    It records each candidate's rank in every fold, so the matrix measured at
    the largest B answers every smaller B by filtering (see `subset_to`), and
    every aggregation strategy reads the same file. Keying it by B was the
    first design and it made a B=10 run and a B=4 run redo identical work."""
    base = (Path(logs_root) / task / "_rank_aggregation" / dataset / model / compressor
            / f"alpha_{alpha:g}" / f"budget_{budget}")
    if random_state is not None:
        base = base / f"rs{random_state}"
    return base / MATRIX_NAME


def _write_matrix(path: Path, candidates: List[Dict[str, float]], ranks: List[Dict[int, int]],
                  matrix: np.ndarray, param_columns: List[str], n_best_measured: int) -> None:
    """Long format, one row per (candidate, fold): the reusable measurement.

    `rank_in_fold` is this candidate's position in THAT fold's top list, or
    blank if it was not in it. That column is what makes the file reusable at
    any B <= n_best_measured."""
    rows = []
    for c, (candidate, rank_map) in enumerate(zip(candidates, ranks)):
        for k in range(matrix.shape[1]):
            row = {"candidate": c, "fold": k + 1,
                   "rank_in_fold": rank_map.get(k + 1, ""),
                   "n_best_measured": n_best_measured,
                   "task_metric": float(matrix[c, k, 0]), "avg_cr": float(matrix[c, k, 1])}
            row.update({name: float(candidate[name]) for name in param_columns})
            rows.append(row)
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(path, index=False)


def _read_matrix(path: Path
                 ) -> Tuple[List[Dict[str, float]], List[Dict[int, int]], np.ndarray, int]:
    frame = pd.read_csv(path)
    meta = ("candidate", "fold", "rank_in_fold", "n_best_measured", "task_metric", "avg_cr")
    param_columns = [c for c in frame.columns if c not in meta]
    n_candidates = int(frame["candidate"].max()) + 1
    n_folds = int(frame["fold"].max())
    matrix = np.empty((n_candidates, n_folds, 2), dtype=float)
    candidates: List[Dict[str, float]] = [dict() for _ in range(n_candidates)]
    ranks: List[Dict[int, int]] = [dict() for _ in range(n_candidates)]
    for _, row in frame.iterrows():
        c, k = int(row["candidate"]), int(row["fold"])
        matrix[c, k - 1] = (row["task_metric"], row["avg_cr"])
        candidates[c] = {name: float(row[name]) for name in param_columns}
        if pd.notna(row["rank_in_fold"]) and str(row["rank_in_fold"]).strip() != "":
            ranks[c][k] = int(row["rank_in_fold"])
    return candidates, ranks, matrix, int(frame["n_best_measured"].iloc[0])


def _trace_dir(logs_root: str, task: str, source_optimizer: str, dataset: str, model: str,
               compressor: str, alpha: float, budget: int, random_state: int | None = None) -> Path:
    """The directory holding this run's fold_*/ traces.

    Two layouts coexist during the seed migration: runs launched with
    ``--seeded-trace-dirs`` put an ``rs<N>`` segment before ``fold_<n>`` so that
    two seeds cannot overwrite each other, older runs do not. The seeded layout
    wins when it exists, because that is the only one that can be trusted to
    belong to THIS random_state - the flat one predates the distinction.
    """
    base = (Path(logs_root) / task / _SOURCE_LOG_SUBDIR[source_optimizer] / dataset / model
            / compressor / f"alpha_{alpha:g}" / f"budget_{budget}")
    if random_state is None:
        return base
    seeded = base / f"rs{random_state}"
    if any(seeded.glob("fold_*/evaluations.csv")):
        return seeded
    # Fall back to the flat layout only if it actually belongs to THIS seed.
    # rs32's traces predate the rs<N> segment and sit here; returning them for a
    # different seed would build that seed's consensus from rs32's search.
    if _flat_trace_seed(base) == random_state:
        return base
    return seeded


def _flat_trace_seed(base: Path) -> int | None:
    """The random_state recorded in a flat-layout trace, or None if unreadable."""
    for meta in sorted(base.glob("fold_*/metadata.json")):
        try:
            return int(json.loads(meta.read_text())["run_metadata"]["random_state"])
        except (OSError, ValueError, KeyError, TypeError):
            return None
    return None


# The runners' own merge key (experiments/*_runner.py): a results CSV holds every
# seed, so a seed's rows replace only their own and never the file.
_ROW_KEYS = ("dataset", "compressor", "optimizer", "model", "random_state",
             "alpha", "fold", "n_evaluations")


def _seed_already_written(out_csv: Path, random_state: int) -> bool:
    if not out_csv.exists():
        return False
    try:
        seeds = pd.read_csv(out_csv, usecols=["random_state"])["random_state"]
    except (ValueError, OSError, pd.errors.EmptyDataError, pd.errors.ParserError):
        return False
    return bool((seeds == random_state).any())


def _merge_write(out_csv: Path, pieces: List[pd.DataFrame], column_order) -> None:
    """Write these seeds' rows into out_csv, leaving every other seed intact."""
    new = pd.concat(pieces, ignore_index=True)
    if out_csv.exists():
        prev = pd.read_csv(out_csv)
        keys = [c for c in _ROW_KEYS if c in prev.columns and c in new.columns]
        prev = prev[~prev.set_index(keys).index.isin(new.set_index(keys).index)]
        new = pd.concat([prev, new], ignore_index=True)
    new = new[[c for c in column_order if c in new.columns]]
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    new.to_csv(out_csv, index=False)


def process(path: Path, args, deployment: TestScorer,
            voters_cache: Dict[Any, FoldVoters], task: str | None = None) -> Dict[str, Any]:
    """One source results CSV -> one consensus per random_state it contains.

    A source CSV holds every seed that has been run for that (task, compressor,
    optimizer, model, budget, alpha, dataset), so each seed is an independent
    search and gets its own consensus. Treating the file as one run would mix
    K*S fold rows from S different searches into a single vote.
    """
    frame = pd.read_csv(path)
    folds = frame[frame["fold"] >= 1]
    if folds.empty or "best_params" not in frame.columns:
        return {"status": "skipped", "reason": "no fold rows"}
    first = folds.iloc[0]
    # parts[1] only reads as the task for a path shaped `results/<task>/...`, which
    # is what main() builds. An absolute results root (a test's tmpdir, say) makes it
    # some directory name, and both the output tree and the trace lookup silently
    # move under it. Callers that know the task pass it.
    task = task or path.parts[1]
    slug = optimizer_slug(args.n_best, args.aggregation, args.weighting)
    out_csv = (Path(args.results_root) / task / str(first["compressor"]) / slug
               / str(first["model"]) / f"budget_{int(first['n_evaluations'])}"
               / f"alpha_{float(first['alpha']):g}" / f"{first['dataset']}.csv".lower())

    seeds = sorted(int(v) for v in folds["random_state"].dropna().unique())
    summaries, pieces = [], []
    for seed in seeds:
        summary, rows = _process_seed(frame[frame["random_state"] == seed], path, args,
                                      deployment, voters_cache, seed, out_csv, task)
        summaries.append((seed, summary))
        if rows is not None:
            pieces.append(rows)
    if pieces and not args.dry_run:
        _merge_write(out_csv, pieces, frame.columns)

    written = [(s, r) for s, r in summaries if r["status"] == "written"]
    live = [(s, r) for s, r in summaries if r["status"] in ("written", "dry-run")]
    if not live:
        reason = "; ".join(f"rs{s}: {r.get('reason')}" for s, r in summaries)
        return {"status": "skipped", "reason": reason, "out": out_csv}
    combined = {"status": "written" if written else "dry-run", "out": out_csv,
                "candidates": live[0][1].get("candidates"),
                "evaluations": sum(int(r.get("evaluations") or 0) for _, r in live),
                "elapsed": sum(float(r.get("elapsed") or 0.0) for _, r in live),
                "reused": all(r.get("reused") for _, r in live),
                "changed": any(bool(r.get("changed")) for _, r in live)}
    if len(seeds) > 1:
        combined["reason"] = (
            f"{combined['evaluations']} evals, {combined['elapsed']:.0f}s, "
            + ", ".join(f"rs{s}:" + ("CHANGED" if r.get("changed") else
                                     "kept" if r["status"] == "written" else
                                     "dry" if r["status"] == "dry-run" else
                                     str(r.get("reason")))
                        for s, r in summaries))
    return combined


def _process_seed(frame: pd.DataFrame, path: Path, args, deployment: TestScorer,
                  voters_cache: Dict[Any, FoldVoters], random_state: int,
                  out_csv: Path, task: str | None = None) -> tuple[Dict[str, Any], pd.DataFrame | None]:
    """One (source CSV, random_state) -> that seed's consensus rows.

    Returns (summary, rows) and does NOT write: the caller merges every seed's
    rows into the one output CSV, because a results file holds all seeds (the
    runners merge on a unique key that includes random_state) and overwriting it
    with one seed's frame would erase the others.
    """
    folds = frame[frame["fold"] >= 1]
    if folds.empty or "best_params" not in frame.columns:
        return {"status": "skipped", "reason": "no fold rows"}, None
    first = folds.iloc[0]
    # parts[1] only reads as the task for a path shaped `results/<task>/...`, which
    # is what main() builds. An absolute results root (a test's tmpdir, say) makes it
    # some directory name, and both the output tree and the trace lookup silently
    # move under it. Callers that know the task pass it.
    task = task or path.parts[1]
    dataset, model = str(first["dataset"]), str(first["model"])
    compressor, alpha = str(first["compressor"]), float(first["alpha"])
    budget = int(first["n_evaluations"])
    primary = str(first["primary_metric"]).strip().lower()

    slug = optimizer_slug(args.n_best, args.aggregation, args.weighting)
    if args.skip_done and _seed_already_written(out_csv, random_state):
        return {"status": "skipped", "reason": "already written", "out": out_csv}, None

    trace_dir = _trace_dir(args.logs_dir, task, args.source_optimizer, dataset, model,
                           compressor, alpha, budget, random_state)
    # The matrix follows whatever layout the traces are in, so rs32's existing
    # flat matrices stay where they are and a seeded run gets its own instead of
    # overwriting them - one cross_evaluation.csv per (run, seed), never shared.
    matrix_path = _matrix_path(args.logs_dir, task, dataset, model, compressor, alpha, budget,
                               random_state if trace_dir.name.startswith("rs") else None)
    trace_paths = sorted(trace_dir.glob("fold_*/evaluations.csv"),
                         key=lambda p: int(p.parent.name.split("_")[1]))
    if not trace_paths:
        return {"status": "skipped", "reason": f"no traces under {trace_dir}"}, None

    # A stored matrix serves ANY B up to the one it was measured at.
    measured = 0
    if matrix_path.exists() and not args.remeasure:
        measured = int(pd.read_csv(matrix_path, usecols=["n_best_measured"], nrows=1).iloc[0, 0])
    # A stored matrix is reusable only if it covers BOTH what we aggregate at and
    # what was explicitly asked to be measured. Comparing against --n-best alone
    # made --measure-n-best a no-op whenever a shallower matrix already existed,
    # which is exactly when you want it to deepen one.
    reusable = measured >= max(args.n_best, args.measure_n_best)
    measure_at = max(args.n_best, args.measure_n_best, measured if args.remeasure else 0)

    if args.dry_run:
        candidates, _ = top_candidates(trace_paths, measure_at if not reusable else args.n_best)
        return {"status": "dry-run", "candidates": len(candidates), "reused": reusable,
                "evaluations": 0 if reusable else len(candidates) * len(trace_paths),
                "note": f"stored matrix covers B<={measured}" if measured else "no stored matrix",
                "out": out_csv}, None

    backend = _backend_for(compressor)
    elapsed, n_evals = 0.0, 0
    if reusable:
        # PHASE 2 ONLY. The expensive part is on disk, so a different aggregation
        # - or a smaller B - costs nothing.
        candidates, ranks, matrix, measured = _read_matrix(matrix_path)
    else:
        if args.aggregate_only:
            return {"status": "skipped",
                    "reason": f"stored matrix covers B<={measured}, need B={args.n_best}"}, None
        candidates, ranks = top_candidates(trace_paths, measure_at)
        key = (task, dataset, model, random_state, compressor)
        if key not in voters_cache:
            voters_cache[key] = FoldVoters(task, dataset, model, random_state, compressor,
                                           alpha, primary, args.logs_dir)
        voters = voters_cache[key]
        if len(voters) != len(trace_paths):
            return {"status": "skipped",
                    "reason": f"{len(trace_paths)} traces but {len(voters)} folds rebuilt"}, None
        backend = voters.backend
        started = time.perf_counter()
        matrix = voters.cross_evaluate(candidates)
        elapsed = time.perf_counter() - started
        n_evals = int(matrix.shape[0] * matrix.shape[1])
        _write_matrix(matrix_path, candidates, ranks, matrix,
                      _param_columns(pd.read_csv(trace_paths[0])), measure_at)
        measured = measure_at

    if measured > args.n_best:
        candidates, ranks, matrix = subset_to(args.n_best, candidates, ranks, matrix)

    if args.aggregation in SCORE_AGGREGATIONS:
        scores = mean_fitness_scores(matrix, alpha,
                                     normalize=args.aggregation == "mean_normalized")
        order, prefiltered = list(np.argsort(-scores, kind="mergesort")), False
    else:
        ballots, weights = ballots_for(matrix, alpha, args.weighting)
        order, scores, prefiltered = consensus_order(
            ballots, weights, args.aggregation, kemeny_max_items=args.kemeny_max_items)
    winner_index = order[0]
    winner = candidates[winner_index]

    # What best_val would have deployed: of the K fold WINNERS (rank 1 in their
    # own fold), the one with the highest fitness on its own validation split.
    # Taken from the rank map rather than a stored "home fold", so it is the same
    # answer whether the matrix was measured at this B or a larger one.
    winners = [(c, fold) for c, rank_map in enumerate(ranks)
               for fold, rank in rank_map.items() if rank == 1]
    argmax_index = max(
        winners, key=lambda cf: combine_fitness(matrix[cf[0], cf[1] - 1, 0],
                                                matrix[cf[0], cf[1] - 1, 1], alpha))[0]

    # ---- test side: ONE pipeline, scored by the ONE deployment model ----
    entry = deployment.get(task, dataset, model, random_state)
    # The optimizer works in a raw float vector; the backend needs the converted
    # dict (int rounding, method-index -> method name). Objectives do this inside
    # _compute_components, so the cross-evaluation above never had to - the test
    # side does. Same conversion the runners apply before storing best_params.
    winner_params = backend.params_from_vector(winner)
    test_metrics, test_cr_stats = deployment.score(entry, backend, winner_params)
    test_cr = test_cr_stats.mean_cr

    # The written CSV keeps the source schema EXACTLY - no extra columns. Every
    # analysis script then reads it as an ordinary results file, and provenance
    # (source optimizer, B, aggregation, cross-evaluation cost) lives in the
    # trace JSON beside it instead of widening a schema four figure scripts read.
    updates = {"best_params": json.dumps(winner_params, default=float),
               "optimizer": slug,
               "test_avg_cr": round(float(test_cr), 4),
               "test_pooled_cr": round(float(test_cr_stats.pooled_cr), 4),
               "optimization_time": round(elapsed, 2)}
    for name, value in test_metrics.items():
        updates[f"test_{name}"] = round(float(value), 4)

    out = frame.copy()
    for column, value in updates.items():
        out[column] = value
    # val_* is the ONLY per-fold-varying part left: the consensus pipeline
    # measured on each fold's own validation split, straight from the matrix.
    for position, (_, row) in enumerate(out[out["fold"] >= 1].iterrows()):
        fold_index = int(row["fold"]) - 1
        metric_value, ratio = matrix[winner_index, fold_index]
        out.loc[row.name, "val_avg_cr"] = round(float(ratio), 4)
        out.loc[row.name, f"val_{primary}"] = round(float(metric_value), 4)
    # Other val_* columns described a DIFFERENT pipeline (the source fold's own
    # winner), so they are blanked rather than carried over as if they described
    # this one. Nothing downstream reads them - fold_selection, correlations and
    # validation_reward all use val_<primary> and val_avg_cr - and computing the
    # full metric set for the winner on all K folds would cost K more model
    # predictions to fill columns no figure opens.
    for column in [c for c in out.columns
                   if c.startswith("val_") and c not in ("val_avg_cr", f"val_{primary}")]:
        out[column] = np.nan

    # Only refresh columns the SOURCE already had - clustering's runner writes no
    # acc_delta/acc_impact_%, and inventing them here would make this file a
    # different shape from the run it is compared against.
    baseline_primary = first.get(f"baseline_{primary}")
    if "acc_delta" in out.columns and baseline_primary is not None and float(baseline_primary) != 0:
        ba, ta = float(baseline_primary), float(test_metrics[primary])
        out["acc_delta"] = round(ba - ta, 4)
        out["acc_impact_%"] = round((ba - ta) / ba * 100.0, 2)
    if "baseline_cr" in out.columns:
        out["cr_improvement_x"] = round(float(test_cr) / float(first["baseline_cr"]), 2)

    numeric = out[out["fold"] >= 1].select_dtypes(include=[np.number]).columns
    for column in numeric:
        if column in ("fold", "random_state", "n_evaluations", "selection_evaluations"):
            continue
        out.loc[out["fold"] == 0, column] = round(
            float(pd.to_numeric(out.loc[out["fold"] >= 1, column], errors="coerce").mean()), 2)

    # Column ORDER matters as much as membership: an ordinary results file is
    # what every analysis script expects to read, so this one is byte-comparable
    # in shape to the run it is compared against.
    out = out[[c for c in frame.columns]]

    trace_out = matrix_path.parent / slug
    trace_out.mkdir(parents=True, exist_ok=True)
    (trace_out / "consensus.json").write_text(json.dumps({
        "source": {"optimizer": args.source_optimizer, "trace_dir": str(trace_dir),
                   "results_csv": str(path)},
        "design": "B (cross-fold): every candidate scored on every fold's validation split",
        "optimizer_slug": slug, "matrix": str(matrix_path), "matrix_reused": bool(reusable),
        "n_best_per_fold": args.n_best, "n_candidates": len(candidates),
        "n_folds": int(matrix.shape[1]), "selection_evaluations": n_evals,
        "elapsed_sec": round(elapsed, 2),
        "aggregation": args.aggregation, "weighting": args.weighting,
        "kemeny_prefiltered": bool(prefiltered),
        "consensus_order": [int(i) for i in order],
        "consensus_scores": [float(s) for s in scores],
        "n_best_measured": int(measured),
        "candidate_ranks": [{str(f): int(r) for f, r in rank_map.items()} for rank_map in ranks],
        "winner_vector": {k: float(v) for k, v in winner.items()},
        "winner_params": winner_params,
        "best_val_pick_vector": {k: float(v) for k, v in candidates[argmax_index].items()},
        "selection_changed": bool(winner_index != argmax_index),
    }, indent=2, default=float))

    return {"status": "written", "out": out_csv, "candidates": len(candidates),
            "evaluations": n_evals, "elapsed": elapsed, "reused": reusable,
            "changed": bool(winner_index != argmax_index)}, out


# The settings the in-runner call uses. They are the ones every reported number
# was produced with, and they are pinned here rather than defaulted per caller so
# `run_experiments.py` and a by-hand CLI run cannot drift into different trees.
CONSENSUS_DEFAULTS = dict(n_best=1, measure_n_best=1, aggregation="mean_fitness",
                          weighting="split")


def consensus_for_results_csv(path, *, results_root="results", logs_dir=".logs",
                              source_optimizer="genetic", task=None, skip_done=True,
                              remeasure=False, aggregate_only=False, dry_run=False,
                              kemeny_max_items=8, **overrides):
    """Run stage 2 for ONE source results CSV. The in-process entry point.

    `run_experiments.py` calls this right after a runner writes its CSV, so a
    single `run_experiments.py` invocation produces both the source row and the
    consensus row the figures read - no second command, and no window in which
    `results/` holds a search whose deployed pipeline was never selected.

    Same code path as the CLI (`process()`), so the numbers are identical by
    construction rather than by agreement between two implementations. Seeds
    already present in the output are skipped, so the call costs nothing on a
    re-run and a multi-seed CSV only ever votes on the seed just added.
    """
    settings = dict(CONSENSUS_DEFAULTS)
    settings.update(overrides)
    args = argparse.Namespace(
        results_root=str(results_root), logs_dir=str(logs_dir),
        source_optimizer=source_optimizer, skip_done=skip_done, remeasure=remeasure,
        aggregate_only=aggregate_only, dry_run=dry_run,
        kemeny_max_items=kemeny_max_items, **settings)
    path = Path(path)
    if task is None:                      # `<results_root>/<task>/<compressor>/...`
        task = path.resolve().relative_to(Path(results_root).resolve()).parts[0]
    return process(path, args, TestScorer(str(logs_dir)), {}, task=task)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--tasks", nargs="*", default=list(TASKS), choices=list(TASKS))
    ap.add_argument("--results-root", default="results")
    ap.add_argument("--logs-dir", default=".logs")
    ap.add_argument("--compressor", default="tersets_mab")
    ap.add_argument("--source-optimizer", default="genetic", choices=sorted(_SOURCE_LOG_SUBDIR))
    ap.add_argument("--n-best", type=int, default=4,
                    help="B: best distinct pipelines taken from EACH fold to AGGREGATE over "
                         "(default 4). A stored matrix measured at a larger B is reused by "
                         "filtering, so this is free once the measurement exists.")
    ap.add_argument("--measure-n-best", type=int, default=0,
                    help="measure the matrix at this B even when aggregating at a smaller one. "
                         "Set it to the largest B you will ever want (e.g. 10): the measurement "
                         "costs B*K*K evaluations ONCE and every smaller B is then free.")
    ap.add_argument("--aggregation", default="borda",
                    choices=("borda", "kemeny", "markov", "mean_fitness", "mean_normalized"),
                    help="borda/kemeny/markov are rank-based (magnitude discarded); "
                         "mean_fitness averages the scalarized fitness across folds; "
                         "mean_normalized min-max scales both objectives over the candidate "
                         "pool within each fold before combining. The last two ignore "
                         "--weighting, which is a property of ballots.")
    ap.add_argument("--weighting", default="split", choices=("split", "scalarized"))
    ap.add_argument("--kemeny-max-items", type=int, default=8)
    ap.add_argument("--datasets", default=None, help="comma-separated subset")
    ap.add_argument("--budgets", default=None, help="comma-separated budgets (default: all)")
    ap.add_argument("--alphas", default=None, help="comma-separated alphas (default: all)")
    ap.add_argument("--aggregate-only", action="store_true",
                    help="never measure: aggregate from stored cross_evaluation.csv matrices "
                         "only, and skip any cell that has none. Cheap - this is how you try "
                         "a second aggregation strategy.")
    ap.add_argument("--remeasure", action="store_true",
                    help="recompute the cross-evaluation matrix even if one is stored")
    ap.add_argument("--skip-done", action="store_true",
                    help="skip files whose output already exists - makes a run resumable")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args(argv)

    wanted = {d.strip() for d in args.datasets.split(",")} if args.datasets else None
    budgets = {int(b) for b in args.budgets.split(",")} if args.budgets else None
    alphas = {float(a) for a in args.alphas.split(",")} if args.alphas else None

    files: List[Path] = []
    for task in args.tasks:
        root = Path(args.results_root) / task / args.compressor / args.source_optimizer
        for path in sorted(root.rglob("*.csv")):
            if budgets is not None and not any(
                    p.startswith("budget_") and int(p.split("_", 1)[1]) in budgets for p in path.parts):
                continue
            if alphas is not None and not any(
                    p.startswith("alpha_") and float(p.split("_", 1)[1]) in alphas for p in path.parts):
                continue
            if wanted is not None:
                head = pd.read_csv(path, usecols=["dataset"], nrows=1)
                if head.empty or head["dataset"].iloc[0] not in wanted:
                    continue
            files.append(path)
    if not files:
        raise SystemExit(f"No {args.compressor}/{args.source_optimizer} results matched.")

    print(f"source        : {args.compressor}/{args.source_optimizer}")
    print(f"tasks         : {', '.join(args.tasks)}")
    print(f"files         : {len(files)}")
    slug = optimizer_slug(args.n_best, args.aggregation, args.weighting)
    print(f"B (per fold)  : {args.n_best}   aggregation: {args.aggregation}/{args.weighting}")
    print(f"writes to     : results/<task>/{args.compressor}/{slug}/...")
    print(f"mode          : {'aggregate-only' if args.aggregate_only else 'measure + aggregate'}"
          f"{' (forcing remeasure)' if args.remeasure else ''}")
    print()

    deployment = TestScorer(args.logs_dir)
    voters_cache: Dict[Any, FoldVoters] = {}
    totals = {"written": 0, "skipped": 0, "dry-run": 0, "changed": 0, "evaluations": 0}
    for index, path in enumerate(files, start=1):
        result = process(path, args, deployment, voters_cache)
        totals[result["status"]] = totals.get(result["status"], 0) + 1
        totals["evaluations"] += int(result.get("evaluations", 0) or 0)
        totals["changed"] += int(bool(result.get("changed")))
        detail = result.get("reason") or (
            f"{result.get('candidates')} candidates, {result.get('evaluations')} evals"
            f"{' (matrix reusable)' if result.get('reused') else ''}"
            if args.dry_run else
            f"{result.get('candidates')} candidates, "
            f"{'matrix reused' if result.get('reused') else f'{result.get("evaluations")} evals, {result.get("elapsed", 0):.0f}s'}, "
            f"{'CHANGED' if result.get('changed') else 'kept'} the best_val pick")
        print(f"[{index:3d}/{len(files)}] {path.relative_to(args.results_root)}  "
              f"{result['status']}: {detail}")

    print()
    print(f"written {totals['written']}, skipped {totals['skipped']}, "
          f"{totals['evaluations']} cross-evaluations, "
          f"selection changed the pick in {totals['changed']} file(s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
