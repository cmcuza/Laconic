"""Preference-based weight elicitation for the GA optimizer -- minimal POC.

Implements the parts of the design that every surviving document version
agrees on (`docs/PREFERENTIAL_OPTIMIZATION_V2.md`,
`docs/PREFERENTIAL_OPTIMIZATION_V2_ASSESSMENT.md`,
`docs/PREFERENTIAL_OPTIMIZATION_IMPLEMENTATION_PLAN.md`,
`docs/PREFERENTIAL_OPTIMIZATION_DESCRIPTION.md`):

- Fit the weight posterior in RAW (task_metric, avg_cr) coordinates. Ideal/nadir
  normalization is used ONLY to pick which pair to ask about, never inside the
  likelihood -- fitting in normalized coordinates was v1's bug: the target
  drifts as the front grows.
- `B = 1 - 1/avg_cr`, unchanged.
- Posterior is recomputed from scratch on every update -- never incrementally.
- `beta` is a fixed constant, calibrated offline (`calibrate_beta`), never
  recalibrated during a live run.
- Comparisons are never stopped early on "looks stable"; the schedule runs to
  `max_comparisons` or the generations run out, whichever first. Stability is
  reported only as a final diagnostic.

Four decisions below are NOT in any version of the design docs -- the first
two come from your own offline experiments, the last two from a measured
clustering run; none are in the doc trail, and all are recorded here for the
same reason everything else in this file cites its source:

- **Prior: Beta(prior_alpha=2, prior_beta=2), not uniform.** Every doc version
  uses a flat prior over `w`. Beta(2,2) is unimodal and vanishes at the
  boundaries, discouraging the posterior from resting on an extreme corner
  (w~0 or w~1, i.e. "only compression matters" / "only accuracy matters")
  without evidence for it. Beta(1,1) is exactly uniform, so the old behavior
  is still reachable as a special case, not lost.
- **Acquisition: expected information gain (EIG), not closest-tau-to-median.**
  Every doc version ranks candidate pairs by how close their threshold `tau`
  is to the current posterior median -- a proxy for "this pair is near the
  decision boundary." EIG is the thing that proxy is standing in for: the
  expected reduction in posterior entropy over `w` from asking about a given
  pair, computed in closed form on the grid (Bayesian experimental design; no
  simulation needed, since the grid already discretizes the one thing being
  integrated over). Measured offline to beat both the tau heuristic and
  Fisher information.
- **A pair is never asked twice in a run** (`select_query`'s `already_asked`,
  which carries the rationale and the measurement).
- **A round may ask more than once** (`comparisons_per_round`). One ask per
  generation made `max_comparisons` unreachable whenever it exceeded `gens`;
  the posterior is recomputed between asks, so the second query of a round is
  selected against what the first taught, on the same unchanged archive.

Originally built to touch no other file in this repo; framework wiring
(`docs/PREFERENCE_OPTIMIZER.md`) has since changed two of the choices that
forced, noted inline below where they no longer hold:

- `RunLogger`'s CSV schema (`evaluations.csv`) was fixed when this was written,
  so it could not gain `task_metric`/`avg_cr` columns; every preference-specific
  event (comparison rounds, posterior state, final diagnostics) goes through
  `RunLogger.log_event()` instead, which accepts an arbitrary dict. That is
  still the right home for the event stream, but the first half no longer
  holds: `RunLogger(extra_eval_fields=[...])` now adds per-evaluation columns
  (the two signals `reward` scalarizes away irreversibly).
  This optimizer could adopt it -- it has both components in hand -- and has
  not, only because nothing has needed its trace re-analysed on the two
  objectives separately yet.
- This optimizer's objective contract is `objective(candidate) ->
  (task_metric, avg_cr)`, NOT the rest of the repo's `objective(candidate) ->
  float` -- the real `experiments/objectives.py` classes collapse that pair
  into a scalar for every other optimizer. `experiments.objectives.
  PreferenceObjective` now supplies the real-backend wrapper (delegates to
  each task's own `_compute_components`, added alongside it); the `__main__`
  block below still runs against a synthetic two-objective toy problem for a
  no-dataset smoke test.
- `genetic.py` is not touched, so this class does not subclass its `maximize`
  (the batch-evaluation seam the staged plan adds to `genetic.py` does not
  exist here). It subclasses `GeneticOptimizer` only for its already-correct,
  reusable helpers (`_tournament`, `_sample_unique_population`,
  `_propose_elite_refinement`, `_propose_unique_child` -- single-underscore
  names, which is a Python naming convention for "internal API", not access
  control; a subclass can call them exactly as written) and writes its own
  `maximize`.
- A real framework run now passes both an `oracle` (built by
  `run_experiments.py` from `cfg/oracle/*.yaml` via `optimizer.oracle.
  build_oracle`) and an `alpha` -- the ground-truth target that oracle was
  built from, used here only to give `RunLogger` a real `alpha_<target>/`
  directory path in place of the placeholder below. Without a real `alpha`
  (the disconnected toy/harness path, which passes `oracle` directly and no
  `alpha`), `RunLogger` still needs *some* numeric value for its path
  (`alpha_{alpha:g}`); the inherited `self.alpha` stays hardcoded to
  `float("nan")` for that case, producing `alpha_nan/` -- an honest
  placeholder ("no single fixed alpha") rather than a misleading number.

A third decision not in the doc trail, reversing this file's own earlier
position: **scoring and the final pick use the posterior MEAN; `.median()`
is reported only** (`WeightPosterior.mean`'s docstring has the one-line
proof of why). Every doc version uses the median for both; an earlier
revision here followed them without examining why, which didn't survive
examination. `IMPLEMENTATION_NOTES.md`'s A8 gets this right.

One place this file still does NOT follow the newest document
(`docs/PREFERENTIAL_OPTIMIZATION_IMPLEMENTATION_NOTES.md`), because it is
itself unmeasured and no ablation has been run against it here:

- That document proposes a second, independent query filter (a `|slope|`
  floor on top of the tradeoff-sign filter). The single min-separation filter
  below is the one the `_V2_ASSESSMENT.md` ablation actually measured (a null
  result -- free to keep). The second filter is not implemented here.

Not implemented at all (out of scope for a single-file POC): the
`beta_oracle`-calibration machinery from `IMPLEMENTATION_NOTES.md` (that is
harness/oracle design, not the optimizer), multiprocessing correctness beyond
what `genetic.py` already established (the pool below reuses that exact
pattern), and any of Stage 3's result-CSV/MLflow columns.
"""

from __future__ import annotations

import math
import time
import multiprocessing
import concurrent.futures as cf
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from .space import Space
from .run_logger import RunLogger
from .genetic import GeneticOptimizer
from evals.metrics import combine_fitness


# =====================================================================
# Part A -- pure preference math. No repo imports beyond evals.metrics'
# combine_fitness (reused, not reimplemented, per the implementation plan's
# explicit instruction). Unit-testable in isolation, like optimizer/space.py.
# =====================================================================


def compression_term(avg_cr):
    """B = 1 - 1/avg_cr. Works on scalars and numpy arrays alike."""
    return 1.0 - 1.0 / avg_cr


def _entropy(density: np.ndarray) -> float:
    """Shannon entropy of a discrete density, with the `0 * log(0) := 0`
    convention (grid cells with zero mass contribute nothing)."""
    positive = density > 0
    return float(-np.sum(density[positive] * np.log(density[positive])))


@dataclass(frozen=True)
class ComparisonRecord:
    """One answered pairwise query, stored in RAW objective coordinates.

    Raw storage is the point: the posterior is rebuilt from scratch every
    round, and raw vectors are the only representation that does not move
    when the front grows. `avg_cr_x`/`avg_cr_y` are the raw compression
    ratio, not `1 - 1/cr` -- `B` is derived at the point of use so the
    objective form is revisitable without re-collecting any comparison.
    """

    task_metric_x: float
    avg_cr_x: float
    task_metric_y: float
    avg_cr_y: float
    prefers_x: bool


def _non_dominated_indices(task_metric: np.ndarray, compression: np.ndarray) -> np.ndarray:
    """Indices of the non-dominated set, maximizing both arguments.

    O(n^2); fine at the archive sizes a GA at budget <= 500 ever produces.
    """
    task_metric = np.asarray(task_metric, dtype=float)
    compression = np.asarray(compression, dtype=float)
    n = task_metric.shape[0]
    dominated = np.zeros(n, dtype=bool)
    for i in range(n):
        at_least_as_good = (task_metric >= task_metric[i]) & (compression >= compression[i])
        strictly_better_somewhere = (task_metric > task_metric[i]) | (compression > compression[i])
        dominates_i = at_least_as_good & strictly_better_somewhere
        dominates_i[i] = False
        if np.any(dominates_i):
            dominated[i] = True
    return np.where(~dominated)[0]


def normalize_front(
    task_metric,
    avg_cr,
    prev_ideal_nadir: Optional[Tuple[Tuple[float, float], Tuple[float, float]]] = None,
    eps: float = 1e-9,
):
    """Ideal/nadir min-max normalization, for QUERY SELECTION ONLY -- never
    for the posterior likelihood (that is fit in raw coordinates; see the
    module docstring's D1/D2).

    Ideal and nadir are computed over the non-dominated subset only -- the
    worst metric ever evaluated typically comes from a dominated individual
    and would stretch the scale over territory no reasonable solution
    occupies. If the front has collapsed on an objective this round
    (span < eps), falls back to the previous round's normalization rather
    than dividing by ~0.

    Returns `(normalized_task_metric, normalized_compression,
    non_dominated_indices, ideal, nadir)`, where the normalized arrays cover
    the FULL input (not just the non-dominated subset) so callers can slice
    as needed -- or `None` when the front has collapsed and there is no
    previous round to fall back to.

    **`None` means "no query available this round", not an error.** Two
    distinct non-dominated points must each be strictly better on one
    objective, so a collapsed span means the front is one point repeated,
    which admits no tradeoff pair under *any* positive scaling -- exactly the
    state `select_query` reports by returning `None`. The fallback branch
    above lands in the same place (it rescales a front that still holds one
    distinct point, so `tradeoff_pairs` finds no strict inequality either),
    so both paths agree; raising here instead would have aborted a whole run
    over a round that had nothing to ask. It is a routine early-search state:
    one candidate that happens to be best on both objectives dominates the
    archive, which is common whenever the task metric is coarse (clustering
    ARI on a small dataset takes few distinct values).
    """
    task_metric = np.asarray(task_metric, dtype=float)
    avg_cr = np.asarray(avg_cr, dtype=float)
    compression = compression_term(avg_cr)

    non_dominated_idx = _non_dominated_indices(task_metric, compression)
    front_metric, front_compression = task_metric[non_dominated_idx], compression[non_dominated_idx]
    ideal = (float(front_metric.max()), float(front_compression.max()))
    nadir = (float(front_metric.min()), float(front_compression.min()))
    metric_span = ideal[0] - nadir[0]
    compression_span = ideal[1] - nadir[1]

    if metric_span < eps or compression_span < eps:
        if prev_ideal_nadir is None:
            return None
        ideal, nadir = prev_ideal_nadir
        metric_span = ideal[0] - nadir[0]
        compression_span = ideal[1] - nadir[1]

    normalized_task_metric = (task_metric - nadir[0]) / metric_span
    normalized_compression = (compression - nadir[1]) / compression_span
    return normalized_task_metric, normalized_compression, non_dominated_idx, ideal, nadir


def pair_key(task_metric_x, avg_cr_x, task_metric_y, avg_cr_y):
    """Order-independent identity of a query, in RAW components.

    Keyed by value rather than archive index so that two candidates which
    evaluated to the same `(task_metric, avg_cr)` count as the same question --
    which they are: the oracle sees only the components.
    """
    return tuple(sorted(((float(task_metric_x), float(avg_cr_x)),
                         (float(task_metric_y), float(avg_cr_y)))))


def _tau_and_slope(metric_gap: float, compression_gap: float) -> Tuple[float, float]:
    """`(tau, slope)` for one tradeoff pair, from its signed `x - y` deltas.

    Shared by `tradeoff_pairs` (normalized coordinates) and `select_query`
    (raw), which need the same formula in two different coordinate systems --
    see `select_query` for why both are computed and why they differ.
    """
    slope = metric_gap - compression_gap
    return -compression_gap / slope, slope


def tradeoff_pairs(
    normalized_task_metric, normalized_compression, min_sep: float = 0.05
) -> List[Tuple[int, int, float, float]]:
    """Enumerate genuine tradeoff pairs and their computable threshold `tau`.

    A pair `(i, j)` is a tradeoff only if one candidate is better on the task
    metric and worse on compression -- same-sign deltas mean one dominates
    the other, the answer is predictable, and the query would be wasted. `x`
    is assigned to whichever side is better on the task metric, so for both
    deltas taken as `x - y`: `metric_gap = normalized_task_metric[x] -
    normalized_task_metric[y] > 0` and `compression_gap =
    normalized_compression[x] - normalized_compression[y] < 0`. The `w`
    threshold above which a linear decision-maker prefers `x` is then

        tau = -compression_gap / (metric_gap - compression_gap)

    (derived from `w*metric_gap + (1-w)*compression_gap > 0`). Both deltas
    are signed `x - y` throughout -- no separate "d" quantity, so no sign
    flip to track.

    `min_sep` drops a pair unless `max(metric_gap, -compression_gap) >=
    min_sep` -- the one query filter validated as free by `_V2_ASSESSMENT.md`'s
    ablation. This filter is the reason the caller normalizes at all: it is one
    scalar spanning two objectives with unrelated raw scales, so only against
    the front's own extent does a single constant mean the same thing on both
    axes and across runs.

    Returns `(x_idx, y_idx, tau, slope)` tuples, where
    `slope = metric_gap - compression_gap` (needed by `calibrate_beta`), both
    expressed in whatever coordinates the caller passed in.
    """
    normalized_task_metric = np.asarray(normalized_task_metric, dtype=float)
    normalized_compression = np.asarray(normalized_compression, dtype=float)
    n = normalized_task_metric.shape[0]
    pairs: List[Tuple[int, int, float, float]] = []
    for i in range(n):
        for j in range(i + 1, n):
            if normalized_task_metric[i] > normalized_task_metric[j] and normalized_compression[i] < normalized_compression[j]:
                x_idx, y_idx = i, j
            elif normalized_task_metric[j] > normalized_task_metric[i] and normalized_compression[j] < normalized_compression[i]:
                x_idx, y_idx = j, i
            else:
                continue  # dominated pair or exact tie: not informative

            metric_gap = normalized_task_metric[x_idx] - normalized_task_metric[y_idx]
            compression_gap = normalized_compression[x_idx] - normalized_compression[y_idx]
            if max(metric_gap, -compression_gap) < min_sep:
                continue

            tau, slope = _tau_and_slope(metric_gap, compression_gap)
            pairs.append((x_idx, y_idx, float(tau), float(slope)))
    return pairs


@dataclass(frozen=True)
class QueryChoice:
    """The pair `select_query` picked, carrying `tau`/`slope` in BOTH
    coordinate systems because the two are not interchangeable.

    `tau`/`slope` (unqualified) are RAW, matching the space the posterior and
    `expected_information_gain` work in, so `tau` is directly comparable to
    `WeightPosterior.mean()`/`.median()`. The `*_normalized` pair describes
    the same query in the front-normalized space the `min_sep` filter runs in.
    """

    x_idx: int
    y_idx: int
    expected_information_gain: float
    tau: float
    slope: float
    tau_normalized: float
    slope_normalized: float


def select_query(
    normalized_task_metric,
    normalized_compression,
    raw_task_metric,
    raw_avg_cr,
    posterior: "WeightPosterior",
    min_sep: float = 0.05,
    already_asked: Optional[set] = None,
) -> Optional[QueryChoice]:
    """Pick the `tradeoff_pairs` candidate with the highest expected
    information gain (`WeightPosterior.expected_information_gain`), scored
    from `raw_task_metric`/`raw_avg_cr` (index-aligned with the normalized
    arrays, which are used only for the `min_sep` filter).

    Returns `None` when no pair clears `min_sep` (docs' 8.5: don't force a
    comparison between near-identical candidates) or when every surviving pair
    is in `already_asked`, else a `QueryChoice`.

    `already_asked` holds `pair_key`s answered earlier in this run. Re-asking is
    not a free repeat measurement: a deterministic oracle returns the same
    answer, and `WeightPosterior.recompute` multiplies that identical likelihood
    term in once per stored record, so the posterior narrows on evidence it never
    received. (Measured on a clustering run before this filter existed: 67 of 153
    comparisons were repeats.) A stochastic oracle would make repeats legitimate;
    if one is ever used, this set is what to stop passing.
    Neither `tau` nor `slope` drives selection -- EIG does -- but `slope` is
    what lets a later `beta_oracle` calibration (`IMPLEMENTATION_NOTES.md` S5)
    recover the utility gap this query actually probed:
    `gap(w) = slope * (w - tau)` for any `w`, including a harness's
    known-but-elicitation-invisible ground-truth alpha.

    **`tau`/`slope` are reported in RAW coordinates, not the normalized ones
    the pairs were filtered in.** Ideal/nadir normalization rescales each axis
    independently, which preserves every delta's sign (so the same pairs are
    tradeoffs either way, and `min_sep` is the only part of the filter that
    actually needs normalizing) but *moves* `tau`: with front spans `s_A`,
    `s_B` and `r = s_A / s_B`,

        tau = tau_normalized / (tau_normalized + r * (1 - tau_normalized))

    -- the same front-span reparameterization that makes v1's normalized-
    coordinate fit wrong (module docstring D1, `_V2.md` section 3.3). Two
    consequences if the normalized value is reported instead, both measured on
    a real classification front (`r ~ 2.6`): a logged `tau` sitting next to
    `w_mean` in the same record is not on its comparable scale (the front's
    two extreme endpoints, the most frequently offered pair, always yield
    `tau_normalized = 0.5` exactly -- `dA_hat = d_hat = 1` -- while actually
    cutting at `tau ~ 0.28`), and `gap(w)` above reconstructs a gap in the
    wrong units, off by the same factor. The normalized pair is kept alongside
    so the `min_sep` filter's own geometry stays traceable.
    """
    pairs = tradeoff_pairs(normalized_task_metric, normalized_compression, min_sep=min_sep)
    if not pairs:
        return None
    raw_task_metric = np.asarray(raw_task_metric, dtype=float)
    raw_avg_cr = np.asarray(raw_avg_cr, dtype=float)
    raw_compression = compression_term(raw_avg_cr)
    if already_asked:
        pairs = [
            pair for pair in pairs
            if pair_key(raw_task_metric[pair[0]], raw_avg_cr[pair[0]],
                        raw_task_metric[pair[1]], raw_avg_cr[pair[1]]) not in already_asked
        ]
        if not pairs:
            return None
    choices = []
    for x_idx, y_idx, tau_normalized, slope_normalized in pairs:
        tau, slope = _tau_and_slope(
            raw_task_metric[x_idx] - raw_task_metric[y_idx],
            raw_compression[x_idx] - raw_compression[y_idx],
        )
        choices.append(QueryChoice(
            x_idx=x_idx,
            y_idx=y_idx,
            expected_information_gain=posterior.expected_information_gain(
                raw_task_metric[x_idx], raw_avg_cr[x_idx],
                raw_task_metric[y_idx], raw_avg_cr[y_idx],
            ),
            tau=float(tau),
            slope=float(slope),
            tau_normalized=tau_normalized,
            slope_normalized=slope_normalized,
        ))
    return max(choices, key=lambda choice: choice.expected_information_gain)


def calibrate_beta(raw_components, delta: float = 0.5) -> float:
    """Harness-only helper for calibrating `beta` OFFLINE. Never called by
    `PreferenceGeneticOptimizer` itself -- `beta` is a fixed constant by
    design (recalibrating mid-run shifts the posterior *median*, not just its
    spread; see the module docstring).

    `beta = 4 / (median(slope) * delta)`, where `slope` is `tradeoff_pairs`'
    `metric_gap - compression_gap`, measured on genuine tradeoff pairs in the
    SAME coordinate space `raw_components` is expressed in (that space is
    what the parameter name signals -- pass raw `(task_metric, avg_cr)` pairs
    here to calibrate for the raw-coordinate likelihood this file actually
    fits; passing normalized pairs calibrates for a different, unused space
    and silently produces a beta ~9x too small).

    `delta` is the target posterior transition width in `w`-space
    (`sigma` running from ~0.12 to ~0.88 over `w in [median-delta/2,
    median+delta/2]`); 0.5 is a deliberately soft default.
    """
    raw_components = np.asarray(raw_components, dtype=float)
    task_metric = raw_components[:, 0]
    compression = compression_term(raw_components[:, 1])
    pairs = tradeoff_pairs(task_metric, compression, min_sep=0.0)
    if not pairs:
        raise ValueError("No tradeoff pairs in raw_components; cannot calibrate beta.")
    slopes = np.array([pair[3] for pair in pairs])
    return float(4.0 / (np.median(slopes) * delta))


class WeightPosterior:
    """Grid posterior over `w in [0,1]`, fit in RAW `(task_metric, avg_cr)`
    coordinates.

    Prior is Beta(prior_alpha, prior_beta), default (2, 2): unimodal at 0.5,
    vanishing at the boundaries, so the posterior needs actual evidence to
    rest on an extreme corner. Beta(1, 1) is exactly uniform (every design
    doc's prior), so that behavior is still reachable, not lost. Rebuilt
    from scratch by every `recompute()` call; there is no incremental-update
    path, by design (a stored partial state that can go stale is exactly the
    class of silent bug this design avoids elsewhere).
    """

    def __init__(
        self,
        n_grid: int = 200,
        beta: float = 80.0,
        prior_alpha: float = 2.0,
        prior_beta: float = 2.0,
    ):
        self.n_grid = int(n_grid)
        self.beta = float(beta)
        self.prior_alpha = float(prior_alpha)
        self.prior_beta = float(prior_beta)
        self._grid = np.linspace(0.0, 1.0, self.n_grid)
        self._log_prior = self._beta_log_density(self._grid, self.prior_alpha, self.prior_beta)
        self._posterior = self._normalize_log_density(self._log_prior)

    @staticmethod
    def _beta_log_density(grid: np.ndarray, prior_alpha: float, prior_beta: float) -> np.ndarray:
        # math.lgamma (stdlib) keeps this numpy-only, no scipy dependency.
        # At the grid endpoints, log(0) = -inf is the mathematically correct
        # value whenever prior_alpha/prior_beta > 1 (the Beta density is
        # exactly 0 there) -- expected, not an error, so only that warning
        # class is suppressed.
        log_normalizer = math.lgamma(prior_alpha) + math.lgamma(prior_beta) - math.lgamma(prior_alpha + prior_beta)
        with np.errstate(divide="ignore"):
            return (
                (prior_alpha - 1.0) * np.log(grid)
                + (prior_beta - 1.0) * np.log(1.0 - grid)
                - log_normalizer
            )

    @staticmethod
    def _normalize_log_density(log_density: np.ndarray) -> np.ndarray:
        shifted = log_density - np.max(log_density)
        density = np.exp(shifted)
        return density / density.sum()

    def recompute(self, records: Sequence[ComparisonRecord]) -> None:
        if not records:
            self._posterior = self._normalize_log_density(self._log_prior)
            return

        grid = self._grid
        log_likelihood_total = np.zeros_like(grid)
        for record in records:
            compression_x = compression_term(record.avg_cr_x)
            compression_y = compression_term(record.avg_cr_y)
            score_x = grid * record.task_metric_x + (1.0 - grid) * compression_x
            score_y = grid * record.task_metric_y + (1.0 - grid) * compression_y
            score_difference = (score_x - score_y) if record.prefers_x else (score_y - score_x)
            # log sigmoid(beta*score_difference), via logaddexp for stability
            # at large beta*score_difference.
            log_likelihood_total += -np.logaddexp(0.0, -self.beta * score_difference)

        self._posterior = self._normalize_log_density(self._log_prior + log_likelihood_total)

    def _cdf(self) -> np.ndarray:
        return np.cumsum(self._posterior)

    def median(self) -> float:
        cdf = self._cdf()
        idx = min(int(np.searchsorted(cdf, 0.5)), self.n_grid - 1)
        return float(self._grid[idx])

    def mean(self) -> float:
        """The point estimate that drives every scoring decision (rescoring,
        the final pick) -- NOT `median()`, which is reported only.

        `score(w) = w*A + (1-w)*B` is linear in `w` for a fixed candidate, so
        by linearity of expectation `E[score(w)] = score(E[w])` for ANY
        distribution over `w` -- no symmetry assumption needed. Scoring a
        candidate at `mean()` is therefore not an approximation of "expected
        utility over the whole posterior"; it is that quantity, computed in
        O(1) instead of O(n_grid) per candidate. `median(w)` has no such
        identity and equals this only when the posterior happens to be
        symmetric.
        """
        return float(np.sum(self._grid * self._posterior))

    def credible_interval(self, mass: float = 0.90) -> Tuple[float, float]:
        cdf = self._cdf()
        low_mass = (1.0 - mass) / 2.0
        high_mass = 1.0 - low_mass
        low_idx = min(int(np.searchsorted(cdf, low_mass)), self.n_grid - 1)
        high_idx = min(int(np.searchsorted(cdf, high_mass)), self.n_grid - 1)
        return float(self._grid[low_idx]), float(self._grid[high_idx])

    def decision_is_stable(
        self, components: np.ndarray, mass: float = 0.90
    ) -> Tuple[bool, int]:
        """`components`: (n, 2) array of raw `(task_metric, avg_cr)`.

        Stable iff every `w` in the credible interval selects the same row.
        The returned index is the same `mean()`-driven pick used everywhere
        else, needed whether or not the decision is stable (docs' 8.6).
        """
        components = np.asarray(components, dtype=float)
        task_metric = components[:, 0]
        compression = compression_term(components[:, 1])

        low, high = self.credible_interval(mass)
        in_range = (self._grid >= low) & (self._grid <= high)
        grid_in_range = self._grid[in_range]
        if grid_in_range.size == 0:
            grid_in_range = np.array([self.mean()])

        scores = grid_in_range[:, None] * task_metric[None, :] + (1.0 - grid_in_range[:, None]) * compression[None, :]
        argmaxes = np.argmax(scores, axis=1)
        stable = bool(np.all(argmaxes == argmaxes[0]))

        mean_w = self.mean()
        expected_scores = mean_w * task_metric + (1.0 - mean_w) * compression
        pick_idx = int(np.argmax(expected_scores))
        return stable, pick_idx

    def expected_information_gain(
        self, task_metric_x: float, avg_cr_x: float, task_metric_y: float, avg_cr_y: float
    ) -> float:
        """Expected reduction in posterior entropy over `w` from asking about
        this pair (see module docstring for why EIG over the tau heuristic).

        Standard one-step Bayesian experimental-design objective, computed in
        closed form on the grid: for each grid point `w`, the Bradley-Terry
        likelihood gives `P(prefers x | w)`; marginalizing over the CURRENT
        posterior gives the predictive `P(prefers x)`, and Bayes' rule gives
        the two possible posteriors after each answer. No simulation needed.

        Evaluated in RAW coordinates, like `recompute` (D1) -- never normalized.
        """
        grid = self._grid
        compression_x = compression_term(avg_cr_x)
        compression_y = compression_term(avg_cr_y)
        score_x = grid * task_metric_x + (1.0 - grid) * compression_x
        score_y = grid * task_metric_y + (1.0 - grid) * compression_y

        # P(prefers x | w), the same likelihood recompute() uses.
        probability_prefers_x_given_w = np.exp(-np.logaddexp(0.0, -self.beta * (score_x - score_y)))

        current = self._posterior
        predictive_prefers_x = float(np.sum(current * probability_prefers_x_given_w))
        predictive_prefers_y = 1.0 - predictive_prefers_x

        entropy_before = _entropy(current)
        entropy_after = 0.0
        if predictive_prefers_x > 0.0:
            posterior_if_x = current * probability_prefers_x_given_w
            posterior_if_x /= posterior_if_x.sum()
            entropy_after += predictive_prefers_x * _entropy(posterior_if_x)
        if predictive_prefers_y > 0.0:
            posterior_if_y = current * (1.0 - probability_prefers_x_given_w)
            posterior_if_y /= posterior_if_y.sum()
            entropy_after += predictive_prefers_y * _entropy(posterior_if_y)

        return entropy_before - entropy_after


# =====================================================================
# Part B -- the GA integration. Subclasses GeneticOptimizer for its
# already-correct helpers only; writes its own maximize() rather than the
# seam-based override the staged plan uses, since that seam requires editing
# genetic.py.
# =====================================================================


_PO_WORKER_OBJECTIVE = None


def _po_initialize_worker(objective) -> None:
    global _PO_WORKER_OBJECTIVE
    _PO_WORKER_OBJECTIVE = objective


def _po_evaluate_worker_candidate(candidate):
    """Runs in the worker process. `objective` here must return
    `(task_metric, avg_cr)` -- this optimizer's objective contract, not the
    rest of the repo's `objective(candidate) -> float` (see module
    docstring)."""
    return _PO_WORKER_OBJECTIVE(candidate)


def _po_kill_pool(executor: cf.ProcessPoolExecutor) -> None:
    for worker_process in list(executor._processes.values()):
        worker_process.kill()
    executor.shutdown(wait=False, cancel_futures=True)


class PreferenceGeneticOptimizer(GeneticOptimizer):
    """GA whose scalarization weight is elicited during the run from pairwise
    comparisons, instead of being a fixed, known `alpha`.

    Everything about the search itself -- `Space`, crossover, mutation,
    tournament selection, elitism, the forkserver pool, the batch timeout
    deadlock net -- is reused unchanged from `GeneticOptimizer`. What differs:
    every evaluated candidate carries its raw `(task_metric, avg_cr)`
    alongside its GA fitness, an archive accumulates every candidate ever
    evaluated, and on a generation schedule a comparison is asked against the
    archive's current non-dominated front (the pair chosen by expected
    information gain, see `WeightPosterior.expected_information_gain`),
    folded into a `WeightPosterior`, and the whole population *and* archive
    are rescored from their stored components at zero additional evaluation
    cost.

    `total_budget`/`n_evaluations` accounting is untouched: comparisons never
    trigger a pipeline evaluation, so this is exactly `GeneticOptimizer`'s
    `initial_pop_size + gens * (pop_size - elitism)`.
    """

    def __init__(
        self,
        oracle: Optional[Callable[[Tuple[float, float], Tuple[float, float]], bool]] = None,
        alpha: Optional[float] = None,
        beta: float = 80.0,
        n_grid: int = 200,
        prior_alpha: float = 2.0,
        prior_beta: float = 2.0,
        comparison_every: int = 2,
        comparisons_per_round: int = 1,
        max_comparisons: int = 12,
        min_sep: float = 0.1,
        credible_mass: float = 0.90,
        pop_size=12,
        initial_pop_size=None,
        gens=8,
        cx_prob=0.8,
        mut_prob=0.5,
        mut_per_var_prob=0.6,
        sigma_frac=0.1,
        log_sigma=0.75,
        elitism=1,
        tournament_k=2,
        random_state=2048,
        n_workers=10,
        batch_timeout_sec=3600,
        verbose=1,
        log_subdir="_preference",
    ):
        target_alpha = float(alpha) if alpha is not None else None
        self.target_alpha = target_alpha
        if oracle is None:
            if target_alpha is None:
                raise ValueError(
                    "PreferenceGeneticOptimizer requires either an explicit oracle "
                    "callable or alpha (to build a fallback hidden-alpha oracle)."
                )
            oracle = _make_hidden_alpha_oracle(target_alpha)
        self.oracle = oracle
        self.beta = float(beta)
        self.n_grid = int(n_grid)
        self.prior_alpha = float(prior_alpha)
        self.prior_beta = float(prior_beta)
        self.comparison_every = int(comparison_every)
        self.comparisons_per_round = int(comparisons_per_round)
        self.max_comparisons = int(max_comparisons)
        self.min_sep = float(min_sep)
        self.credible_mass = float(credible_mass)

        super().__init__(
            pop_size=pop_size,
            initial_pop_size=initial_pop_size,
            gens=gens,
            cx_prob=cx_prob,
            mut_prob=mut_prob,
            mut_per_var_prob=mut_per_var_prob,
            alpha=float("nan"),  # no single fixed alpha; see module docstring
            sigma_frac=sigma_frac,
            log_sigma=log_sigma,
            elitism=elitism,
            tournament_k=tournament_k,
            # The GA's own "no new best in N generations" early stop is
            # unrelated to comparison scheduling and not part of what any
            # version of the docs discusses; disabled here (not "never
            # relitigated", just out of this POC's scope) rather than left to
            # interact with rescoring in an unreviewed way.
            early_stop_patience=10**9,
            random_state=random_state,
            n_workers=n_workers,
            batch_timeout_sec=batch_timeout_sec,
            verbose=verbose,
            log_subdir=log_subdir,
        )
        # Diagnostics, set at the end of maximize() -- a side-channel
        # attribute, same pattern as the base class's self.last_run_dir,
        # since maximize()'s return type stays Dict[str, float] everywhere
        # in this repo.
        self.last_w_median: Optional[float] = None
        self.last_w_mean: Optional[float] = None
        self.last_credible_interval: Optional[Tuple[float, float]] = None
        self.last_n_comparisons: Optional[int] = None
        self.last_stability_reached: Optional[bool] = None

    def _score(self, components: Sequence[Tuple[float, float]], w: float) -> List[float]:
        # combine_fitness, not inlined arithmetic -- makes drift from the
        # canonical definition structurally impossible.
        return [combine_fitness(task_metric, avg_cr, w) for task_metric, avg_cr in components]

    def maximize(
        self, objective, search_space, space_definition, log_dir, run_metadata, **kwargs
    ) -> Dict[str, float]:
        parameter_space = Space(search_space, space_definition)
        parameter_names = list(parameter_space.variable_specs.keys())

        logger = RunLogger(
            log_dir=log_dir,
            run_metadata=run_metadata,
            optimizer_name="preference",
            optimizer_meta={
                "pop_size": self.pop_size,
                "initial_pop_size": self.initial_pop_size,
                "gens": self.gens,
                "cx_prob": self.cx_prob,
                "mut_prob": self.mut_prob,
                "mut_per_var_prob": self.mut_per_var_prob,
                "sigma_frac": self.sigma_frac,
                "log_sigma": self.log_sigma,
                "elitism": self.elitism,
                "tournament_k": self.tournament_k,
                # Real target alpha for a framework run, NaN placeholder for the
                # disconnected toy/harness path -- see module docstring.
                "alpha": self.target_alpha if self.target_alpha is not None else self.alpha,
                "random_state": self.random_state,
                "n_workers": self.n_workers,
                "beta": self.beta,
                "n_grid": self.n_grid,
                "prior_alpha": self.prior_alpha,
                "prior_beta": self.prior_beta,
                "comparison_every": self.comparison_every,
                "comparisons_per_round": self.comparisons_per_round,
                "max_comparisons": self.max_comparisons,
                "min_sep": self.min_sep,
                "credible_mass": self.credible_mass,
            },
            search_space=search_space,
            space_definition=space_definition,
            param_names=parameter_names,
            total_budget=self.total_budget,
            log_subdir=self.log_subdir,
            verbose=self.verbose,
        )
        self.last_run_dir = logger.run_dir

        executor = (
            cf.ProcessPoolExecutor(
                max_workers=min(int(self.n_workers), max(self.pop_size, self.initial_pop_size)),
                mp_context=multiprocessing.get_context("forkserver"),
                initializer=_po_initialize_worker,
                initargs=(objective,),
            )
            if (self.n_workers and self.n_workers > 1)
            else None
        )

        posterior = WeightPosterior(
            n_grid=self.n_grid, beta=self.beta,
            prior_alpha=self.prior_alpha, prior_beta=self.prior_beta,
        )
        records: List[ComparisonRecord] = []
        # posterior.mean(), not .median() -- the operating value that drives
        # rescoring and the final pick (see WeightPosterior.mean's docstring
        # for why the mean, specifically, equals expected utility here).
        w = posterior.mean()

        archive_candidates: List[Dict[str, float]] = []
        archive_components: List[Tuple[float, float]] = []

        evaluations = 0
        best_candidate: Optional[Dict[str, float]] = None
        best_score = float("-inf")
        pool_killed = False
        prev_ideal_nadir: Optional[Tuple[Tuple[float, float], Tuple[float, float]]] = None
        n_comparisons_so_far = 0
        asked_pairs = set()  # pair_key of every query answered this run; see select_query

        def _evaluate_components(candidate_batch):
            nonlocal pool_killed
            batch_started_at = time.perf_counter()
            if executor is not None:
                try:
                    batch_components = list(
                        executor.map(
                            _po_evaluate_worker_candidate,
                            candidate_batch,
                            timeout=self.batch_timeout_sec,
                        )
                    )
                except TimeoutError as timeout_error:
                    pool_killed = True
                    _po_kill_pool(executor)
                    raise TimeoutError(
                        f"preference: a batch of {len(candidate_batch)} "
                        f"evaluations exceeded batch_timeout_sec="
                        f"{self.batch_timeout_sec}s; the worker pool stopped "
                        f"making progress after {evaluations} evaluation(s)."
                    ) from timeout_error
            else:
                batch_components = [objective(candidate) for candidate in candidate_batch]
            batch_elapsed_seconds = time.perf_counter() - batch_started_at
            return batch_components, batch_elapsed_seconds / len(candidate_batch)

        def _log_components_batch(candidate_batch, components_batch, scores, phase, proposal_source, seconds_per_candidate):
            nonlocal evaluations, best_score, best_candidate
            if isinstance(proposal_source, str):
                proposal_source = [proposal_source] * len(candidate_batch)
            for candidate, (task_metric, avg_cr), score, source in zip(
                candidate_batch, components_batch, scores, proposal_source
            ):
                evaluations += 1
                global_best_before = best_score
                is_new_best = score > best_score
                if is_new_best:
                    best_score = score
                    best_candidate = candidate
                logger.log_evaluation(
                    evaluation=evaluations,
                    params_raw=candidate,
                    params_model=candidate,
                    reward=float(score),
                    elapsed_sec=seconds_per_candidate,
                    global_best_before=global_best_before,
                    global_best_after=best_score,
                    is_new_global_best=is_new_best,
                    phase=phase,
                    proposal_source=source,
                )
                # task_metric/avg_cr have no RunLogger CSV columns (that file
                # is not edited here); logged as an event instead, RunLogger's
                # existing arbitrary-payload seam.
                logger.log_event({
                    "event": "evaluation_components",
                    "evaluation": evaluations,
                    "task_metric": float(task_metric),
                    "avg_cr": float(avg_cr),
                    "w_at_eval": float(w),
                })

        def _resync_best_from_archive():
            # Called after every w update: the incrementally-tracked
            # best_score/best_candidate above were computed under the OLD w
            # and are stale the instant w moves -- this is the "population
            # rescoring" requirement applied to the running best-tracker too,
            # not just the tournament/elitism scores below.
            nonlocal best_score, best_candidate
            scores = self._score(archive_components, w)
            idx = int(np.argmax(scores))
            best_score = scores[idx]
            best_candidate = archive_candidates[idx]

        try:
            population, seen_keys, search_space_exhausted = (
                self._sample_unique_population(parameter_space, parameter_names, self.initial_pop_size)
            )
            population_components, seconds_per_candidate = _evaluate_components(population)
            population_scores = self._score(population_components, w)
            archive_candidates.extend(population)
            archive_components.extend(population_components)
            _log_components_batch(
                population, population_components, population_scores,
                "init", "unique_random_init", seconds_per_candidate,
            )

            generation = 0
            while generation < self.gens and not search_space_exhausted:
                generation += 1
                ranked = sorted(
                    zip(population, population_components, population_scores),
                    key=lambda triple: triple[2],
                    reverse=True,
                )[: self.elitism]
                elite_population, elite_components, elite_scores = map(list, zip(*ranked))

                offspring_batch = []
                proposal_sources = []
                for elite_candidate in elite_population:
                    refined_candidate, refined_key = self._propose_elite_refinement(
                        parameter_space, elite_candidate, parameter_names, seen_keys
                    )
                    offspring_batch.append(refined_candidate)
                    proposal_sources.append("elite_error_refinement")
                    seen_keys.add(refined_key)

                while len(offspring_batch) < self.offspring_per_generation:
                    offspring, offspring_key = self._propose_unique_child(
                        parameter_space, population, population_scores, parameter_names, seen_keys
                    )
                    offspring_batch.append(offspring)
                    proposal_sources.append("tournament_distinct_parents_unique")
                    seen_keys.add(offspring_key)

                offspring_components, seconds_per_candidate = _evaluate_components(offspring_batch)
                offspring_scores = self._score(offspring_components, w)
                archive_candidates.extend(offspring_batch)
                archive_components.extend(offspring_components)
                _log_components_batch(
                    offspring_batch, offspring_components, offspring_scores,
                    "generation", proposal_sources, seconds_per_candidate,
                )

                population = elite_population + offspring_batch
                population_components = elite_components + offspring_components
                population_scores = elite_scores + offspring_scores

                # -- comparison round: generations, not evaluation counts
                # (schedule drifts with budget otherwise), never skipped for
                # "looks stable" (D11).
                if n_comparisons_so_far < self.max_comparisons:
                    archive_metric = np.array([components[0] for components in archive_components])
                    archive_avg_cr = np.array([components[1] for components in archive_components])
                    # The archive does not change inside a round, so the front and its
                    # normalization are computed once; only the posterior moves between
                    # the asks below, which is what re-running select_query picks up.
                    front = normalize_front(
                        archive_metric, archive_avg_cr, prev_ideal_nadir=prev_ideal_nadir
                    )
                    if front is None:
                        # Collapsed front: no pair to ask about (see normalize_front).
                        non_dominated_idx = _non_dominated_indices(
                            archive_metric, compression_term(archive_avg_cr)
                        )
                    else:
                        normalized_task_metric, normalized_compression, non_dominated_idx, ideal, nadir = front
                        prev_ideal_nadir = (ideal, nadir)

                    asked_this_round = 0
                    while (n_comparisons_so_far < self.max_comparisons
                           and asked_this_round < self.comparisons_per_round):
                        query = None if front is None else select_query(
                            normalized_task_metric[non_dominated_idx],
                            normalized_compression[non_dominated_idx],
                            archive_metric[non_dominated_idx],
                            archive_avg_cr[non_dominated_idx],
                            posterior,
                            min_sep=self.min_sep,
                            already_asked=asked_pairs,
                        )
                        if query is None:
                            if asked_this_round == 0:
                                logger.log_event({
                                    "event": "comparison_skipped_no_pair",
                                    "generation": generation,
                                    "front_collapsed": front is None,
                                    "n_front_points": int(non_dominated_idx.size),
                                    "n_already_asked": len(asked_pairs),
                                })
                            break
                        x_idx = int(non_dominated_idx[query.x_idx])
                        y_idx = int(non_dominated_idx[query.y_idx])
                        task_metric_x, avg_cr_x = archive_components[x_idx]
                        task_metric_y, avg_cr_y = archive_components[y_idx]

                        prefers_x = bool(self.oracle((task_metric_x, avg_cr_x), (task_metric_y, avg_cr_y)))
                        records.append(ComparisonRecord(
                            task_metric_x=task_metric_x, avg_cr_x=avg_cr_x,
                            task_metric_y=task_metric_y, avg_cr_y=avg_cr_y,
                            prefers_x=prefers_x,
                        ))
                        n_comparisons_so_far += 1
                        asked_this_round += 1
                        asked_pairs.add(pair_key(task_metric_x, avg_cr_x, task_metric_y, avg_cr_y))

                        posterior.recompute(records)
                        w = posterior.mean()  # operating value; see WeightPosterior.mean
                        population_scores = self._score(population_components, w)
                        _resync_best_from_archive()

                        credible_low, credible_high = posterior.credible_interval(self.credible_mass)
                        logger.log_event({
                            "event": "comparison_round",
                            "generation": generation,
                            "n_comparisons_so_far": n_comparisons_so_far,
                            "ask_within_round": asked_this_round,
                            # raw coordinates, i.e. w_mean's own scale -- see QueryChoice
                            "tau_asked": query.tau,
                            "slope_asked": query.slope,
                            "tau_asked_normalized": query.tau_normalized,
                            "slope_asked_normalized": query.slope_normalized,
                            "expected_information_gain": float(query.expected_information_gain),
                            "w_mean": float(w),  # operating value -- drives rescoring/pick
                            "w_median": float(posterior.median()),  # reported diagnostic only
                            "credible_interval_low": float(credible_low),
                            "credible_interval_high": float(credible_high),
                            "task_metric_x": float(task_metric_x),
                            "avg_cr_x": float(avg_cr_x),
                            "task_metric_y": float(task_metric_y),
                            "avg_cr_y": float(avg_cr_y),
                            "prefers_x": prefers_x,
                        })
                        if self.verbose:
                            print(
                                f"[preference] gen {generation}: "
                                f"comparison {n_comparisons_so_far}, "
                                f"EIG={query.expected_information_gain:.4f}, "
                                f"tau={query.tau:.3f}, "
                                f"slope={query.slope:.3f}, w_mean={w:.3f}, "
                                f"90% CI=({credible_low:.3f}, {credible_high:.3f})"
                            )
        finally:
            if executor is not None and not pool_killed:
                executor.shutdown(wait=True)

        assert best_candidate is not None  # a failed evaluation raises out of _evaluate_components

        final_components = np.array(archive_components, dtype=float)
        stable, pick_idx = posterior.decision_is_stable(final_components, mass=self.credible_mass)
        best_candidate = archive_candidates[pick_idx]
        best_score = self._score([archive_components[pick_idx]], w)[0]
        credible_low, credible_high = posterior.credible_interval(self.credible_mass)

        self.last_w_mean = w  # operating value -- drove rescoring and this pick
        self.last_w_median = posterior.median()  # reported diagnostic only
        self.last_credible_interval = (credible_low, credible_high)
        self.last_n_comparisons = n_comparisons_so_far
        self.last_stability_reached = stable

        logger.log_event({
            "event": "preference_summary",
            "n_comparisons": n_comparisons_so_far,
            "w_mean": float(w),
            "w_median": float(self.last_w_median),
            "credible_interval_low": float(credible_low),
            "credible_interval_high": float(credible_high),
            "stability_reached": stable,
        })

        if self.verbose:
            print(
                f"Optimization finished. w_mean={w:.4f} "
                f"(90% CI=({credible_low:.3f}, {credible_high:.3f})), "
                f"stability_reached={stable}, best score: {round(best_score, 4)}"
            )
            print("Best candidate:", {k: round(v, 4) for k, v in best_candidate.items()})

        logger.write_summary(
            evaluations=evaluations,
            total_budget=self.total_budget,
            best_reward=best_score,
            best_params=best_candidate,
            status="completed",
        )
        return best_candidate


# =====================================================================
# Self-contained smoke test. No datasets, no other repo files, no disk
# writes (log_dir=None makes RunLogger a no-op by its own contract).
#
#   python -m optimizer.preference
#
# A deterministic toy problem with a genuine accuracy/compression tradeoff:
# task_metric falls and avg_cr rises with `x`; `y` is a decoy dimension the
# GA has to learn to ignore. The oracle is a hidden alpha chosen far from the
# Beta(2,2) prior's median (0.5), so recovery is a real test, not a vacuous
# one.
# =====================================================================


class _ToyObjective:
    def __call__(self, candidate: Dict[str, float]) -> Tuple[float, float]:
        x = candidate["x"]
        task_metric = 1.0 - 0.6 * x**2
        avg_cr = 1.0 + 20.0 * x
        return float(task_metric), float(avg_cr)


def _make_hidden_alpha_oracle(hidden_alpha: float):
    def oracle(components_x, components_y) -> bool:
        score_x = combine_fitness(components_x[0], components_x[1], hidden_alpha)
        score_y = combine_fitness(components_y[0], components_y[1], hidden_alpha)
        return score_x > score_y
    return oracle


if __name__ == "__main__":
    hidden_alpha = 0.30  # far from the prior's median -- a real test

    search_space = {"x": [0.0, 1.0], "y": [0.0, 1.0]}
    space_definition = {
        "x": {"type": "float", "scale": "linear"},
        "y": {"type": "float", "scale": "linear"},
    }

    optimizer = PreferenceGeneticOptimizer(
        oracle=_make_hidden_alpha_oracle(hidden_alpha),
        beta=80.0,
        comparison_every=2,
        max_comparisons=12,
        pop_size=12,
        gens=8,
        n_workers=1,  # serial: simplest path for a first smoke test
        verbose=1,
    )
    best = optimizer.maximize(
        _ToyObjective(), search_space, space_definition, log_dir=None, run_metadata=None,
    )
    print()
    print(f"hidden alpha:   {hidden_alpha}")
    print(f"recovered w:    median={optimizer.last_w_median:.4f} mean={optimizer.last_w_mean:.4f}")
    print(f"90% CI:         {optimizer.last_credible_interval}")
    print(f"n_comparisons:  {optimizer.last_n_comparisons}")
    print(f"stable:         {optimizer.last_stability_reached}")
    print(f"best candidate: {best}")
