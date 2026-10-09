"""Ballot aggregation: turn per-voter scores into one consensus ranking.

Pure functions over arrays - no optimizer, no objective, no I/O. They were
extracted from `optimizer/rank_aggregation.py` when the within-fold
rank-aggregation *optimizer* was removed (it implemented the assessment's weaker
Design B\' and never produced a reported number). The aggregation itself is very
much alive: it is what `scripts/rank_aggregation_selection.py` - stage 2, the
cross-fold consensus every reported number comes from - votes with.

They live in `evals/` rather than `optimizer/` because that is what they are:
scoring and ranking over measured values, the same family as `evals/metrics.py`,
whose `combine_fitness` is their only non-numpy dependency.

`docs/RANK_AGGREGATION_ASSESSMENT.md` is the argument these implement, and
`scripts/rank_aggregation_toy.py` reproduces its worked example against them.
"""
from __future__ import annotations

import itertools
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from evals.metrics import combine_fitness

AGGREGATIONS = ("borda", "kemeny", "markov", "mean_fitness", "mean_normalized")
# The two that are NOT rank-based: they keep magnitudes.
SCORE_AGGREGATIONS = ("mean_fitness", "mean_normalized")
WEIGHTINGS = ("split", "scalarized")
VOTER_SCHEMES = ("bootstrap", "partition")


def rank_positions(values: Sequence[float], higher_is_better: bool = True) -> np.ndarray:
    """Competition ranks (1 = best), ties sharing their average position."""
    scores = np.asarray(values, dtype=float)
    if not higher_is_better:
        scores = -scores
    order = np.argsort(-scores, kind="mergesort")
    ranks = np.empty(len(scores), dtype=float)
    position = 0
    while position < len(order):
        stop = position + 1
        while stop < len(order) and scores[order[stop]] == scores[order[position]]:
            stop += 1
        ranks[order[position:stop]] = (position + stop + 1) / 2.0
        position = stop
    return ranks


def pairwise_preference_matrix(
    ballots: Sequence[np.ndarray], weights: Sequence[float]
) -> np.ndarray:
    """`M[i, j]` = total weight of voters ranking item i above item j.

    A voter that ties i and j contributes to neither direction, which is what
    keeps a coarse task metric (clustering ARI ties often) from inventing a
    preference it never expressed."""
    n_items = len(ballots[0])
    matrix = np.zeros((n_items, n_items), dtype=float)
    for ballot, weight in zip(ballots, weights):
        better = ballot[:, None] < ballot[None, :]
        matrix += float(weight) * better
    return matrix


def borda_scores(ballots: Sequence[np.ndarray], weights: Sequence[float]) -> np.ndarray:
    """Weighted mean rank position - LOWER is better."""
    total = float(sum(weights))
    stacked = np.vstack([np.asarray(w, dtype=float) * b for b, w in zip(ballots, weights)])
    return stacked.sum(axis=0) / total


def kemeny_order(matrix: np.ndarray) -> Tuple[int, ...]:
    """Exact Kemeny-optimal order (best first) by brute force over n!.

    Cost of placing i before j is the weight of voters who preferred j. NP-hard
    in general - callers must prefilter; see the module docstring.

    **Ties are common and the tie-break is load-bearing.** On the assessment's
    own five-pipeline toy at alpha=0.75, four distinct orders share the optimal
    cost of 9.0 and they do not agree on the winner (A vs C). A coarse task
    metric makes this worse, not better - clustering ARI ties constantly. The
    strict `<` below keeps the FIRST minimum found, i.e. the one closest to the
    identity order of the matrix it is handed; `consensus_order` hands it a
    Borda-sorted submatrix, so ties there break toward the Borda consensus.
    That is a deterministic, documented choice, not an accident of iteration
    order - do not "optimize" it into `<=`."""
    n_items = matrix.shape[0]
    best_order, best_cost = None, float("inf")
    for order in itertools.permutations(range(n_items)):
        cost = sum(
            matrix[order[b], order[a]]
            for a in range(n_items)
            for b in range(a + 1, n_items)
        )
        if cost < best_cost:
            best_cost, best_order = cost, order
    return best_order


def markov_scores(matrix: np.ndarray, damping: float = 0.85, iterations: int = 5000) -> np.ndarray:
    """MC4 stationary distribution - HIGHER is better.

    Damping is not optional: without it a Condorcet winner is an absorbing
    state and the distribution collapses to a single 1.0 with every other
    candidate at exactly 0, which is an ordering with no margins in it."""
    n_items = matrix.shape[0]
    transitions = np.zeros((n_items, n_items), dtype=float)
    for source in range(n_items):
        for target in range(n_items):
            if source != target and matrix[target, source] > matrix[source, target]:
                transitions[source, target] = 1.0 / n_items
        transitions[source, source] = 1.0 - transitions[source].sum()
    transitions = damping * transitions + (1.0 - damping) / n_items
    distribution = np.full(n_items, 1.0 / n_items)
    for _ in range(iterations):
        updated = distribution @ transitions
        if np.allclose(updated, distribution, atol=1e-14):
            distribution = updated
            break
        distribution = updated
    return distribution


def mean_fitness_scores(matrix: np.ndarray, alpha: float, normalize: bool) -> np.ndarray:
    """Mean scalarized fitness across folds - HIGHER is better. Not a rank rule.

    `matrix` is (n_candidates, n_folds, 2) of (task_metric, avg_cr).

    Rank aggregation throws away magnitude, and that is not always harmless: on
    a measured clustering fold set (Lightning2), four of five candidates had
    *identical* ARI on four of the five folds, so the task-metric ballots - the
    ones carrying weight `alpha` - were a four-way tie contributing nothing, and
    the CR ballots at weight `1 - alpha` decided the winner outright. The
    candidate with the best mean fitness across all folds lost. Keeping
    magnitudes makes a tie contribute a zero difference instead of erasing the
    dimension.

    `normalize=True` min-max scales task_metric and avg_cr over the candidate
    pool *within each fold* before combining. Two things that buys: the two
    objectives get comparable spans, so `alpha` means what it says instead of
    being applied to `1 - 1/CR`'s saturated scale (see the assessment's section
    2); and per-fold scaling removes the between-fold level differences that
    make `best_val` mostly a measure of which fold drew the easy split. A fold
    where every candidate ties contributes 0.5 to everyone - no information,
    rather than noise."""
    n_candidates, n_folds, _ = matrix.shape
    scores = np.zeros(n_candidates, dtype=float)
    for k in range(n_folds):
        metrics, ratios = matrix[:, k, 0].copy(), matrix[:, k, 1].copy()
        if normalize:
            for values in (metrics, ratios):
                low, high = values.min(), values.max()
                values[:] = 0.5 if high - low <= 0 else (values - low) / (high - low)
            scores += alpha * metrics + (1.0 - alpha) * ratios
        else:
            scores += np.array([combine_fitness(m, c, alpha) for m, c in zip(metrics, ratios)])
    return scores / n_folds


def consensus_order(
    ballots: Sequence[np.ndarray],
    weights: Sequence[float],
    aggregation: str,
    damping: float = 0.85,
    kemeny_max_items: int = 8,
) -> Tuple[List[int], np.ndarray, bool]:
    """(order best-first, per-item score, whether Kemeny was prefiltered).

    The returned score is the aggregator's own quantity - mean rank for borda
    (lower better), stationary mass for markov (higher better), and for kemeny
    the borda score, since a Kemeny consensus is an order without a cardinal
    score attached."""
    if aggregation not in AGGREGATIONS:
        raise ValueError(f"Unknown aggregation '{aggregation}'. Expected one of {AGGREGATIONS}.")
    if aggregation in SCORE_AGGREGATIONS:
        raise ValueError(
            f"'{aggregation}' is score-based, not rank-based: call mean_fitness_scores(matrix, "
            "alpha, normalize=...) with the raw (n_candidates, n_folds, 2) matrix. It cannot be "
            "computed from ballots, which is the point of it."
        )
    borda = borda_scores(ballots, weights)
    if aggregation == "borda":
        return list(np.argsort(borda, kind="mergesort")), borda, False

    matrix = pairwise_preference_matrix(ballots, weights)
    if aggregation == "markov":
        scores = markov_scores(matrix, damping=damping)
        return list(np.argsort(-scores, kind="mergesort")), scores, False

    # kemeny: exact over at most kemeny_max_items, Borda-prefiltered. `keep` is
    # in Borda order, so the exact search's tie-break (see kemeny_order) resolves
    # toward the Borda consensus rather than toward candidate numbering.
    n_items = len(borda)
    prefiltered = n_items > kemeny_max_items
    borda_order = list(np.argsort(borda, kind="mergesort"))
    keep = borda_order[: min(n_items, kemeny_max_items)]
    order = [keep[index] for index in kemeny_order(matrix[np.ix_(keep, keep)])]
    # Anything the prefilter dropped keeps its Borda position, behind the
    # Kemeny-ordered head. Only reachable when prefiltered is True.
    ranked = set(order)
    order += [index for index in borda_order if index not in ranked]
    return order, borda, prefiltered
