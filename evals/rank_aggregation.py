"""Ballot aggregation: turn per-voter scores into one consensus ranking."""
from __future__ import annotations

import itertools
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from evals.metrics import combine_fitness

AGGREGATIONS = ("borda", "kemeny", "markov", "mean_fitness", "mean_normalized")
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
    """`M[i, j]` = total weight of voters ranking item i above item j."""
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
    """Exact Kemeny-optimal order (best first) by brute force over n!."""
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
    """MC4 stationary distribution - HIGHER is better."""
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
    """Mean scalarized fitness across folds - HIGHER is better."""
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
    """(order best-first, per-item score, whether Kemeny was prefiltered)."""
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

    n_items = len(borda)
    prefiltered = n_items > kemeny_max_items
    borda_order = list(np.argsort(borda, kind="mergesort"))
    keep = borda_order[: min(n_items, kemeny_max_items)]
    order = [keep[index] for index in kemeny_order(matrix[np.ix_(keep, keep)])]
    ranked = set(order)
    order += [index for index in borda_order if index not in ranked]
    return order, borda, prefiltered
