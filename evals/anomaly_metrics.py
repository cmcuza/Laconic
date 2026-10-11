"""Anomaly detection metrics, computed with TSB-AD's evaluation code."""
from __future__ import annotations
from typing import Dict
import numpy as np


# TSB-AD point-wise metrics
TSB_AD_METRIC_KEYS = ("AUC-PR", "AUC-ROC", "Standard-F1")

# Quality measures the objective can optimize
AD_Q_MEASURES = ("AUC-PR", "VUS-PR")

# Version and threshold count of the VUS volume, as in TSB-AD's get_metrics.
VUS_VERSION, VUS_THRE = "opt", 250


class NonFiniteScores(RuntimeError):
    """The detector returned NaN or infinite scores."""


def evaluate_anomaly_tsb(y_true, y_score) -> Dict[str, float]:
    """AUC-PR, AUC-ROC and Standard-F1 for one series.

    Args:
      y_true: (T,) 0/1 ground truth.
      y_score: (T,) detector scores, not rescaled.

    Returns:
      A dict keyed by TSB_AD_METRIC_KEYS.
    """
    # Local import, so the other tasks can import evals without TSB-AD installed.
    from TSB_AD.evaluation.basic_metrics import basic_metricor

    scores, labels = _as_score_label_pair(y_score, y_true)
    grader = basic_metricor()
    return {
        "AUC-PR": float(grader.metric_PR(labels, scores)),
        "AUC-ROC": float(grader.metric_ROC(labels, scores)),
        "Standard-F1": float(grader.metric_PointF1(labels, scores, preds=None)),
    }


def vus_pr(y_true, y_score, sliding_window: int) -> float:
    """VUS-PR for one series, or for one index range of it.

    Args:
      y_true: (T,) 0/1 ground truth.
      y_score: (T,) detector scores.
      sliding_window: Maximum buffer width of the volume. It is computed on the
        uncompressed series, so it is the same for every pipeline.
    """
    from TSB_AD.evaluation.basic_metrics import generate_curve

    scores, labels = _as_score_label_pair(y_score, y_true)
    return float(generate_curve(labels, scores, int(sliding_window), VUS_VERSION, VUS_THRE)[-1])


def anomaly_q(y_true, y_score, measure: str = "AUC-PR", sliding_window=None) -> float:
    """Detection quality Q under `measure`. AUC-PR is sklearn's average precision."""
    if measure == "AUC-PR":
        from sklearn.metrics import average_precision_score
        return float(average_precision_score(y_true, y_score))
    if measure == "VUS-PR":
        if sliding_window is None:
            raise ValueError("VUS-PR needs the sliding_window of the uncompressed series")
        return vus_pr(y_true, y_score, sliding_window)
    raise ValueError(f"Unknown quality measure '{measure}'. Available: {list(AD_Q_MEASURES)}")


def _as_score_label_pair(y_score, y_true):
    scores = np.asarray(y_score, dtype=float).ravel()
    labels = np.asarray(y_true).astype(int).ravel()
    if scores.shape != labels.shape:
        raise ValueError(f"{scores.shape[0]} scores for {labels.shape[0]} labels")
    return scores, labels
