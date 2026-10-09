from __future__ import annotations
from dataclasses import dataclass
from typing import Callable, Dict, Iterable, Optional, Tuple, Sequence
import numpy as np
from sklearn import metrics as skm
from sklearn.preprocessing import label_binarize, MinMaxScaler

def _is_binary(y_true: np.ndarray) -> bool:
    return np.unique(y_true).shape[0] == 2

def _positive_col_index(classes, positive_label=None) -> int:
    if classes is None:
        raise ValueError("Score-based binary metrics need `classes` to locate the positive-class column.")
    if positive_label is None:
        positive_label = classes[-1]
    return list(classes).index(positive_label)

def _binary_pos_scores(y_true, y_score, classes=None):
    y_score = np.asarray(y_score)
    if y_score.ndim == 1:
        return y_score
    return y_score[:, _positive_col_index(classes)]


@dataclass(frozen=True)
class MetricSpec:
    fn: Callable[..., float]
    needs_score: bool        
    value_range: Tuple[float, float]

    def normalized(self, v: float) -> float:
        lo, hi = self.value_range
        if hi == lo:
            return 0.0
        v = max(min(v, hi), lo)
        return (v - lo) / (hi - lo)

def __ss_errors(y_true, y_pred): return np.sum((y_true - y_pred) ** 2)
def __ss_tot(y_true): return np.sum((y_true - np.nanmean(y_true)) ** 2)

def _accuracy(y_true, y_pred, **_): return float(skm.accuracy_score(y_true, y_pred))
def _balanced_accuracy(y_true, y_pred, **_): return float(skm.balanced_accuracy_score(y_true, y_pred))
def _f1_macro(y_true, y_pred, **_): return float(skm.f1_score(y_true, y_pred, average="macro"))
def _f1_weighted(y_true, y_pred, **_): return float(skm.f1_score(y_true, y_pred, average="weighted"))
def _precision_macro(y_true, y_pred, **_): return float(skm.precision_score(y_true, y_pred, average="macro", zero_division=0))
def _recall_macro(y_true, y_pred, **_): return float(skm.recall_score(y_true, y_pred, average="macro", zero_division=0))
def _cohen_kappa(y_true, y_pred, **_): return float(skm.cohen_kappa_score(y_true, y_pred))
def _mcc(y_true, y_pred, **_): return float(skm.matthews_corrcoef(y_true, y_pred))
def _ari(y_true, y_pred, **_): return (float(skm.adjusted_rand_score(y_true, y_pred)) + 1.0)/2.0
def _nmi(y_true, y_pred, **_): return float(skm.normalized_mutual_info_score(y_true, y_pred))
def _ami(y_true, y_pred, **_): return float(skm.adjusted_mutual_info_score(y_true, y_pred))
def _homogeneity(y_true, y_pred, **_): return float(skm.homogeneity_score(y_true, y_pred))
def _completeness(y_true, y_pred, **_): return float(skm.completeness_score(y_true, y_pred))
def _v_measure(y_true, y_pred, **_): return float(skm.v_measure_score(y_true, y_pred))
def _fmi(y_true, y_pred, **_): return float(skm.fowlkes_mallows_score(y_true, y_pred))
def _smape(y_true, y_pred, **_): return 2 * np.mean(np.abs(y_true - y_pred) / (np.abs(y_true) + np.abs(y_pred) + 1e-8))
def _rmse(y_true, y_pred, **_): return np.sqrt(np.mean((y_true-y_pred)**2))
def _mae(y_true, y_pred, **_): return np.mean(np.abs(y_true-y_pred))
def _r2(y_true, y_pred, **_): return 1 - __ss_errors(y_true, y_pred) / __ss_tot(y_true)
def _maape(y_true, y_pred): return np.mean(np.arctan2(np.abs(y_true - y_pred), np.abs(y_true)))
def _imaape(y_true, y_pred): return 1 - _maape(y_true, y_pred)


def _purity(y_true, y_pred, **_): 
    C = skm.cluster.contingency_matrix(y_true, y_pred, sparse=False)
    return float(np.sum(np.max(C, axis=0)) / np.sum(C))

def _roc_auc_macro(y_true, y_score=None, classes=None, **_):
    if y_score is None:
        raise ValueError("roc_auc_macro requires scores")
    if _is_binary(y_true):
        s = _binary_pos_scores(y_true, y_score, classes)
        return float(skm.roc_auc_score(y_true, s))
    return float(skm.roc_auc_score(y_true, y_score, average="macro", multi_class="ovr"))

def _pr_auc_macro(y_true, y_score=None, classes=None, **_):
    if y_score is None:
        raise ValueError("pr_auc_macro requires scores")
    if _is_binary(y_true):
        s = _binary_pos_scores(y_true, y_score, classes)
        return float(skm.average_precision_score(y_true, s))
    if classes is None:
        classes = np.unique(y_true)
    Y = label_binarize(y_true, classes=classes)
    return float(skm.average_precision_score(Y, y_score, average="macro"))

def _treshold_f1(y_true, y_score, threshold=None, **_):
    if threshold is None: 
        threshold = np.mean(y_score)+3*np.std(y_score)
    y_pred = (y_score >= threshold).astype(int)
    return float(skm.f1_score(y_true, y_pred))


_REGISTRY: Dict[str, MetricSpec] = {
    "accuracy":             MetricSpec(_accuracy,           needs_score=False, value_range=(0.0, 1.0)),
    "balanced_accuracy":    MetricSpec(_balanced_accuracy,  needs_score=False, value_range=(0.0, 1.0)),
    "f1_macro":             MetricSpec(_f1_macro,           needs_score=False, value_range=(0.0, 1.0)),
    "f1_weighted":          MetricSpec(_f1_weighted,        needs_score=False, value_range=(0.0, 1.0)),
    "precision_macro":      MetricSpec(_precision_macro,    needs_score=False, value_range=(0.0, 1.0)),
    "recall_macro":         MetricSpec(_recall_macro,       needs_score=False, value_range=(0.0, 1.0)),
    "cohen_kappa":          MetricSpec(_cohen_kappa,        needs_score=False, value_range=(-1.0, 1.0)),
    "mcc":                  MetricSpec(_mcc,                needs_score=False, value_range=(-1.0, 1.0)),
    "roc_auc_macro":        MetricSpec(_roc_auc_macro,      needs_score=True,  value_range=(0.0, 1.0)),
    "pr_auc_macro":         MetricSpec(_pr_auc_macro,       needs_score=True,  value_range=(0.0, 1.0)),
    "f1":                   MetricSpec(_treshold_f1,        needs_score=True, value_range=(0.0, 1.0)),
    "roc_auc":              MetricSpec(_roc_auc_macro,      needs_score=True,  value_range=(0.0, 1.0)),
    "pr_auc":               MetricSpec(_pr_auc_macro,       needs_score=True,  value_range=(0.0, 1.0)),
    "ari":                  MetricSpec(_ari,                needs_score=False, value_range=(0.0, 1.0)),
    "nmi":                  MetricSpec(_nmi,                needs_score=False, value_range=(0.0, 1.0)),
    "ami":                  MetricSpec(_ami,                needs_score=False, value_range=(0.0, 1.0)),
    "homogeneity":          MetricSpec(_homogeneity,        needs_score=False, value_range=(0.0, 1.0)),
    "completeness":         MetricSpec(_completeness,       needs_score=False, value_range=(0.0, 1.0)),
    "v_measure":            MetricSpec(_v_measure,          needs_score=False, value_range=(0.0, 1.0)),
    "fmi":                  MetricSpec(_fmi,                needs_score=False, value_range=(0.0, 1.0)),
    "purity":               MetricSpec(_purity,             needs_score=False, value_range=(0.0, 1.0)),
    "maape":                MetricSpec(_maape,              needs_score=False, value_range=(0.0, 1.5)),
    "imaape":               MetricSpec(_imaape,             needs_score=False, value_range=(-0.5, 1.0)),
    "smape":                MetricSpec(_smape,              needs_score=False, value_range=(0.0, 2.0)),
    "rmse":                 MetricSpec(_rmse,               needs_score=False, value_range=(0.0, np.inf)),
    "mae":                  MetricSpec(_mae,                needs_score=False, value_range=(0.0, np.inf)),
    "r2":                   MetricSpec(_r2,                needs_score=False, value_range=(-np.inf, 1)),
}


def get_metric(name: str) -> MetricSpec:
    key = name.lower()
    if key not in _REGISTRY:
        raise ValueError(f"Unknown metric '{name}'. Available: {sorted(_REGISTRY)}")
    return _REGISTRY[key]

def evaluate_metrics(
    names: Iterable[str],
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_score: Optional[np.ndarray] = None,
    *,
    classes: Optional[Sequence] = None,
    analytics: str = "classification",
    threshold: float = None,
) -> Dict[str, float]:
    out: Dict[str, float] = {}
    
    if analytics == "anomaly":
        y_score = MinMaxScaler(feature_range=(0,1)).fit_transform(y_score.reshape(-1,1)).ravel()

    for name in names:
        spec = get_metric(name)
        if spec.needs_score and y_score is None:
            raise ValueError(f"Metric '{name}' needs scores (predict_proba/decision_function).")
        if analytics in ("classification", "clustering"):
            val = spec.fn(y_true=y_true, y_pred=y_pred, y_score=y_score, classes=classes)
        elif analytics == "anomaly":
            val = spec.fn(y_true=y_true, y_score=y_score, threshold=threshold)
        elif analytics in ("forecasting", "regression"):
            val = spec.fn(y_true=y_true, y_pred=y_pred)
        else:
            raise ValueError(f"Unknown analytics task '{analytics}'.")
        out[name] = float(val)
    return out

def default_primary_regression_metric():
    return "imaape"

def default_report_regression_metrics():
    return ["imaape", "maape", "smape", "rmse", "mae", "r2"]

def default_primary_classification_metric():
    return "balanced_accuracy"

def default_report_classification_metrics():
    return ["accuracy", "balanced_accuracy", "precision_macro", "recall_macro", "f1_macro", "roc_auc_macro", "pr_auc_macro"]

def default_primary_anomaly_metric():
    return "pr_auc"

def default_report_anomaly_metrics():
    return ["roc_auc","pr_auc","f1"]

def default_primary_clustering_metric():
    return "ari"

def default_report_clustering_metrics():
    return ["ari", "nmi", "ami", "homogeneity", "completeness", "v_measure", "fmi", "purity"]

def default_primary_forecasting_metric():
    return "imaape"

def default_report_forecasting_metrics():
    return ["imaape", "maape", "smape", "rmse", "mae", "r2"]


def select_threshold_by(y_true, y_score, mode="f1"):
    """Pick a threshold on validation scores."""
    if y_score is None or len(y_score)==0: return None
    if mode == "f1":
        uniq = np.unique(y_score)
        best_t, best = None, -1.0
        for t in uniq:
            f1 = skm.f1_score(y_true, (y_score>=t).astype(int), zero_division=0)
            if f1 > best: best, best_t = f1, t
        return float(best_t)
    if mode == "youden":
        fpr, tpr, thr = skm.roc_curve(y_true, y_score)
        j = tpr - fpr
        return float(thr[int(np.argmax(j))])
    raise ValueError(f"Unknown threshold selection mode: {mode}")


def classification_metrics(y_true, y_pred) -> dict:
    return {"accuracy": float(skm.accuracy_score(y_true, y_pred)), 
            "f1_macro": skm.f1_score(y_true, y_pred, average="macro")}


def anomaly_metrics(y_true, scores) -> dict:
    s = np.asarray(scores, dtype=float)
    if s.size and (s.max() - s.min()) > 0:
        s = (s - s.min()) / (s.max() - s.min())
    prec, rec, _ = skm.precision_recall_curve(y_true, s)
    return {"auc_pr": float(skm.auc(rec, prec))}


def combine_fitness(avg_acc: float, avg_cr: float, alpha: float) -> float:
    """Scalarization: alpha * task_score + (1 - alpha) * (1 - 1/avg_cr)."""
    avg_acc, avg_cr = float(avg_acc), float(avg_cr)
    if not np.isfinite(avg_acc) or not np.isfinite(avg_cr) or avg_cr <= 0.0:
        raise ValueError(f"Non-finite fitness inputs: task_metric={avg_acc}, avg_cr={avg_cr}")
    return alpha * avg_acc + (1.0 - alpha) * (1.0 - 1.0 / avg_cr)
