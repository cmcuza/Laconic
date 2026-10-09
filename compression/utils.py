import json
import time
import numpy as np
import pandas as pd
import zstandard as zstd
from .backend import Backend
from collections import Counter
from typing import Any, Dict, List, NamedTuple, Tuple
from profiling import stage_clock


class BatchCR(NamedTuple):
    """Both compression-ratio aggregates for one compressed batch, one pass."""

    mean_cr: float
    n_elements: float
    n_inv: float

    @property
    def pooled_cr(self) -> float:
        if self.n_inv <= 0.0:
            raise ValueError("pooled_cr undefined: total compressed size is zero")
        return float(self.n_elements / self.n_inv)

    @staticmethod
    def combine(parts: List["BatchCR"]) -> "BatchCR":
        """Pool several batches (e.g. one per dimension) into one."""
        if not parts:
            raise ValueError("BatchCR.combine received no parts")
        return BatchCR(
            mean_cr=float(np.mean([p.mean_cr for p in parts])),
            n_elements=float(sum(p.n_elements for p in parts)),
            n_inv=float(sum(p.n_inv for p in parts)),
        )


def _batch_cr(lengths: List[int], ratios: List[float]) -> BatchCR:
    n = np.asarray(lengths, dtype=np.float64)
    cr = np.asarray(ratios, dtype=np.float64)
    if np.any(cr <= 0.0):
        raise ValueError("Non-positive per-series compression ratio in batch.")
    return BatchCR(mean_cr=float(np.mean(cr)), n_elements=float(n.sum()),
                   n_inv=float(np.sum(n / cr)))

def reconstruct_array(X, backend, params):
    """X: np.ndarray with shape (T,) or (T, D) Returns: (X_rec np.ndarray with same shape, avg_cr float)."""
    X = np.asarray(X)
    if X.ndim == 1:
        rec, cr = backend.apply(X.tolist(), params)
        return np.asarray(rec), float(cr)

    rec_cols, crs = [], []
    for d in range(X.shape[1]):
        r, c = backend.apply(X[:, d].tolist(), params)
        rec_cols.append(np.asarray(r))
        crs.append(float(c))
    X_rec = np.stack(rec_cols, axis=1)
    return X_rec, float(np.mean(crs))


def compute_average_params(df: pd.DataFrame) -> Dict[str, Any]:
    """Average best_params across folds (mean for numbers, mode for strings)."""
    all_params = [json.loads(params) for params in df["best_params"]]
    averaged_params = {}

    for key in all_params[0].keys():
        values = [params[key] for params in all_params]
        if isinstance(values[0], float):
            averaged_params[key] = sum(values) / len(values)
        elif isinstance(values[0], str):
            averaged_params[key] = Counter(values).most_common(1)[0][0]
        else:
            raise ValueError(f"Unsupported parameter type for key '{key}': {type(values[0])}")

    return averaged_params

def compress_and_decompress_batch(X: np.ndarray, backend: Backend, params: Dict[str, Any]) -> Tuple[np.ndarray, float]:
    """Compress and decompress each series in X with the given compressor params."""
    X_rec, stats = compress_and_decompress_batch_cr(X, backend, params)
    return X_rec, stats.mean_cr


def compress_and_decompress_batch_cr(
    X: np.ndarray, backend: Backend, params: Dict[str, Any]
) -> Tuple[np.ndarray, BatchCR]:
    """As `compress_and_decompress_batch`, but returning BOTH CR aggregates."""
    if len(X) == 0:
        raise ValueError("compress_and_decompress_batch received an empty batch.")
    rec_list, crs, lengths = [], [], []
    for series in X:
        if stage_clock._ENABLED:
            t0 = time.perf_counter()
        s = np.squeeze(series).copy().tolist()
        if stage_clock._ENABLED:
            stage_clock.tick("marshal", time.perf_counter() - t0)

        if stage_clock._ENABLED:
            t0 = time.perf_counter()
        rec, cr = backend.apply(s, params)
        if stage_clock._ENABLED:
            stage_clock.tick("codec", time.perf_counter() - t0)

        rec_list.append(rec)
        crs.append(float(cr))
        lengths.append(len(s))
        assert len(rec) == len(s)
    return np.asarray(rec_list, dtype=np.float64), _batch_cr(lengths, crs)

def zstd_baseline_cr(X: np.ndarray) -> float:
    """Zstd baseline CR over the given set (mean of per-series CRs)."""
    return zstd_baseline_cr_stats(X).mean_cr


def zstd_baseline_cr_stats(X: np.ndarray) -> BatchCR:
    """The zstd reference, with both aggregates."""
    cctx = zstd.ZstdCompressor(level=3)
    if X.shape[1] > 1:
        crs: List[float] = []
        lengths: List[int] = []
        for series in X:
            arr = np.squeeze(series).astype(np.float64)
            raw = arr.tobytes()
            crs.append(len(raw) / max(1, len(cctx.compress(raw))))
            lengths.append(int(arr.size))
        return _batch_cr(lengths, crs)
    arr = np.squeeze(X).astype(np.float64)
    raw = arr.tobytes()
    return _batch_cr([int(arr.size)], [len(raw) / max(1, len(cctx.compress(raw)))])