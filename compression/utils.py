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
    """Both compression-ratio aggregates for one compressed batch, one pass.

    ``mean_cr`` is the historical metric: the arithmetic mean of the per-series
    ratios. ``pooled_cr`` pools the sizes instead,

        CR_pooled = 64 * sum_i(n_i) / sum_i(c_i)

    which, since every backend defines ``cr_i = 64*n_i / c_i`` (raw_size is
    ``size * itemsize``, 8 bytes per element), equals ``sum(n_i) / sum(n_i/cr_i)``
    - the LENGTH-WEIGHTED HARMONIC MEAN of the same per-series ratios. By AM-HM
    it is never larger than ``mean_cr``, and the gap grows with the dispersion of
    the per-series ratios.

    ``n_elements`` and ``n_inv`` are kept so batches can be COMBINED exactly:
    a pooled ratio is not an average of pooled ratios, so regression - which
    compresses each dimension as its own batch - has to sum these two and divide
    once, not average the per-dimension pooled values.
    """

    mean_cr: float
    n_elements: float      # sum n_i
    n_inv: float           # sum n_i / cr_i, i.e. sum c_i / 64

    @property
    def pooled_cr(self) -> float:
        if self.n_inv <= 0.0:
            raise ValueError("pooled_cr undefined: total compressed size is zero")
        return float(self.n_elements / self.n_inv)

    @staticmethod
    def combine(parts: List["BatchCR"]) -> "BatchCR":
        """Pool several batches (e.g. one per dimension) into one.

        ``mean_cr`` is the mean of the parts' means, reproducing the runners'
        existing double-averaging exactly; the pooled side sums the raw
        quantities, which is what makes it a true pooled ratio over everything.
        """
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
    """
    X: np.ndarray with shape (T,) or (T, D)
    Returns: (X_rec np.ndarray with same shape, avg_cr float)
    """
    X = np.asarray(X)
    if X.ndim == 1:
        rec, cr = backend.apply(X.tolist(), params)
        return np.asarray(rec), float(cr)

    # 2D: compress each column
    rec_cols, crs = [], []
    for d in range(X.shape[1]):
        r, c = backend.apply(X[:, d].tolist(), params)
        rec_cols.append(np.asarray(r))
        crs.append(float(c))
    X_rec = np.stack(rec_cols, axis=1)
    return X_rec, float(np.mean(crs))


def compute_average_params(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Compute the average of the best_params across folds.
    - For numerical values, compute the mean.
    - For string values, compute the mode (most common value).
    """
    all_params = [json.loads(params) for params in df["best_params"]]
    averaged_params = {}

    # Iterate over all keys in the parameter dictionaries
    for key in all_params[0].keys():
        values = [params[key] for params in all_params]
        if isinstance(values[0], float):  # If the value is a float, compute the mean
            averaged_params[key] = sum(values) / len(values)
        elif isinstance(values[0], str):  # If the value is a string, compute the mode
            averaged_params[key] = Counter(values).most_common(1)[0][0]
        else:
            raise ValueError(f"Unsupported parameter type for key '{key}': {type(values[0])}")

    return averaged_params

def compress_and_decompress_batch(X: np.ndarray, backend: Backend, params: Dict[str, Any]) -> Tuple[np.ndarray, float]:
    """
    Compress and decompress each series in X with the given compressor params.
    Expects X as an array of shape (n_samples, ...) where each sample squeezes to 1D.
    Returns (X_rec: (n_samples, series_len), avg_cr: float)
    """
    X_rec, stats = compress_and_decompress_batch_cr(X, backend, params)
    return X_rec, stats.mean_cr


def compress_and_decompress_batch_cr(
    X: np.ndarray, backend: Backend, params: Dict[str, Any]
) -> Tuple[np.ndarray, BatchCR]:
    """As `compress_and_decompress_batch`, but returning BOTH CR aggregates.

    The two-value wrapper above is kept and left byte-identical because it is
    what `experiments/objectives.py` calls inside the optimization loop: the
    searched objective must keep scoring on the metric the stored results were
    optimized under, so nothing here may change what it returns.
    """
    if len(X) == 0:
        raise ValueError("compress_and_decompress_batch received an empty batch.")
    rec_list, crs, lengths = [], [], []
    for series in X:
        # Per-series loop seam. A @contextmanager costs ~1us per entry even
        # when its body does nothing, and this fires 2 x n_series x D times
        # per evaluation (not once) - see profiling/stage_clock.py's module
        # docstring - so a bare `if _ENABLED:` guard + tick() is used instead
        # of stage_clock.stage(), testing the module global directly rather
        # than through a function call.
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
    """The zstd reference, with both aggregates.

    Carried alongside the pipeline's own CR because `cr_improvement_x` divides
    one by the other: comparing a pooled pipeline ratio against a mean-of-ratios
    baseline would mix the two definitions in a single number.
    """
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