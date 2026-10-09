"""SerfXOR compression via the batched C++ binding."""
import math
import numpy as np

try:
    from compression.lib import batch_pyserf
except ImportError as exc:
    raise ImportError(
        "compression.lib.batch_pyserf is not built. serfxor (and adaedge, which "
        "samples the serfxor arm) require it - build it with "
        "`bash compression/lib/batch_serfxor/build.sh` (clones and builds a pinned "
        "Serf checkout automatically when SERF_ROOT is unset; set SERF_ROOT to reuse "
        "an existing checkout, and PYTHON_BIN if the build should target a different "
        "interpreter than the one running this import). See "
        "compression/lib/batch_serfxor/README.md."
    ) from exc

SERF_WINDOW_SIZE = 1000
SERF_BLOCK_SIZE = 1000

def serf_adjust_digit(ts_min: float, ts_max: float) -> int:
    span = math.floor(ts_max) - math.floor(ts_min) + 1
    u = math.ceil(math.log2(span))
    return max(0, (1 << u) - int(math.floor(ts_min)))

def serfxor_compress_series(series, error_bound: float):
    """Compress a single float64 time series with SerfXOR."""
    series = np.asanyarray(series, dtype=np.float64)
    n = series.size
    ts_min, ts_max = float(series.min()), float(series.max())
    ts_range = max(abs(ts_max - ts_min), 1e-12)
    abs_eb = float(error_bound) * ts_range

    adjust = serf_adjust_digit(ts_min, ts_max)

    packs = batch_pyserf.compress(series, SERF_WINDOW_SIZE, abs_eb, adjust, SERF_BLOCK_SIZE)
    cmp_size = sum(len(pack) for pack in packs)
    rec = batch_pyserf.decompress(packs, adjust).tolist()

    raw_size = n * series.itemsize
    cr = round(raw_size / max(1, cmp_size), 2)

    if len(rec) != n:
        raise ValueError("The size of both reconstructed and the original time series should be the same")

    return rec, cr