"""SerfXOR compression via the batched C++ binding.

``serfxor_compress_series`` used to call ``compression.lib.pyserf`` once per
value (one pybind11 crossing per element, on a boxed numpy scalar) - roughly
307 ns/value, 16.8-33.7x slower than driving the whole series through C++ in
one call (measured in `compression/lib/batch_serfxor/README.md`). It now
routes through ``compression.lib.batch_pyserf`` instead, mirroring
``scripts/benchmark_compression_throughput.py::serfxor_batch_halves`` (the
reference implementation - see that function rather than re-deriving this).

Why the per-value loop had to decompress each block immediately, and why the
batched wrapper does not need to: `compression/lib/batch_serfxor/README.md`
("It also removes an aliasing trap"). Short version: upstream's ``get()`` binds
with ``py::return_value_policy::reference``, returning a reference into the
encoder's own buffer rather than an owned copy, so a pack held across further
encoder use silently becomes a different pack. The batched wrapper copies each
block into an owned ``py::bytes`` before the encoder is touched again, so packs
are safe to collect without interleaving compression and decompression.

Both ``serfxor`` and ``adaedge`` (which samples the ``serfxor`` arm) hard-depend
on the built ``compression/lib/batch_pyserf*.so`` - build it with
``compression/lib/batch_serfxor/build.sh`` (clones and builds a pinned Serf
checkout when ``SERF_ROOT`` is unset; see that script and its README for the
pinned commit). There is no fallback to the per-value binding: a missing .so
fails at import, not silently at 20x the intended cost.

``compression/lib/pyserf.so`` (the per-value binding) is still shipped and
still used by ``scripts/benchmark_compression_throughput.py --methods
serfxor_pyloop`` and by ``tests/test_serfxor_binding_equivalence.py``'s
in-test reference - it is what the batched path is checked against, not an
alternate production path.
"""
import math
import numpy as np

try:
    from compression.lib import batch_pyserf
except ImportError as exc:  # fail-fast: no best-effort fallback to the per-value binding
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
    """Compress a single float64 time series with SerfXOR.

    This is what used to be ``serfxor_compress_series_v2``: the windowed,
    digit-adjusted encoder. The original single-shot encoder was removed when
    the two were collapsed into one compressor - keeping both under names that
    differed by a suffix meant every results row, log path and plot label had
    to carry the distinction too, for a variant that was never competitive.

    Batched via ``compression.lib.batch_pyserf`` (see module docstring) -
    byte-identical to the per-value binding this replaced, which is why there
    is no behavioural note here beyond the return contract.
    """
    series = np.asanyarray(series, dtype=np.float64)
    n = series.size
    ts_min, ts_max = float(series.min()), float(series.max())
    ts_range = max(abs(ts_max - ts_min), 1e-12)             # avoid 0-range
    abs_eb = float(error_bound) * ts_range             # scale REL→ABS

    adjust = serf_adjust_digit(ts_min, ts_max)

    packs = batch_pyserf.compress(series, SERF_WINDOW_SIZE, abs_eb, adjust, SERF_BLOCK_SIZE)
    cmp_size = sum(len(pack) for pack in packs)
    rec = batch_pyserf.decompress(packs, adjust).tolist()

    raw_size = n * series.itemsize
    cr = round(raw_size / max(1, cmp_size), 2)

    # ensure exact length & list type for downstream
    if len(rec) != n:
        raise ValueError("The size of both reconstructed and the original time series should be the same")

    return rec, cr