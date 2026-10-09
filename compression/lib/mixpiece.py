import numpy as np
import jpype

JAR_PATH = "compression/lib/mixpiece.jar"

MixPiece = None

# --- globalMinB: upstream's cross-call state leak ----------------------------
# `io.github.xkitsios.MixPiece` is a static singleton. `compress()` resets
# `epsilon` and `lastTimeStamp`, and `merge()` reassigns the three segment
# lists, but `globalMinB` is a RUNNING MINIMUM seeded from whatever the previous
# call left behind (bytecode: getstatic globalMinB -> Math.min -> putstatic), so
# across a long-lived JVM it only ever ratchets down. `decompress()` writes it
# too, and this wrapper calls both per invocation.
#
# Every segment's intercept is encoded as an offset from it, under variable-byte
# encoding, so a value dragged down by earlier UNRELATED series inflates the
# payload: measured 49.26 -> 44.57 on one clustering/Wine series, a constant
# 4-byte varint boundary. Reconstruction is unaffected - bit-identical, error
# bound respected - so only the compression ratio moves.
#
# Upstream ships as a CLI (one compress per process), which is why it never
# surfaced there. See docs/COMPRESSION_RATIO_METRICS.md.
#
# ROLLOUT: this must stay False while a sweep is in flight. `run_suite.py`
# spawns a fresh subprocess per cell, so flipping it mid-sweep would leave one
# sweep half-measured under the bug and half under the fix.
#
# Turned on 2026-09-14, after the last sweep exited and after every stored
# mixpiece/adaedge CR - in BOTH the source and the consensus trees, all four
# tasks, all three seeds - had been re-measured with the reset in force by
# a one-off repair pass. Everything on disk and everything produced from here
# on therefore shares one binding behaviour, so no repair tooling is kept.
RESET_GLOBAL_MIN_B = True

_global_min_b_field = None


def state_is_reset() -> bool:
    """Whether the globalMinB reset is in force, i.e. whether CR is reproducible."""
    return RESET_GLOBAL_MIN_B


def _reset_global_min_b() -> None:
    """Seed globalMinB to Integer.MAX_VALUE so the running min sees only this series.

    Integer.MAX_VALUE, not 0: 0 is merely the JVM's default for the static int,
    so a "fresh process" run computes min(0, trueMin) and clamps at zero whenever
    every intercept is positive. MAX_VALUE gives the value the algorithm means.
    """
    global _global_min_b_field
    if _global_min_b_field is None:
        field = jpype.JClass("io.github.xkitsios.MixPiece").class_.getDeclaredField("globalMinB")
        field.setAccessible(True)
        _global_min_b_field = field
    _global_min_b_field.setInt(None, 2 ** 31 - 1)

def mixpiece_compress_series(series, error_bound: float):
    """
    Compress a single float64 time series with MixPiece
    """
    global MixPiece
    series = np.asanyarray(series, dtype=np.float64)
    n = series.size
    ts_min, ts_max = float(series.min()), float(series.max())
    ts_range = max(abs(ts_max - ts_min), 1e-12)             # avoid 0-range
    abs_eb = float(error_bound) * ts_range             # scale REL→ABS

    if not jpype.isJVMStarted():
        jpype.startJVM(classpath=[JAR_PATH])
    if MixPiece is None:
        MixPiece = jpype.JClass("io.github.xkitsios.MixPieceBinding")

    if RESET_GLOBAL_MIN_B:
        _reset_global_min_b()

    # The binding is `compress(double[], double)`. Handing it the ndarray lets
    # jpype bulk-transfer the buffer; `series.tolist()` built a 100K-element
    # Python list that jpype then converted element by element. Measured on 100K
    # values: 19.4 ms -> 5.6 ms, with byte-identical output (it is the same
    # doubles either way), so no stored result changes.
    payload = MixPiece.compress(series, abs_eb)
    mx_decomp = MixPiece.decompress(payload)

    # Sizes are computed in memory. Both operands are byte-identical to what
    # the previous version wrote to a temp file and then stat()'d:
    # series.tobytes() is exactly n * itemsize bytes, and len(payload) is the
    # size of the same buffer that was written verbatim. Neither file was ever
    # read back - only their sizes were - so the round trip through disk cost
    # two writes per objective evaluation and changed nothing.
    raw_size = n * series.itemsize
    cmp_size = len(payload)
    cr = round(raw_size / max(1, cmp_size), 2)

    # ensure exact length & list type for downstream
    if len(mx_decomp) != n:
        raise ValueError("The size of both reconstructed and the original time series should be the same")

    return mx_decomp, cr
