import numpy as np
import jpype

JAR_PATH = "compression/lib/mixpiece.jar"

MixPiece = None

RESET_GLOBAL_MIN_B = True

_global_min_b_field = None


def state_is_reset() -> bool:
    """Whether the globalMinB reset is in force, i.e. whether CR is reproducible."""
    return RESET_GLOBAL_MIN_B


def _reset_global_min_b() -> None:
    global _global_min_b_field
    if _global_min_b_field is None:
        field = jpype.JClass("io.github.xkitsios.MixPiece").class_.getDeclaredField("globalMinB")
        field.setAccessible(True)
        _global_min_b_field = field
    _global_min_b_field.setInt(None, 2 ** 31 - 1)

def mixpiece_compress_series(series, error_bound: float):
    """Compress a single float64 time series with MixPiece."""
    global MixPiece
    series = np.asanyarray(series, dtype=np.float64)
    n = series.size
    ts_min, ts_max = float(series.min()), float(series.max())
    ts_range = max(abs(ts_max - ts_min), 1e-12)
    abs_eb = float(error_bound) * ts_range

    if not jpype.isJVMStarted():
        jpype.startJVM(classpath=[JAR_PATH])
    if MixPiece is None:
        MixPiece = jpype.JClass("io.github.xkitsios.MixPieceBinding")

    if RESET_GLOBAL_MIN_B:
        _reset_global_min_b()

    payload = MixPiece.compress(series, abs_eb)
    mx_decomp = MixPiece.decompress(payload)

    raw_size = n * series.itemsize
    cmp_size = len(payload)
    cr = round(raw_size / max(1, cmp_size), 2)

    if len(mx_decomp) != n:
        raise ValueError("The size of both reconstructed and the original time series should be the same")

    return mx_decomp, cr
