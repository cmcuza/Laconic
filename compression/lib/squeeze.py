import ctypes
import ctypes.util
import os
import subprocess
import numpy as np
import tempfile

SZ3_MIN_LEN = 64


SZ3_LIB_PATH = os.environ.get(
    "SZ3_LIB_PATH", os.path.expanduser("~/opt/sz/lib/libSZ3c.so")
)

_SZ_DOUBLE = 1
_SZ_ABS = 0

_lib = None


def _sz3_lib():
    global _lib
    if _lib is not None:
        return _lib
    if not os.path.exists(SZ3_LIB_PATH):
        raise ImportError(
            f"SZ3 C library not found at {SZ3_LIB_PATH}. Build SZ3 with the C API "
            "or set SZ3_LIB_PATH to libSZ3c.so."
        )
    lib = ctypes.CDLL(SZ3_LIB_PATH)
    lib.SZ_compress_args.restype = ctypes.POINTER(ctypes.c_ubyte)
    lib.SZ_compress_args.argtypes = [
        ctypes.c_int, ctypes.c_void_p, ctypes.POINTER(ctypes.c_size_t),
        ctypes.c_int, ctypes.c_double, ctypes.c_double, ctypes.c_double,
        ctypes.c_size_t, ctypes.c_size_t, ctypes.c_size_t, ctypes.c_size_t, ctypes.c_size_t,
    ]
    lib.SZ_decompress.restype = ctypes.c_void_p
    lib.SZ_decompress.argtypes = [
        ctypes.c_int, ctypes.POINTER(ctypes.c_ubyte), ctypes.c_size_t,
        ctypes.c_size_t, ctypes.c_size_t, ctypes.c_size_t, ctypes.c_size_t, ctypes.c_size_t,
    ]
    _lib = lib
    return _lib


def sz_compress_series(series, error_bound: float):
    """Compress one float64 series with SZ3 (ABS mode, bound scaled by data range)."""
    lib = _sz3_lib()
    series = np.ascontiguousarray(series, dtype=np.float64)
    n = series.size
    ts_min, ts_max = float(series.min()), float(series.max())
    ts_range = max(abs(ts_max - ts_min), 1e-12)
    abs_eb = float(error_bound) * ts_range

    padded = series
    if n < SZ3_MIN_LEN:
        padded = np.ascontiguousarray(
            np.concatenate([series, np.full(SZ3_MIN_LEN - n, series[-1])])
        )

    out_size = ctypes.c_size_t(0)
    comp_ptr = lib.SZ_compress_args(
        _SZ_DOUBLE, padded.ctypes.data_as(ctypes.c_void_p), ctypes.byref(out_size),
        _SZ_ABS, abs_eb, 0.0, 0.0,
        0, 0, 0, 0, padded.size,
    )
    if not comp_ptr:
        raise RuntimeError(f"SZ3 compression failed (n={n}, abs_eb={abs_eb})")

    try:
        cmp_size = int(out_size.value)
        dec_ptr = lib.SZ_decompress(
            _SZ_DOUBLE, comp_ptr, cmp_size, 0, 0, 0, 0, padded.size
        )
        if not dec_ptr:
            raise RuntimeError(f"SZ3 decompression failed (n={n}, abs_eb={abs_eb})")
        try:
            rec = np.ctypeslib.as_array(
                ctypes.cast(dec_ptr, ctypes.POINTER(ctypes.c_double)), shape=(padded.size,)
            ).copy()
        finally:
            _free(dec_ptr)
    finally:
        _free(comp_ptr)

    raw_size = n * series.itemsize
    cr = round(raw_size / max(1, cmp_size), 2)

    if rec.size != padded.size:
        raise ValueError("The size of both reconstructed and the original time series should be the same")
    return rec[:n].tolist(), cr


_libc = ctypes.CDLL(ctypes.util.find_library("c") or "libc.so.6")
_libc.free.argtypes = [ctypes.c_void_p]
_libc.free.restype = None


def _free(ptr):
    _libc.free(ctypes.cast(ptr, ctypes.c_void_p))