import ctypes
import ctypes.util
import os
import subprocess
import numpy as np
import tempfile

# SZ3's CLI pre-allocates the compressed-output buffer proportional to the input
# size; for very short series the format's fixed header (~240 bytes) overflows it
# and the binary aborts with SIGABRT ("The buffer for compressed data is not
# large enough"). Empirically n=24 always fails and n=32 is borderline, so tiny
# series are edge-padded up to this length before compression and the
# reconstruction is truncated back. CR is computed against the ORIGINAL series
# bytes, so short series still honestly pay SZ3's header overhead (CR may be <1).
# For n >= SZ3_MIN_LEN the behavior is byte-identical to the unpadded version.
SZ3_MIN_LEN = 64

# # SZ3 is invoked as an external CLI (`sz3 -i <in> -z <z> -o <out>`), so unlike
# # every other compressor here it genuinely needs files on disk - there is no
# # in-process binding to hand buffers to. Put them on tmpfs when one is
# # available so the round trip stays in RAM: the CLI's own read/write cost is
# # unavoidable, but hitting a physical disk for it is not.
# _TMPFS = "/dev/shm"
# SZ3_TMP_DIR = _TMPFS if os.path.isdir(_TMPFS) and os.access(_TMPFS, os.W_OK) else None

# # This version of SZ compression was too slow. 
# def sz_compress_series_v1(series, error_bound: float):
#     """
#     Compress a single float64 time series with SZ3 CLI using ABS mode scaled by data range.
#     Returns (reconstructed:list[float], cr:float).
#     """
#     series = np.asanyarray(series, dtype=np.float64)
#     n = series.size
#     ts_min, ts_max = float(series.min()), float(series.max())
#     ts_range = max(abs(ts_max - ts_min), 1e-12)             # avoid 0-range
#     abs_eb = float(error_bound) * ts_range             # scale REL→ABS

#     # Pad tiny series (repeating the last value: stays inside [min,max] and
#     # compresses to almost nothing) only to clear SZ3's buffer allocation.
#     padded = series
#     if n < SZ3_MIN_LEN:
#         padded = np.concatenate([series, np.full(SZ3_MIN_LEN - n, series[-1])])

#     with tempfile.TemporaryDirectory(dir=SZ3_TMP_DIR, prefix="sz3_") as tmp:
#         in_dat  = os.path.join(tmp, "ts.dat")
#         out_sz  = os.path.join(tmp, "ts.sz")
#         out_dat = os.path.join(tmp, "ts.out")

#         # write input
#         with open(in_dat, "wb") as f:
#             f.write(padded.tobytes())

#         # compress
#         try:
#             subprocess.run(
#                 ["sz3", "-d", "-i", in_dat, "-z", out_sz, "-o", out_dat, "-M", "ABS", str(abs_eb), "-1", str(padded.size)],
#                 check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
#             )
#         except subprocess.CalledProcessError as e:
#             raise RuntimeError(
#                 f"SZ3 compression failed (n={n}, abs_eb={abs_eb}): {e}"
#             ) from e

#         # sizes for CR — original bytes over compressed bytes (padding excluded
#         # from the numerator; identical to getsize(in_dat) when no padding)
#         raw_size = n * series.itemsize
#         cmp_size = os.path.getsize(out_sz)
#         cr = round(raw_size / max(1, cmp_size), 2)

#         # read reconstruction
#         with open(out_dat, "rb") as f:
#             rec = np.frombuffer(f.read(), dtype=np.float64)

#     # ensure exact length & list type for downstream
#     if rec.size != padded.size:
#         raise ValueError("The size of both reconstructed and the original time series should be the same")
#     rec = rec[:n]

#     return rec.tolist(), cr


# ----------------------------------------------------------------------
# v2: the same SZ3 build, called in-process through its C API
# ----------------------------------------------------------------------
# `sz_compress_series` above shells out to the `sz3` CLI once per series, which
# costs ~3.8 ms of fork/exec/dynamic-link before any compression happens - at a
# 100-evaluation budget over thousands of series that dominates the runtime.
# libSZ3c.so ships with the same build and exposes the whole operation as two
# calls, so v2 binds it directly with ctypes: no subprocess, no temp files, and
# (deliberately) no version change, so a v1-vs-v2 difference can only come from
# the calling convention.

SZ3_LIB_PATH = os.environ.get(
    "SZ3_LIB_PATH", os.path.expanduser("~/opt/sz/lib/libSZ3c.so")
)

# sz3c.h
_SZ_DOUBLE = 1
_SZ_ABS = 0

_lib = None


def _sz3_lib():
    """Load libSZ3c.so once, with the argtypes the header declares.

    Declaring argtypes/restype is not optional: without them ctypes passes
    pointers as 32-bit ints on 64-bit builds and the call corrupts memory
    silently rather than failing.
    """
    global _lib
    if _lib is not None:
        return _lib
    if not os.path.exists(SZ3_LIB_PATH):
        raise ImportError(
            f"SZ3 C library not found at {SZ3_LIB_PATH}. Build SZ3 with the C API "
            "or set SZ3_LIB_PATH to libSZ3c.so."
        )
    lib = ctypes.CDLL(SZ3_LIB_PATH)
    # unsigned char *SZ_compress_args(int dataType, void *data, size_t *outSize,
    #     int errBoundMode, double absErrBound, double relBoundRatio,
    #     double pwrBoundRatio, size_t r5, r4, r3, r2, r1)
    lib.SZ_compress_args.restype = ctypes.POINTER(ctypes.c_ubyte)
    lib.SZ_compress_args.argtypes = [
        ctypes.c_int, ctypes.c_void_p, ctypes.POINTER(ctypes.c_size_t),
        ctypes.c_int, ctypes.c_double, ctypes.c_double, ctypes.c_double,
        ctypes.c_size_t, ctypes.c_size_t, ctypes.c_size_t, ctypes.c_size_t, ctypes.c_size_t,
    ]
    # void *SZ_decompress(int dataType, unsigned char *bytes, size_t byteLength,
    #     size_t r5, r4, r3, r2, r1)
    lib.SZ_decompress.restype = ctypes.c_void_p
    lib.SZ_decompress.argtypes = [
        ctypes.c_int, ctypes.POINTER(ctypes.c_ubyte), ctypes.c_size_t,
        ctypes.c_size_t, ctypes.c_size_t, ctypes.c_size_t, ctypes.c_size_t, ctypes.c_size_t,
    ]
    _lib = lib
    return _lib


def sz_compress_series(series, error_bound: float):
    """
    Compress a single float64 time series with SZ3 via its C API (ABS mode,
    error bound scaled by data range). Returns (reconstructed:list[float], cr:float).
    """
    lib = _sz3_lib()
    series = np.ascontiguousarray(series, dtype=np.float64)
    n = series.size
    ts_min, ts_max = float(series.min()), float(series.max())
    ts_range = max(abs(ts_max - ts_min), 1e-12)             # avoid 0-range
    abs_eb = float(error_bound) * ts_range             # scale REL→ABS

    # Same short-series padding as v1: SZ3's fixed header can exceed its own
    # output buffer allocation for tiny inputs. Kept identical so v1/v2 remain
    # comparable on the short UCR series.
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
            # Copy out of the C buffer before freeing it.
            rec = np.ctypeslib.as_array(
                ctypes.cast(dec_ptr, ctypes.POINTER(ctypes.c_double)), shape=(padded.size,)
            ).copy()
        finally:
            _free(dec_ptr)
    finally:
        _free(comp_ptr)

    # CR against the ORIGINAL series bytes (padding excluded from the
    # numerator), exactly as v1 computes it.
    raw_size = n * series.itemsize
    cr = round(raw_size / max(1, cmp_size), 2)

    if rec.size != padded.size:
        raise ValueError("The size of both reconstructed and the original time series should be the same")
    return rec[:n].tolist(), cr


_libc = ctypes.CDLL(ctypes.util.find_library("c") or "libc.so.6")
_libc.free.argtypes = [ctypes.c_void_p]
_libc.free.restype = None


def _free(ptr):
    """SZ_compress_args/SZ_decompress return malloc'd buffers the caller owns;
    without this every objective evaluation leaks them."""
    _libc.free(ctypes.cast(ptr, ctypes.c_void_p))