from __future__ import annotations
from typing import List, Tuple, Dict
import compression.lib.laconic as laconic
import numpy as np
from compression.lib.laconic import Method
import struct
import math
import zstandard as zstd

VW_AREA_SCALE = 25.0
VW_AREA_EXPONENT = 1.4

DFT_HILL_CEILING = 0.92
DFT_HILL_SCALE = 0.03
DFT_HILL_EXPONENT = 1.4
DFT_TAIL_CEILING = 1.0
DFT_TAIL_EXPONENT = 3.0
DFT_CALIBRATION_ERROR_MINIMUM = 0.001
DFT_CALIBRATION_ERROR_MAXIMUM = 0.15

COEFFICIENT_ZERO_TOLERANCE = 1e-15

def _dft_discard_fraction(relative_error: float) -> float:
    error = float(relative_error)
    z = (error / DFT_HILL_SCALE) ** DFT_HILL_EXPONENT
    base_discard = DFT_HILL_CEILING * z / (1.0 + z)
    position = min(
        1.0,
        max(
            0.0,
            (error - DFT_CALIBRATION_ERROR_MINIMUM)
            / (
                DFT_CALIBRATION_ERROR_MAXIMUM
                - DFT_CALIBRATION_ERROR_MINIMUM
            ),
        ),
    )
    discard = base_discard + (
        DFT_TAIL_CEILING - base_discard
    ) * position**DFT_TAIL_EXPONENT
    return min(1.0, max(0.0, discard))


def create_abs_error_bound(x: list, error_bound: float) -> Dict[str, float]:
    ts_min, ts_max = float(np.min(x)), float(np.max(x))
    ts_range = max(abs(ts_max - ts_min), 1e-5)
    return {"abs_error_bound" : ts_range * error_bound}

def create_dft_config(x: list, keep_ratio: float) -> Dict[str, int]:
    return {"number_of_coefficients": int(max(2, len(x) * keep_ratio))}

def create_dft_valid_config(x: list, error_bound: float) -> Dict[str, int]:
    number_of_bins = len(x) // 2 + 1
    discard = _dft_discard_fraction(error_bound)
    kept = math.ceil(number_of_bins * (1.0 - discard))
    return {
        "number_of_coefficients": int(
            max(2, min(number_of_bins, kept))
        )
    }

def create_vw_config(x: list, error_bound: float) -> Dict[str, float]:
    ts_min, ts_max = float(np.min(x)), float(np.max(x))
    ts_range = max(abs(ts_max - ts_min), 1e-5)
    return {"area_under_curve_error": ts_range * (error_bound)}

def create_vw_valid_config(x: list, error_bound: float) -> Dict[str, float]:
    ts_min, ts_max = float(np.min(x)), float(np.max(x))
    ts_range = max(abs(ts_max - ts_min), 1e-5)
    relative_error = VW_AREA_SCALE * float(error_bound) ** VW_AREA_EXPONENT
    return {"area_under_curve_error": ts_range * relative_error}


def create_decimal_precision_camel(decimal_precision: float) -> Dict[str, int]:
    return {"decimal_precision": int(-np.log10(decimal_precision).clip(1, 4))}

def create_decimal_precision_buff(decimal_precision: float) -> Dict[str, int]:
    return {"decimal_precision": int(max(1, -np.log10(decimal_precision).clip(1, 10)))}


METHODS:  Dict[str, Dict[str, callable]] = {
    "PoorMansCompressionMean": {
        "compress": lambda x, e: laconic.compress(x, Method.PoorMansCompressionMean, create_abs_error_bound(x, e)),
        "extract":  lambda comp: laconic.extract(comp),
        "rebuild":  lambda tmps, coeffs: laconic.rebuild(tmps, coeffs, Method.PoorMansCompressionMean),
        "decompress": lambda comp: laconic.decompress(comp),
    },
    "SwingFilter": {
        "compress": lambda x, e: laconic.compress(x, Method.SwingFilter, create_abs_error_bound(x, e)),
        "extract":  lambda comp: laconic.extract(comp),
        "rebuild":  lambda tmps, coeffs: laconic.rebuild(tmps, coeffs, Method.SwingFilter),
        "decompress": lambda comp: laconic.decompress(comp),
    },
    "SlideFilter": {
        "compress": lambda x, e: laconic.compress(x, Method.SlideFilter, create_abs_error_bound(x, e)),
        "extract":  lambda comp: laconic.extract(comp),
        "rebuild":  lambda tmps, coeffs: laconic.rebuild(tmps, coeffs, Method.SlideFilter),
        "decompress": lambda comp: laconic.decompress(comp),
    },
    "VW": {
        "compress": lambda x, e: laconic.compress(x, Method.VisvalingamWhyatt, create_vw_valid_config(x, e)),
        "extract":  lambda comp: laconic.extract(comp),
        "rebuild":  lambda tmps, coeffs: laconic.rebuild(tmps, coeffs, Method.VisvalingamWhyatt),
        "decompress": lambda comp: laconic.decompress(comp),
    },
    "MixPiece":{
        "compress": lambda x, e: laconic.compress(x, Method.MixPiece, create_abs_error_bound(x, e)),
        "extract":  lambda comp: laconic.extract(comp),
        "rebuild":  lambda tmps, coeffs: laconic.rebuild(tmps, coeffs, Method.MixPiece),
        "decompress": lambda comp: laconic.decompress(comp),
    },
    "DFT": {
        "compress":   lambda x, e: laconic.compress(x, Method.DiscreteFourierTransform, create_dft_valid_config(x, e)),
        "extract":    lambda comp: laconic.extract(comp),
        "rebuild":    lambda tmps, coeffs: laconic.rebuild(tmps, coeffs, Method.DiscreteFourierTransform),
        "decompress": lambda comp: laconic.decompress(comp),
    },
    "BitPackedQuantization": {
        "compress": lambda x, e: laconic.compress(x, Method.BitPackedQuantization, create_abs_error_bound(x, e)),
        "decompress": lambda comp: laconic.decompress(comp),
    },
    "BitPackedBUFF": {
        "compress": lambda x, e: laconic.compress(x, Method.BitPackedBUFF, create_decimal_precision_buff(e)),
        "decompress": lambda comp: laconic.decompress(comp),
    },
    "SerfQT":{
        "compress": lambda x, e: laconic.compress(x, Method.SerfQT, create_abs_error_bound(x, e)),
        "decompress": lambda comp: laconic.decompress(comp),
    },
    "SerfXOR":{
            "compress": lambda x, e: laconic.compress(x, Method.SerfXOR, create_abs_error_bound(x, e)),
            "decompress": lambda comp: laconic.decompress(comp),
    },
    "MacaqueS": {
        "compress": lambda x, e: laconic.compress(x, Method.MacaqueS, create_abs_error_bound(x, e)),
        "decompress": lambda comp: laconic.decompress(comp),
    },
    "MacaqueV": {
        "compress": lambda x, e: laconic.compress(x, Method.MacaqueV, create_abs_error_bound(x, e)),
        "decompress": lambda comp: laconic.decompress(comp),
    },
    "Camel": {
        "compress": lambda x, e: laconic.compress(x, Method.Camel, create_decimal_precision_camel(e)),
        "decompress": lambda comp: laconic.decompress(comp),
    },
    "DeltaEncoding": {
        "compress": lambda x: laconic.compress_indices(x, Method.DeltaEncoding),
        "decompress": lambda comp: laconic.decompress_indices(comp),
    },
    "DeltatoDeltaPFOREncoding": {
        "compress": lambda x: laconic.compress_indices(x, Method.DeltatoDeltaPFOREncoding),
        "decompress": lambda comp: laconic.decompress_indices(comp),
    },
    "DeltaEliasGammaEncoding": {
        "compress": lambda x: laconic.compress_indices(x, Method.DeltaEliasGammaEncoding),
        "decompress": lambda comp: laconic.decompress_indices(comp),
    },
    "DeltaFORPFOREncoding": {
        "compress": lambda x: laconic.compress_indices(x, Method.DeltaFORPFOREncoding),
        "decompress": lambda comp: laconic.decompress_indices(comp),
    },
}


def normalize_bytes(byte_list: list) -> bytes:
    """Normalize a list of bytes so that its values are scaled to be between 0 and 256."""
    if not byte_list:
        return bytes()
    arr = np.array(byte_list, dtype=np.uint8)
    min_val = arr.min()
    max_val = arr.max()
    if max_val == min_val:
        return bytes([0] * len(arr))
    norm_arr = ((arr - min_val) * 256 / (max_val - min_val)).clip(0, 255).astype(np.uint8)
    return norm_arr.tobytes()


def tersets_compress_series(series: List[float], params: dict) -> Tuple[List[float], float]:
    """Compress with a (logical, coefficient, indices) pipeline; return (reconstruction, cr)."""
    logical_method = params["logical_method"]
    coefficients_method = params["coefficient_method"]
    indices_method = params["indices_method"]
    logical_method_error = params["logical_method_error"]
    coefficients_method_error = params["coefficient_method_error"]

    np_series = np.asanyarray(series, dtype=np.float64)

    logical_pipeline = METHODS[logical_method]
    compressed_primary = logical_pipeline["compress"](np_series, logical_method_error)
    times, coeffs = logical_pipeline["extract"](compressed_primary)

    coeffs = np.where(
        np.abs(coeffs) < COEFFICIENT_ZERO_TOLERANCE,
        0.0,
        coeffs,
    )

    comp_coeffs = METHODS[coefficients_method]["compress"](coeffs, coefficients_method_error)

    comp_times  = METHODS[indices_method]["compress"](times)

    payload = struct.pack("f", len(comp_times)) + bytes(comp_times) + bytes(comp_coeffs)

    raw_size = np_series.size * np_series.itemsize
    cmp_size = len(payload)
    cr = raw_size / max(1, cmp_size)

    zstd_cmp_size = len(zstd.ZstdCompressor().compress(payload))
    zstd_cr = raw_size / max(1, zstd_cmp_size)
    if zstd_cr > cr:
        cr = zstd_cr

    rec_coeffs = np.asarray(METHODS[coefficients_method]["decompress"](comp_coeffs), np.float64)

    rec_times = np.asarray(METHODS[indices_method]["decompress"](comp_times), np.uint64)
    
    try:
        rebuilt = logical_pipeline["rebuild"](rec_times, rec_coeffs)
    except RuntimeError as e:
        print(f"Error rebuilding logical pipeline: {params}")
        raise e
        

    reconstruction = laconic.decompress(rebuilt)
    if len(reconstruction) != len(series):
        reconstruction = reconstruction[:len(series)]
    return reconstruction, cr
