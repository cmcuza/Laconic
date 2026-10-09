from __future__ import annotations
import numpy as np
from typing import Protocol, Dict, Tuple, Any, List, Type

from .lib.tersets import tersets_compress_series, METHODS
from .lib.squeeze import sz_compress_series
from .lib.mixpiece import mixpiece_compress_series
from .lib.serfxor import serfxor_compress_series

def clamp(value, min_val, max_val):
    return float(max(min_val, min(value, max_val)))


def _get_bounds(raw_bound: Any, field_name: str) -> Tuple[float, float]:
    if isinstance(raw_bound, (list, tuple)) and len(raw_bound) == 2:
        lo, hi = raw_bound
        return float(lo), float(hi)
    raise ValueError(f"Invalid bounds for '{field_name}': expected [lo, hi]. Got: {raw_bound}")


def _get_space_meta(space_meta: Any, field_name: str) -> Dict[str, Any]:
    if not isinstance(space_meta, dict):
        raise ValueError(f"Space definition for '{field_name}' must be a dict.")
    if "type" not in space_meta:
        raise ValueError(f"Space definition for '{field_name}' is missing required 'type' entry.")
    return space_meta


def _cast_by_space_type(value: float, space_meta: Dict[str, Any], field_name: str) -> Any:
    vtype = str(space_meta["type"]).lower()
    if vtype == "int":
        return int(round(value))
    if vtype == "float":
        return float(value)
    raise ValueError(f"Unsupported type '{space_meta['type']}' in space definition for '{field_name}'.")

class Backend(Protocol):
    def __init__(self, bounds: dict, params: dict | None = None): ...
    def bounds(self) -> Dict[str, Tuple[float, float]]: ...
    def params_from_vector(self, v: Dict[str, float]) -> Dict[str, Any]: ...
    def apply(self, series: List[float], params: Dict[str, Any]) -> Tuple[List[float], float]: ...

_BACKENDS: Dict[str, Type[Backend]] = {}

def register_backend(name: str, backend_cls: Type[Backend]) -> None:
    _BACKENDS[name] = backend_cls

def build_backend(name: str, bounds: dict, params: dict) -> Backend:
    try:
        backend_cls = _BACKENDS[name]
    except KeyError:
        raise ValueError(f"Unknown backend '{name}'. Registered: {list(_BACKENDS)}") from None
    return backend_cls(bounds, params)

class TerseTSBackend(Backend):
    names = ["laconic", "tersets_reduced"]

    def __init__(self, bounds: dict, params: dict):
        self._bounds = bounds
        if not isinstance(params, dict):
            raise ValueError("TerseTS backend expects params dict with 'methods' and 'space_definition'.")

        allowed = params.get("methods")
        if not isinstance(allowed, list):
            raise ValueError("TerseTS params must include 'methods' as a list.")

        space_definition = params.get("space_definition")
        if not isinstance(space_definition, dict):
            raise ValueError("TerseTS params must include 'space_definition' as a dict.")
        if "logical_method_error" not in space_definition:
            raise ValueError("Missing 'logical_method_error' entry in TerseTS space_definition.")
        if "coefficient_method_error" not in space_definition:
            raise ValueError("Missing 'coefficient_method_error' entry in TerseTS space_definition.")
        if "logical_method_error" not in bounds:
            raise ValueError("Missing 'logical_method_error' entry in TerseTS bounds.")
        if "coefficient_method_error" not in bounds:
            raise ValueError("Missing 'coefficient_method_error' entry in TerseTS bounds.")

        unknown = [m for m in allowed if m not in METHODS]
        if unknown:
            raise ValueError(f"TerseTS methods not implemented: {unknown}")
        if not allowed:
            raise ValueError("TerseTS 'params.methods' must list at least one method.")
        self._methods = allowed
        self._space_definition = space_definition

    def bounds(self) -> Dict[str, Tuple[float, float]]:
        return self._bounds

    def params_from_vector(self, pipeline_parameters: Dict[str, float]) -> Dict[str, Any]:
        logical_method_idx = int(round(pipeline_parameters["logical_method_index"]))
        coeff_method_idx = int(round(pipeline_parameters["coefficient_method_index"]))
        indices_method_idx = int(round(pipeline_parameters["indices_method_index"]))
        logical_method_error = pipeline_parameters["logical_method_error"]
        coefficient_method_error = pipeline_parameters["coefficient_method_error"]

        if logical_method_idx < 0 or logical_method_idx >= len(self._methods):
            raise ValueError(f"logical_method_index={logical_method_idx} out of range [0, {len(self._methods)-1}].")
        if coeff_method_idx < 0 or coeff_method_idx >= len(self._methods):
            raise ValueError(f"coefficient_method_index={coeff_method_idx} out of range [0, {len(self._methods)-1}].")

        logical_method = self._methods[logical_method_idx]
        coeff_method = self._methods[coeff_method_idx]
        indices_method = self._methods[indices_method_idx]

        logical_bounds = _get_bounds(self._bounds["logical_method_error"], "logical_method_error")
        coeff_bounds = _get_bounds(self._bounds["coefficient_method_error"], "coefficient_method_error")
        logical_value = clamp(logical_method_error, *logical_bounds)
        coeff_value = clamp(coefficient_method_error, *coeff_bounds)

        logical_meta = _get_space_meta(self._space_definition["logical_method_error"], "logical_method_error")
        coeff_meta = _get_space_meta(self._space_definition["coefficient_method_error"], "coefficient_method_error")

        return {
            "logical_method": logical_method,
            "coefficient_method": coeff_method,
            "indices_method": indices_method,
            "logical_method_error": _cast_by_space_type(logical_value, logical_meta, "logical_method_error"),
            "coefficient_method_error": _cast_by_space_type(coeff_value, coeff_meta, "coefficient_method_error"),
        }

    def apply(self, series: List[float], params: Dict[str, Any]) -> Tuple[List[float], float]:
        recon, cr = tersets_compress_series(series, params)
        return recon, cr


class SZBackend(Backend):
    name = "sz"

    def __init__(self, bounds: dict, params: dict | None = None):
        self._bounds = bounds

    def bounds(self) -> Dict[str, Tuple[float, float]]:
        return self._bounds

    def params_from_vector(self, v: Dict[str, float]) -> Dict[str, Any]:
        lo, hi = self._bounds["sz_error"]
        return {"sz_error": float(np.clip(v["sz_error"], lo, hi))}

    def apply(self, series: List[float], params: Dict[str, Any]) -> Tuple[List[float], float]:
        recon, cr = sz_compress_series(series, params["sz_error"])
        return recon, cr


class MixPieceBackend(Backend):
    name = "mixpiece"

    def __init__(self, bounds: dict, params: dict | None = None):
        self._bounds = bounds

    def bounds(self) -> Dict[str, Tuple[float, float]]:
        return self._bounds

    def params_from_vector(self, v: Dict[str, float]) -> Dict[str, Any]:
        lo, hi = self._bounds["mixpiece_error"]
        return {"mixpiece_error": float(np.clip(v["mixpiece_error"], lo, hi))}

    def apply(self, series: List[float], params: Dict[str, Any]) -> Tuple[List[float], float]:
        recon, cr = mixpiece_compress_series(series, params["mixpiece_error"])
        return recon, cr

class SerfXOR(Backend):
    """SerfXOR, windowed + digit-adjusted encoder (what was `serfxor_v2`)."""
    name = "serfxor"

    def __init__(self, bounds: dict, params: dict | None = None):
        self._bounds = bounds

    def bounds(self) -> Dict[str, Tuple[float, float]]:
        return self._bounds

    def params_from_vector(self, v: Dict[str, float]) -> Dict[str, Any]:
        lo, hi = self._bounds["serfxor_error"]
        return {"serfxor_error": float(np.clip(v["serfxor_error"], lo, hi))}

    def apply(self, series: List[float], params: Dict[str, Any]) -> Tuple[List[float], float]:
        recon, cr = serfxor_compress_series(series, params["serfxor_error"])
        return recon, cr


class AdaEdgeBackend(Backend):
    """One of the three baselines, *chosen per evaluation* from a method index."""
    name = "adaedge"

    _COMPRESSORS = {
        "mixpiece": mixpiece_compress_series,
        "serfxor": serfxor_compress_series,
        "sz": sz_compress_series,
    }

    def __init__(self, bounds: dict, params: dict):
        self._bounds = bounds
        methods = params["methods"]
        lo, hi = _get_bounds(bounds["method_index"], "method_index")
        if (int(round(lo)), int(round(hi))) != (0, len(methods) - 1):
            raise ValueError(
                f"AdaEdge method_index bounds {(lo, hi)} must span exactly the "
                f"{len(methods)} configured methods, i.e. [0, {len(methods) - 1}]."
            )
        self._methods = methods

    def bounds(self) -> Dict[str, Tuple[float, float]]:
        return self._bounds

    def params_from_vector(self, v: Dict[str, float]) -> Dict[str, Any]:
        method_idx = int(round(v["method_index"]))
        if method_idx < 0 or method_idx >= len(self._methods):
            raise ValueError(
                f"method_index={method_idx} out of range [0, {len(self._methods) - 1}]."
            )
        lo, hi = _get_bounds(self._bounds["adaedge_error"], "adaedge_error")
        return {
            "method": self._methods[method_idx],
            "adaedge_error": clamp(v["adaedge_error"], lo, hi),
        }

    def apply(self, series: List[float], params: Dict[str, Any]) -> Tuple[List[float], float]:
        recon, cr = self._COMPRESSORS[params["method"]](series, params["adaedge_error"])
        return recon, cr


for _tersets_name in TerseTSBackend.names:
    register_backend(_tersets_name, TerseTSBackend)

register_backend(MixPieceBackend.name, MixPieceBackend)
register_backend(SZBackend.name, SZBackend)
register_backend(SerfXOR.name, SerfXOR)
register_backend(AdaEdgeBackend.name, AdaEdgeBackend)
