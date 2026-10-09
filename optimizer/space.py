from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Tuple, List, Any, Optional
import numpy as np
import math


# Everything a `space:` entry may declare. Anything else is a config error:
# an unrecognized key that is quietly dropped is indistinguishable from one
# that took effect, and both `fixed:` (removed) and a mistyped
# `significant_digits` would silently widen the search rather than fail.
ALLOWED_SPACE_KEYS = frozenset({"type", "scale", "significant_digits"})

# Removed configuration forms, mapped to what replaced them. Named explicitly
# so an old config gets a fix rather than a generic "unknown key".
_REMOVED_SPACE_KEYS = {
    "fixed": (
        "pinning a parameter is now expressed as an equal-bounds range, "
        "`bounds: [value, value]` - sampling, mutation and crossover all "
        "collapse to that value"
    ),
    "default": (
        "per-method conditional bounds were removed; give the parameter one "
        "`[lo, hi]` range covering every method"
    ),
}


def parse_bounds_pair(raw_bound: Any, field_name: str) -> Tuple[float, float]:
    """Parse one ``bounds`` entry into ``(lo, hi)``.

    The only accepted form is a two-element ``[lo, hi]`` sequence. The dict form
    (``{"default": [lo, hi], "MethodX": [lo, hi]}``) used to be accepted here and
    by random/bomab/bosmp, but only ``default`` was ever read - every per-method
    entry was silently discarded, so a config that carefully narrowed the error
    range for one compressor got the default range and results that looked
    entirely plausible. It is rejected rather than ignored.
    """
    if isinstance(raw_bound, (list, tuple)) and len(raw_bound) == 2:
        return float(raw_bound[0]), float(raw_bound[1])
    if isinstance(raw_bound, dict):
        raise ValueError(
            f"Bounds for '{field_name}' use the removed per-method conditional "
            f"form {raw_bound}. Only a single [lo, hi] range is supported; give "
            "the parameter one range covering every method."
        )
    raise ValueError(
        f"Invalid bounds for '{field_name}': expected [lo, hi], got {raw_bound!r}."
    )


def parse_space_metadata(raw_metadata: Any, parameter_name: str) -> Dict[str, Any]:
    """Validate one ``space`` entry and return it.

    Requires ``type`` and ``scale``, and rejects any key outside
    :data:`ALLOWED_SPACE_KEYS` - see that constant for why silence is not an
    option here.
    """
    if not isinstance(raw_metadata, dict):
        raise ValueError(
            f"Space definition for '{parameter_name}' must be a dict with "
            f"'type' and 'scale'. Got: {raw_metadata!r}"
        )
    for removed_key, replacement in _REMOVED_SPACE_KEYS.items():
        if removed_key in raw_metadata:
            raise ValueError(
                f"Space definition for '{parameter_name}' uses the removed "
                f"'{removed_key}' key: {replacement}."
            )
    unknown_keys = set(raw_metadata) - ALLOWED_SPACE_KEYS
    if unknown_keys:
        raise ValueError(
            f"Space definition for '{parameter_name}' has unknown key(s) "
            f"{sorted(unknown_keys)}. Allowed: {sorted(ALLOWED_SPACE_KEYS)}."
        )
    missing_keys = {"type", "scale"} - set(raw_metadata)
    if missing_keys:
        raise ValueError(
            f"Space definition for '{parameter_name}' must provide "
            f"{sorted(missing_keys)}. Got: {raw_metadata}"
        )
    return raw_metadata


def _reflect(value: float, lower_bound: float, upper_bound: float) -> float:
    interval_width = upper_bound - lower_bound
    if interval_width == 0:
        return lower_bound

    offset = (value - lower_bound) % (2 * interval_width)
    if offset <= interval_width:
        return lower_bound + offset

    return upper_bound - (offset - interval_width)

@dataclass
class VariableSpec:
    lower_bound: float
    upper_bound: float
    value_type: str  # "float" | "int"
    scale: str  # "linear" | "log"
    significant_digits: Optional[int] = None


class Space:
    def __init__(
        self, bounds: Dict[str, Tuple[float, float]], space: Dict[str, Dict[str, Any]]
    ):
        """
        bounds: {"param": [lo, hi]}
        space:  per-param dicts: {"param": {"type":"int|float","scale":"linear|log"}}
                plus an optional "significant_digits" for float parameters

        Every parameter in ``bounds`` must be annotated in ``space`` with an
        explicit ``type`` and ``scale`` — there are no name-based or
        range-based defaults, so a missing annotation is a config error, not a
        silently-linear float. Unknown keys are rejected for the same reason
        (see :data:`ALLOWED_SPACE_KEYS`), and a parameter is pinned with
        ``bounds: [value, value]`` rather than a dedicated key.
        """
        if not isinstance(space, dict):
            raise ValueError(
                "Space requires a space definition dict annotating every parameter."
            )
        self.variable_specs: Dict[str, VariableSpec] = {}

        for parameter_name in space:
            if parameter_name not in bounds:
                raise ValueError(
                    f"Space config has unknown parameter '{parameter_name}' "
                    "not listed in bounds."
                )

        for parameter_name, parameter_bounds in bounds.items():
            lower_bound, upper_bound = self._resolve_bounds(
                parameter_bounds, parameter_name
            )
            if parameter_name not in space:
                raise ValueError(
                    f"Parameter '{parameter_name}' is missing from the space definition."
                )
            parameter_metadata = self._resolve_metadata(
                space[parameter_name], parameter_name
            )
            value_type = str(parameter_metadata["type"]).lower()
            scale = str(parameter_metadata["scale"]).lower()
            significant_digits = parameter_metadata.get("significant_digits")
            if value_type not in ("int", "float"):
                raise ValueError(
                    f"Unsupported type '{parameter_metadata['type']}' for "
                    f"'{parameter_name}'. Expected int/float."
                )
            if scale not in ("linear", "log"):
                raise ValueError(
                    f"Unsupported scale '{parameter_metadata['scale']}' for "
                    f"'{parameter_name}'. Expected linear/log."
                )
            if scale == "log" and (lower_bound <= 0 or upper_bound <= 0):
                raise ValueError(
                    f"Log-scaled parameter '{parameter_name}' must have "
                    f"positive bounds. Got {(lower_bound, upper_bound)}."
                )
            if significant_digits is not None:
                if (
                    value_type != "float"
                    or isinstance(significant_digits, bool)
                    or not isinstance(significant_digits, int)
                    or significant_digits < 1
                ):
                    raise ValueError(
                        f"Parameter {parameter_name!r} has invalid "
                        f"significant_digits={significant_digits!r}. Expected a "
                        "positive integer for a float parameter."
                    )
            self.variable_specs[parameter_name] = VariableSpec(
                lower_bound=float(lower_bound),
                upper_bound=float(upper_bound),
                value_type=value_type,
                scale=scale,
                significant_digits=significant_digits,
            )

    _resolve_bounds = staticmethod(parse_bounds_pair)
    _resolve_metadata = staticmethod(parse_space_metadata)

    # ---------- sampling ----------
    def sample(self, rng: np.random.Generator) -> Dict[str, float]:
        sampled_candidate = {}
        for parameter_name, variable_spec in self.variable_specs.items():
            if variable_spec.value_type == "int" and variable_spec.scale == "linear":
                sampled_value = rng.integers(
                    math.ceil(variable_spec.lower_bound),
                    math.floor(variable_spec.upper_bound) + 1,
                )
            elif variable_spec.scale == "log":
                sampled_value = math.exp(
                    rng.uniform(
                        math.log(variable_spec.lower_bound),
                        math.log(variable_spec.upper_bound),
                    )
                )
            else:
                sampled_value = rng.uniform(
                    variable_spec.lower_bound, variable_spec.upper_bound
                )
            sampled_candidate[parameter_name] = self._canonicalize_value(
                parameter_name, sampled_value
            )
        return sampled_candidate

    # ---------- cast/clip/repair ----------
    @staticmethod
    def _round_to_significant_digits(value: float, significant_digits: int) -> float:
        if value == 0.0:
            return 0.0
        decimal_places = significant_digits - math.floor(math.log10(value)) - 1
        return float(round(value, decimal_places))

    def _canonicalize_value(self, parameter_name: str, value: float) -> float:
        """Bound a value and restore its declared representation.

        The declared representation has to be restored *after* bounding, or an
        integer genome could leak a fractional value into deduplication,
        logging, and backend conversion.
        """
        variable_spec = self.variable_specs[parameter_name]
        bounded_value = min(
            max(float(value), variable_spec.lower_bound),
            variable_spec.upper_bound,
        )
        if variable_spec.value_type == "int":
            rounded_value = float(round(bounded_value))
        elif variable_spec.significant_digits is not None:
            rounded_value = self._round_to_significant_digits(
                bounded_value, variable_spec.significant_digits
            )
        else:
            return bounded_value

        # Rounding can step back outside the range it was handed: round(0.5) is
        # 0 under a lower bound of 0.2, and two significant digits take 0.1499
        # up to 0.15. Only the rounded branches need re-bounding.
        return min(
            max(rounded_value, variable_spec.lower_bound), variable_spec.upper_bound
        )

    def clip(self, candidate: Dict[str, float]) -> Dict[str, float]:
        """Bound and canonicalize a whole candidate.

        ``sample``/``mutate``/``crossover`` already return canonical candidates,
        so nothing in the current optimizers calls this. It is the repair entry
        point for an optimizer that proposes raw points some other way (a BO
        acquisition maximizer, say) and needs them put back on the declared
        grid before evaluation or deduplication.
        """
        return {
            parameter_name: self._canonicalize_value(
                parameter_name, candidate[parameter_name]
            )
            for parameter_name in self.variable_specs
        }

    # ---------- mutation ----------
    def mutate(
        self,
        rng: np.random.Generator,
        candidate: Dict[str, float],
        prob: float = 0.5,
        sigma_frac: float = 0.1,
        log_sigma: float = 0.75,
    ) -> Dict[str, float]:
        """
        Integer linear variables choose uniformly among alternative values.
        sigma_frac: fraction of (hi-lo) for continuous linear vars (Gaussian nudge)
        log_sigma: stdev in log-space for log vars
        """
        mutated_candidate = {}
        for parameter_name, variable_spec in self.variable_specs.items():
            current_value = candidate[parameter_name]
            if rng.random() >= prob:
                mutated_candidate[parameter_name] = current_value
                continue

            if variable_spec.value_type == "int" and variable_spec.scale == "linear":
                lower_bound = math.ceil(variable_spec.lower_bound)
                upper_bound = math.floor(variable_spec.upper_bound)
                current_integer = int(round(current_value))
                if lower_bound == upper_bound:
                    mutated_value = float(lower_bound)
                else:
                    mutated_value = int(rng.integers(lower_bound, upper_bound))
                    if mutated_value >= current_integer:
                        mutated_value += 1
                    mutated_value = float(mutated_value)
            elif variable_spec.scale == "log":
                effective_log_sigma = (log_sigma if current_value >= 1e-4 else log_sigma + 0.25)
                proposed_value = np.exp(np.log(current_value) + rng.normal(0.0, effective_log_sigma))
                mutated_value = _reflect(proposed_value, variable_spec.lower_bound, variable_spec.upper_bound)
            else:
                value_range = variable_spec.upper_bound - variable_spec.lower_bound
                mutated_value = current_value + rng.normal(0.0, sigma_frac * value_range)

            mutated_candidate[parameter_name] = self._canonicalize_value(parameter_name, mutated_value)
        return mutated_candidate

    def mutate_elite(
        self,
        rng: np.random.Generator,
        candidate: Dict[str, float],
        sigma_frac: float = 0.1,
        log_sigma: float = 1.0,
    ) -> Dict[str, float]:
        """Local refinement: move the continuous genes, freeze the discrete ones.

        For ``tersets_mab`` this re-tunes ``logical_method_error`` and
        ``coefficient_method_error`` while holding the three method indices, i.e.
        it keeps the pipeline and searches only its error bounds. The split is by
        declared ``type`` (``int`` frozen, ``float`` mutated), never by parameter
        name, so it needs no knowledge of any particular compressor - on a
        single-parameter baseline it degrades to an ordinary mutation of that one
        parameter.

        Unlike :meth:`mutate` there is no per-gene probability: every continuous
        gene moves. A refinement that happened to leave all of them untouched
        would just be the elite again, spending a duplicate-rejection retry to
        learn nothing.
        """
        refined_candidate = {}
        for parameter_name, variable_spec in self.variable_specs.items():
            current_value = candidate[parameter_name]
            if variable_spec.value_type == "int":
                refined_candidate[parameter_name] = current_value
                continue

            if variable_spec.scale == "log":
                # The log-space mutation is a Gaussian nudge in log-space, but the stdev 
                # is increased for very small values to avoid getting stuck at the lower bound.
                effective_log_sigma = log_sigma + 0.25 if current_value >= 1e-4 else log_sigma + 0.5
                proposed_value = np.exp(np.log(current_value) + rng.normal(0.0, effective_log_sigma))
                mutated_value = _reflect(proposed_value, variable_spec.lower_bound, variable_spec.upper_bound)
            else:
                value_range = variable_spec.upper_bound - variable_spec.lower_bound
                mutated_value = current_value + rng.normal(0.0, sigma_frac * value_range)

            refined_candidate[parameter_name] = self._canonicalize_value(parameter_name, mutated_value)
        return refined_candidate

    # ---------- crossover ----------
    def crossover(
        self,
        rng: np.random.Generator,
        first_parent: Dict[str, float],
        second_parent: Dict[str, float],
    ) -> Dict[str, float]:
        offspring = {}
        for parameter_name in self.variable_specs:
            parent_value = first_parent[parameter_name] if rng.random() < 0.5 else second_parent[parameter_name]
            offspring[parameter_name] = parent_value

        return offspring
