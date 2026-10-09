"""Byte-identity guard for the batched SerfXOR binding (`docs/EXECUTION_TIME_STUDY_PLAN.md` §4.3).

Phase 0 deleted the per-value block loop from `compression/lib/serfxor.py` and
replaced it with a call into `compression.lib.batch_pyserf`
(`compression/lib/batch_serfxor/`). After that change,
`scripts/benchmark_compression_throughput.py --methods serfxor` validates the
batched path against `serfxor_compress_series`, which *is* the batched path -
a tautology, not a cross-binding check. This test is what still exercises the
real invariant: the batched binding must reproduce the deleted per-value
binding exactly, on real analytics series, at the error bounds the search
space actually uses.

The per-value reference lives only in this test (built directly from
`compression.lib.pyserf.PySerfXORCompressor`/`PySerfXORDecompressor`) - the
shipping code has no reason to keep a second, slower implementation of the
same algorithm around once nothing calls it.

If this test fails after `compression/lib/batch_serfxor/build.sh`'s pinned
Serf commit is moved, the new pin is wrong, not this test - see that script's
`SERF_COMMIT` comment.
"""
from __future__ import annotations

import numpy as np
import pytest
import yaml

from compression.lib.pyserf import PySerfXORCompressor, PySerfXORDecompressor
from compression.lib.serfxor import SERF_BLOCK_SIZE, SERF_WINDOW_SIZE, serf_adjust_digit, serfxor_compress_series
from data.loaders import MonashUCRLoader, MonashLoader, UCRLoader

PROJECT_ROOT_CFG = "cfg/compression/serfxor.yaml"


def _per_value_reference(series: np.ndarray, error_bound: float):
    """The per-value SerfXOR encode/decode `serfxor.py` used to do, verbatim.

    This is `serfxor_compress_series`'s pre-Phase-0 body - kept alive only
    here, as the thing the batched path is checked against, per this test's
    module docstring.
    """
    series = np.asanyarray(series, dtype=np.float64)
    n = series.size
    ts_min, ts_max = float(series.min()), float(series.max())
    ts_range = max(abs(ts_max - ts_min), 1e-12)
    abs_eb = float(error_bound) * ts_range

    adjust = serf_adjust_digit(ts_min, ts_max)

    serf_comp = PySerfXORCompressor(SERF_WINDOW_SIZE, abs_eb, adjust)
    serf_decomp = PySerfXORDecompressor(adjust)

    rec = []
    cmp_size = 0
    for start in range(0, n, SERF_BLOCK_SIZE):
        for v in series[start:start + SERF_BLOCK_SIZE]:
            serf_comp.add_value(v)
        serf_comp.close()

        pack = serf_comp.get()
        cmp_size += len(pack.to_bytes())
        rec.extend(serf_decomp.decompress(pack))

    raw_size = n * series.itemsize
    cr = round(raw_size / max(1, cmp_size), 2)
    assert len(rec) == n
    return rec, cr, cmp_size


def _serfxor_error_range():
    with open(PROJECT_ROOT_CFG) as fh:
        cfg = yaml.safe_load(fh)
    lo, hi = cfg["bounds"]["serfxor_error"]
    mid = (lo + hi) / 2.0
    return {"low": lo, "mid": mid, "high": hi}


def _coffee_series() -> np.ndarray:
    """One real UCR series - classification/clustering's Coffee, length 286."""
    X, _y = UCRLoader().load_train("Coffee")
    # UCR shape is (n, channels, length); one channel of one series.
    return np.asarray(X)[0, 0, :].astype(np.float64)


def _regression_channel() -> np.ndarray:
    """One real regression channel.

    The plan names Covid3Month (length 84) as the example; it is not present
    under `data/regression/Monash-UCR` in this checkout (only
    AppliancesEnergy/FloodModeling1/IEEEPPG/LiveFuelMoistureContent are).
    FloodModeling1 (length 266) is used instead - still a real regression
    series from the analytics corpus, not the throughput-benchmark one, which
    is the property this test needs.
    """
    X, _y = MonashUCRLoader().load_train("FloodModeling1")
    # regression shape is (n, length, channels); series 0, channel 0.
    return np.asarray(X)[0, :, 0].astype(np.float64)


def _forecasting_window() -> np.ndarray:
    """One real forecasting window - the val split of saugeenday under forward_chain,
    the same shape `ForecastingObjective` evaluates (a single series, no eval_indices)."""
    loader = MonashLoader()
    _train, val, _test = next(loader.forward_chain("saugeenday"))
    return np.asarray(val, dtype=np.float64)


SERIES_CASES = {
    "coffee_ucr": _coffee_series,
    "flood_regression": _regression_channel,
    "saugeenday_forecasting": _forecasting_window,
}


@pytest.fixture(scope="module")
def error_bounds():
    return _serfxor_error_range()


@pytest.fixture(scope="module", params=sorted(SERIES_CASES))
def series_case(request):
    name = request.param
    return name, SERIES_CASES[name]()


@pytest.mark.parametrize("error_level", ["low", "mid", "high"])
def test_batched_matches_per_value_reference(series_case, error_bounds, error_level):
    name, series = series_case
    error_bound = error_bounds[error_level]

    ref_rec, ref_cr, ref_cmp_size = _per_value_reference(series, error_bound)
    batch_rec, batch_cr = serfxor_compress_series(series, error_bound)

    assert isinstance(batch_rec, list), "return contract: rec must be a Python list"
    assert len(batch_rec) == len(series)

    # Byte-identical compressed size.
    assert batch_cr == ref_cr, (
        f"{name}/{error_level}: compressed-size-derived CR diverged "
        f"(batched={batch_cr}, per-value reference={ref_cr})"
    )

    # Elementwise-identical reconstruction.
    ref_arr = np.asarray(ref_rec, dtype=np.float64)
    batch_arr = np.asarray(batch_rec, dtype=np.float64)
    assert np.array_equal(ref_arr, batch_arr), (
        f"{name}/{error_level}: reconstruction diverged between the batched binding "
        f"and the per-value reference"
    )


def test_short_series_exercise_single_partial_block():
    """`coffee_ucr` (286 points) and `flood_regression` (266 points) are both
    under SERF_BLOCK_SIZE=1000, so each exercises exactly one, partial, block -
    the path the README calls out as untested by the 100K-value throughput
    corpus. `saugeenday_forecasting` (2,374 points) instead spans several
    blocks including a final partial one, covering the multi-block case in the
    same run. Assert both shapes explicitly so this guard can't silently stop
    covering either."""
    single_block = {"coffee_ucr", "flood_regression"}
    multi_block = {"saugeenday_forecasting"}
    assert single_block | multi_block == set(SERIES_CASES)

    for name in single_block:
        series = SERIES_CASES[name]()
        assert 0 < series.size < SERF_BLOCK_SIZE, (
            f"{name} has {series.size} values - expected a single partial block (< {SERF_BLOCK_SIZE})"
        )
    for name in multi_block:
        series = SERIES_CASES[name]()
        assert series.size > SERF_BLOCK_SIZE, (
            f"{name} has {series.size} values - expected to span multiple blocks (> {SERF_BLOCK_SIZE})"
        )


def test_serf_adjust_digit_is_well_defined_on_real_series():
    """Sanity check on the shared helper both paths call (imported from
    `serfxor.py`, not reimplemented here), so a divergence in the digit
    adjustment isn't mistaken for one in the batched binding itself."""
    for name, factory in SERIES_CASES.items():
        series = factory()
        lo, hi = float(series.min()), float(series.max())
        adjust = serf_adjust_digit(lo, hi)
        assert isinstance(adjust, int), name
        assert adjust >= 0, name  # serf_adjust_digit clamps to max(0, ...)
