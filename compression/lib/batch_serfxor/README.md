# batch_pyserf — batched SerfXOR binding

**Every experiment that runs `serfxor` or `adaedge` imports this.**
`compression/lib/serfxor.py::serfxor_compress_series` routes through this
module directly — the per-value `pyserf` loop it used to run was deleted, not
kept as an alternative. `adaedge` samples the `serfxor` arm on roughly a third
of its evaluations (`compression/backend.py::AdaEdgeBackend`), so it depends on
this too. `compression/lib/pyserf.so` (the per-value binding) is still shipped
and still used by `scripts/benchmark_compression_throughput.py --methods
serfxor_pyloop` and by `tests/test_serfxor_binding_equivalence.py`'s in-test
reference — that is what the batched path is checked *against*, not an
alternate production path.

## The problem it solves

`pyserf` exposes only `add_value(double)`. Compressing a series from Python
therefore costs one pybind11 call per value — measured at ~307 ns, roughly
**31 ms per 100K values**. SZ3, MixPiece and TerseTS all accept a whole buffer
per call, so in a like-for-like throughput plot SerfXOR would be charged ~30 ms
of interpreter round trips and read as the slowest compressor, which is not what
the codec does. And every real analytics evaluation using `serfxor` paid that
same per-value cost, not only the throughput benchmark.

This module hands the array over once and runs the identical per-block loop in
C++: same compressor instance for the series, same window, same 1000-value block
boundaries, same `Close()` per block.

Measured, best-of-3, single-threaded, on the 5K–100K-value corpus
`scripts/benchmark_compression_throughput.py` uses:

| n | series | shipped | batched | |
|---|---|---|---|---|
| 5,000 | random walk | 3.23 ms | 0.10 ms | 31.9× |
| 5,000 | sine + noise | 3.03 ms | 0.09 ms | 33.7× |
| 100,000 | random walk | 31.40 ms | 1.16 ms | 27.1× |
| 100,000 | sine + noise | 33.31 ms | 1.99 ms | 16.8× |

The analytics series this module now actually compresses in production are far
shorter (UCR/regression/forecasting: 84–637 values per series/channel/window),
where the fixed per-call cost is a larger share — see
`scripts/benchmark_optimizer_overhead.py` / `docs/EXECUTION_TIME.md` for that
number; this table is the throughput-benchmark corpus, not the analytics one.

**The output is byte-identical**, not merely equivalent — verified per block, on
the compressed size, and on the reconstruction. Two places check this now:

- `scripts/benchmark_compression_throughput.py --methods serfxor,serfxor_pyloop`
  (`serfxor` here still names the shipped `compression/lib/serfxor.py` path,
  which *is* the batched path post-replacement — so this validates the batched
  binding against `library_cr`, not against the old per-value binding).
- `tests/test_serfxor_binding_equivalence.py` — the binding's real cross-check
  now that the benchmark's own gate compares the batched path to itself. It
  builds a per-value reference directly from `compression.lib.pyserf` inside
  the test (the shipping code no longer has one to reuse) and checks it against
  `serfxor_compress_series` on real analytics series, not just the throughput
  corpus.

It also removes an aliasing trap. Upstream binds `get()` with
`py::return_value_policy::reference`, so the returned pack is a reference to the
compressor's own buffer: a pack held across further encoder use is no longer the
pack that was produced, and `to_bytes()`/`decompress()` then hang on the stale
handle. That is why the old per-value loop decompressed each block immediately
— the interleaving was load-bearing there. Here every block is copied into an
owned `py::bytes` before the compressor is touched again, so packs are safe to
collect independently of when they're decompressed.

## Building

```bash
bash build.sh                        # clones + builds a pinned Serf commit, writes ../batch_pyserf.*.so
SERF_ROOT=~/Serf PYTHON_BIN=$(which python) bash build.sh   # or: reuse an existing checkout
```

With `SERF_ROOT` unset, `build.sh` clones the pinned commit itself into `./serf/`
(gitignored) and builds `libserf.a` from it before building the wrapper — a bare
`git clone && bash build.sh` on a machine with a C++ compiler and no prior Serf
checkout is enough. The pinned commit hash lives in `build.sh` (search
`SERF_COMMIT`) and in `docs/EXECUTION_TIME_STUDY_PLAN.md` §13; the equivalence
test above is what certifies it — if it ever fails after moving the pin, the pin
is wrong, not the test.

**License.** The [Serf repository](https://github.com/Spatio-Temporal-Lab/Serf)
is licensed CC BY-NC 4.0 (its `README.md`, "License" section) — non-commercial
use only. This project is academic research, so building and linking against it
locally is in scope, but **the Serf source is not vendored into this repo**:
`./serf/` (the clone) and the built `.so` are both gitignored, `wrapper.cpp` plus
this file are what's needed to reproduce the build, and nothing from Serf is
redistributed. Re-check the license before ever changing that.

`build/`, `serf/`, and the installed `.so` are gitignored: a compiled artifact
for one Python ABI and one machine does not belong in the repo, `wrapper.cpp`
plus this file are enough to reproduce it, and the license (above) rules out
vendoring the source. That is deliberately *not* how the shipped `pyserf.so` is
handled — it is committed with no source here, which is the gap worth not
repeating for `batch_pyserf`.

`cmake` is required and is not always preinstalled; if `command -v cmake` fails,
`build.sh` prints `pip install cmake` into the interpreter's own env as the fix
(that is a real, working option — no system package needed).

## If you change it

The one invariant now: byte-identical output to the per-value reference,
checked by `tests/test_serfxor_binding_equivalence.py` and by

```bash
python scripts/benchmark_compression_throughput.py --datasets Basel-temp \
    --methods serfxor,serfxor_pyloop --repeat 2
```

which times both bindings and validates each against the library.
