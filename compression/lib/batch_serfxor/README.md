# batch_pyserf

Batched pybind11 binding for SerfXOR, required by the `serfxor` and `adaedge`
backends. Output is byte-identical to the per-value `pyserf` binding
(`tests/test_serfxor_binding_equivalence.py`).

```bash
bash compression/lib/batch_serfxor/build.sh   # needs cmake, a C++ compiler and pybind11
```

The script clones a pinned [Serf](https://github.com/Spatio-Temporal-Lab/Serf)
commit (or uses `SERF_ROOT`). Serf is CC BY-NC 4.0, so its source and the built
`.so` are gitignored and not redistributed.
