#!/usr/bin/env bash
# Build batch_pyserf and drop the .so next to compression/lib/pyserf.so.
#
# `compression/lib/serfxor.py` imports this module directly (see its module
# docstring) - it is no longer measurement-only. `serfxor` and `adaedge` both
# hard-depend on the resulting .so.
#
# Usage:
#   bash compression/lib/batch_serfxor/build.sh
#
# With no SERF_ROOT set, this clones the pinned Serf commit itself into
# ./serf/ (gitignored) and builds it. Set SERF_ROOT to point at an existing
# checkout (with a built or buildable src/ tree) to skip the clone - the
# checkout still needs the same commit's headers/API; see "Pinned commit"
# below.
set -euo pipefail
here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# --- Pinned commit ---------------------------------------------------------
# https://github.com/Spatio-Temporal-Lab/Serf, commit b38450b56825eabc9
# 6be8e25d6880127dc688c95 (2025-08-13, tip of the repo's `new` default branch
# as of 2026-09-17). Verified 2026-09-17 to build and to reproduce
# tests/test_serfxor_binding_equivalence.py's per-value reference exactly
# (byte-identical compressed size, elementwise-identical reconstruction).
# The repo's own src/CMakeLists.txt hard-codes `add_library(serf SHARED ...)`
# at this commit; wrapper.cpp is linked as a static archive instead (see
# below), so that line is patched to STATIC in the *clone*, never upstream.
# If this pin ever needs to move, verification 1
# (tests/test_serfxor_binding_equivalence.py) is what certifies the new
# commit - a failure there means the pin is wrong, not the test.
#
# License: the Serf repository is CC BY-NC 4.0 (its README.md, "License"
# section). This script clones it locally to build a compiled artifact for
# this (non-commercial, academic) codebase; the clone and the .so are both
# gitignored and never committed or redistributed. Vendoring the Serf source
# into this repo is out of scope for that reason - see README.md.
SERF_COMMIT="b38450b56825eabc96be8e25d6880127dc688c95"
SERF_REPO_URL="https://github.com/Spatio-Temporal-Lab/Serf.git"

serf_root="${SERF_ROOT:-}"
python_bin="${PYTHON_BIN:-python3}"

if [[ -z "$serf_root" ]]; then
    serf_root="$here/serf"
    if [[ ! -d "$serf_root/.git" ]]; then
        echo "SERF_ROOT not set - cloning pinned Serf commit into $serf_root"
        git clone "$SERF_REPO_URL" "$serf_root"
    fi
    git -C "$serf_root" fetch --depth 1 origin "$SERF_COMMIT" 2>/dev/null || true
    git -C "$serf_root" checkout --detach "$SERF_COMMIT"
    # The checked-out src/CMakeLists.txt at this commit builds `serf` SHARED.
    # wrapper.cpp is linked into our own shared module, so the archive it
    # links against must be a *static* (and -fPIC) libserf.a. This edits only
    # the gitignored clone, never anything tracked in this repo.
    sed -i 's/add_library(serf SHARED/add_library(serf STATIC/' "$serf_root/src/CMakeLists.txt"
fi

if [[ ! -f "$serf_root/src/CMakeLists.txt" ]]; then
    echo "error: SERF_ROOT=$serf_root has no src/CMakeLists.txt - not a Serf checkout" >&2
    exit 1
fi

if ! command -v cmake >/dev/null 2>&1; then
    echo "error: cmake not found on PATH. Install it into the running interpreter's env with:" >&2
    echo "    $python_bin -m pip install cmake" >&2
    echo "and re-run (pip's cmake wheel also drops a 'cmake' launcher on PATH once the env's bin/ is on it)." >&2
    exit 1
fi

if ! "$python_bin" -c "import pybind11" >/dev/null 2>&1; then
    echo "error: pybind11 not importable by $python_bin. Set PYTHON_BIN to the env that has it." >&2
    exit 1
fi

# --- Build libserf.a (static, position-independent: it is linked into our
# own shared module below) ---------------------------------------------------
echo "building libserf.a from $serf_root"
cmake -S "$serf_root/src" -B "$serf_root/build/src" \
      -DCMAKE_BUILD_TYPE=Release \
      -DCMAKE_POSITION_INDEPENDENT_CODE=ON
cmake --build "$serf_root/build/src" -j

if [[ ! -f "$serf_root/build/src/libserf.a" ]]; then
    echo "error: build finished but $serf_root/build/src/libserf.a is missing" >&2
    exit 1
fi

# --- Build the wrapper -------------------------------------------------------
cmake -S "$here" -B "$here/build" \
      -DSERF_ROOT="$serf_root" \
      -DPython_EXECUTABLE="$python_bin" \
      -Dpybind11_DIR="$("$python_bin" -c 'import pybind11; print(pybind11.get_cmake_dir())')" \
      -DCMAKE_BUILD_TYPE=Release
cmake --build "$here/build" -j
cp "$here"/build/batch_pyserf*.so "$here/../"
echo "installed: $(ls "$here"/../batch_pyserf*.so)"
