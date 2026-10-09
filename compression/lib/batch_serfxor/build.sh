#!/usr/bin/env bash
# Build batch_pyserf (needed by serfxor/adaedge) into compression/lib/.
# Clones the pinned Serf commit unless SERF_ROOT points at a checkout.
# Serf is CC BY-NC 4.0: the clone and .so are gitignored, never committed.
set -euo pipefail
here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

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
    # wrapper.cpp needs a static libserf.a; patch the clone only.
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

echo "building libserf.a from $serf_root"
cmake -S "$serf_root/src" -B "$serf_root/build/src" \
      -DCMAKE_BUILD_TYPE=Release \
      -DCMAKE_POSITION_INDEPENDENT_CODE=ON
cmake --build "$serf_root/build/src" -j

if [[ ! -f "$serf_root/build/src/libserf.a" ]]; then
    echo "error: build finished but $serf_root/build/src/libserf.a is missing" >&2
    exit 1
fi

cmake -S "$here" -B "$here/build" \
      -DSERF_ROOT="$serf_root" \
      -DPython_EXECUTABLE="$python_bin" \
      -Dpybind11_DIR="$("$python_bin" -c 'import pybind11; print(pybind11.get_cmake_dir())')" \
      -DCMAKE_BUILD_TYPE=Release
cmake --build "$here/build" -j
cp "$here"/build/batch_pyserf*.so "$here/../"
echo "installed: $(ls "$here"/../batch_pyserf*.so)"
