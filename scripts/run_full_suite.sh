#!/usr/bin/env bash
# Run a suite manifest (default: cfg/suites/main_comparison.yaml) via run_suite.py.
# Usage: ./scripts/run_full_suite.sh [manifest.yaml] [run_suite.py flags...]

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"
cd "$PROJECT_ROOT"

manifest="cfg/suites/main_comparison.yaml"
if [[ "${1:-}" == *.yaml ]]; then
    manifest="$1"
    shift
fi

echo "Running suite manifest: $manifest"
exec "$PYTHON_BIN" run_suite.py "$manifest" "$@"
