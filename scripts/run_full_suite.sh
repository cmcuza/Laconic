#!/usr/bin/env bash
# Single-call entry point for a full Laconic-clean comparison suite - the
# equivalent of the old Laconic/scripts/{laconic.sh, run_random_suite.sh,
# run_genetic_suite.sh, rerun_clustering_suite.sh} shell scripts, which each
# hand-listed (and hand-commented in/out) one analytics/model/dataset
# combination per optimizer.
#
# All the actual logic - thread-capping env vars, the task x compressor x
# optimizer x model sweep, and regenerating results/figures/*.pdf + MLflow
# artifacts once the sweep finishes - lives in run_suite.py, driven by a
# cfg/suites/<name>.yaml manifest. This script is just a thin, repo-root-
# relative wrapper so there's still a single `./scripts/...` command to run,
# without duplicating that logic in two places.
#
# Usage:
#   ./scripts/run_full_suite.sh                              # cfg/suites/main_comparison.yaml
#   ./scripts/run_full_suite.sh --dry-run                     # preview the commands first
#   ./scripts/run_full_suite.sh cfg/suites/other_manifest.yaml [run_suite.py flags...]

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
