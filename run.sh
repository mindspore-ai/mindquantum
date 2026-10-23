#!/usr/bin/env bash
set -euo pipefail

ROOT=$(dirname "$(realpath "$0")")
: "${ASCEND_HOME_PATH:?Source the CANN 9.1 set_env.sh before building}"
SOC_VERSION=${SOC_VERSION:-Ascend910B2}
PYTHON_BIN=${PYTHON_BIN:-python3}
JOBS=${JOBS:-8}

# An old extension named kernel_sb shadows the new Python API. Never silently
# load it or delete it on the user's behalf.
if compgen -G "$ROOT/kernel_sb*.so" >/dev/null; then
    echo "Remove the old kernel_sb*.so build artifact before rebuilding." >&2
    exit 1
fi

cmake -S "$ROOT/kernel" -B "$ROOT/kernel/build" \
    -DSOC_VERSION="$SOC_VERSION" \
    -DASCEND_CANN_PACKAGE_PATH="$ASCEND_HOME_PATH" \
    -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX="$ROOT/out"
cmake --build "$ROOT/kernel/build" -j "$JOBS"
cmake --install "$ROOT/kernel/build"
cmake -S "$ROOT/host" -B "$ROOT/host/build" \
    -DASCEND_CANN_PACKAGE_PATH="$ASCEND_HOME_PATH" \
    -DCMAKE_BUILD_TYPE=Release -DPython_EXECUTABLE="$(command -v "$PYTHON_BIN")"
cmake --build "$ROOT/host/build" -j "$JOBS"

# Tests and benchmarks are explicit commands, not side effects of a build.
echo "Built. Test with: $PYTHON_BIN -m pytest -q tests"
