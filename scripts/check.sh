#!/usr/bin/env bash
# Local rehearsal of the CI pipeline. Run before every push.
set -euo pipefail

SRC="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD="${QCDSL_BUILD_DIR:-$HOME/build-qcdsl}"
COV="${QCDSL_COV_DIR:-$HOME/cov-qcdsl}"
export PATH="$HOME/.local/bin:$PATH"

step() { printf '\n== %s ==\n' "$1"; }
cd "$SRC"

step "build"
cmake -S "$SRC" -B "$BUILD" -G Ninja -DCMAKE_BUILD_TYPE=Release > /dev/null
cmake --build "$BUILD" --parallel > /dev/null
echo ok

step "C++ tests"
ctest --test-dir "$BUILD" --output-on-failure | tail -n 30

step "Python tests"
PYTHONPATH="$BUILD/python" python3 -m pytest python/tests -q

step "clang-format"
find include python tests \( -name '*.hpp' -o -name '*.cpp' \) -print0 \
  | xargs -0 clang-format --dry-run --Werror
echo clean

step "clang-tidy"
cmake -S "$SRC" -B "${BUILD}-tidy" -G Ninja -DCMAKE_BUILD_TYPE=Debug -DQCDSL_BUILD_PYTHON=OFF > /dev/null
clang-tidy -p "${BUILD}-tidy" tests/test_gate.cpp tests/test_circuit.cpp tests/test_statevector.cpp 2>/dev/null
echo clean

step "coverage (gate: 90%)"
cmake -S "$SRC" -B "$COV" -G Ninja -DCMAKE_BUILD_TYPE=Debug -DQCDSL_COVERAGE=ON -DQCDSL_BUILD_PYTHON=OFF > /dev/null
cmake --build "$COV" --parallel > /dev/null
ctest --test-dir "$COV" > /dev/null
gcovr --root "$SRC" --filter 'include/qcdsl/' --print-summary --fail-under-line 90 "$COV" 2>/dev/null | tail -3

printf '\nALL GREEN - safe to push\n'