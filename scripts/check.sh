#!/usr/bin/env bash
# Local rehearsal of the CI pipeline. Run before every push.
set -euo pipefail

SRC="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD="${QCDSL_BUILD_DIR:-$HOME/build-qcdsl}"
COV="${QCDSL_COV_DIR:-$HOME/cov-qcdsl}"
export PATH="$HOME/.local/bin:$PATH"

step() { printf '\n== %s ==\n' "$1"; }
cd "$SRC"

step "tool versions (must match ci.yml)"
clang-format --version
gcovr --version | head -1

step "build"
cmake -S "$SRC" -B "$BUILD" -G Ninja -DCMAKE_BUILD_TYPE=Release -DQCDSL_BUILD_BENCH=ON > /dev/null
cmake --build "$BUILD" --parallel > /dev/null
echo ok

step "C++ tests"
ctest --test-dir "$BUILD" --output-on-failure | tail -n 30

step "Python tests (incl. the Qiskit oracle)"
PYTHONPATH="$BUILD/python" python3 -m pytest python/tests -q

step "clang-format"
find include python tests bench \( -name '*.hpp' -o -name '*.cpp' \) -print0 \
  | xargs -0 clang-format --dry-run --Werror
echo clean

step "clang-tidy"
cmake -S "$SRC" -B "${BUILD}-tidy" -G Ninja -DCMAKE_BUILD_TYPE=Debug -DQCDSL_BUILD_PYTHON=OFF > /dev/null
clang-tidy -p "${BUILD}-tidy" tests/test_gate.cpp tests/test_circuit.cpp \
  tests/test_statevector.cpp tests/test_dag.cpp tests/test_passes.cpp tests/test_qasm3.cpp tests/test_kernels.cpp tests/test_routing.cpp 2>/dev/null
echo clean

step "coverage (minimum 90 percent)"
cmake -S "$SRC" -B "$COV" -G Ninja -DCMAKE_BUILD_TYPE=Debug -DQCDSL_COVERAGE=ON -DQCDSL_BUILD_PYTHON=OFF > /dev/null
cmake --build "$COV" --parallel > /dev/null
ctest --test-dir "$COV" > /dev/null
gcovr --root "$SRC" --filter 'include/qcdsl/' --print-summary --fail-under-line 90 "$COV" 2>/dev/null | tail -3

step "optimisation benchmark"
"$BUILD/bench/qcdsl_bench_optimise"

printf '\nALL GREEN - safe to push\n'