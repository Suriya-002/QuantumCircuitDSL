#!/usr/bin/env bash
# Profile the gate kernels.
#
#   bash scripts/profile.sh            # perf if usable, else callgrind
#   bash scripts/profile.sh callgrind  # force callgrind
#
# perf reads real hardware counters (cycles, cache misses, IPC) and is the tool
# that tells you WHY a kernel is slow, not just that it is. It needs a perf build
# matching the running kernel, which WSL2 does not ship -- hence the fallback.
# Callgrind SIMULATES the CPU: its wall-clock numbers are meaningless, but its
# instruction counts and cache-miss ratios are real and comparable.
set -euo pipefail

SRC="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD="${QCDSL_BUILD_DIR:-$HOME/build-qcdsl}"
BIN="$BUILD/bench/qcdsl_bench_simulate"
MODE="${1:-auto}"

# Callgrind SIMULATES every instruction: roughly 50-100x slowdown. The full
# sweep runs for ~20 seconds natively, which is half an hour under callgrind.
# Cap the profiled workload at 18 qubits -- the kernel is the same, there is
# just less of it.
PROFILE_QUBITS="${QCDSL_PROFILE_QUBITS:-18}"

if [[ ! -x "$BIN" ]]; then
  echo "building the benchmark first..."
  cmake -S "$SRC" -B "$BUILD" -G Ninja -DCMAKE_BUILD_TYPE=Release \
        -DQCDSL_BUILD_BENCH=ON > /dev/null
  cmake --build "$BUILD" --target qcdsl_bench_simulate --parallel > /dev/null
fi

have() { command -v "$1" > /dev/null 2>&1; }

if [[ "$MODE" != "callgrind" ]] && have perf && perf stat -e cycles true > /dev/null 2>&1; then
  echo "=============== perf stat ==============="
  perf stat -e cycles,instructions,cache-references,cache-misses,branch-misses \
       "$BIN" "$PROFILE_QUBITS" 2>&1 | tail -n 20

  echo
  echo "=============== perf record: where the time goes ==============="
  perf record -q -g -o /tmp/qcdsl.perf "$BIN" "$PROFILE_QUBITS" > /dev/null 2>&1
  perf report -i /tmp/qcdsl.perf --stdio --sort symbol 2>/dev/null | head -n 25
  exit 0
fi

if ! have valgrind; then
  echo "neither perf nor valgrind is usable here."
  echo "  perf     : needs a perf build matching the running kernel"
  echo "  valgrind : sudo apt-get install -y valgrind"
  exit 1
fi

echo "perf is unavailable; using callgrind."
echo "profiling up to $PROFILE_QUBITS qubits (set QCDSL_PROFILE_QUBITS to change)."
echo "NOTE: callgrind SIMULATES the CPU. Read the instruction counts and the"
echo "      cache-miss ratios, not the wall clock."
echo
echo "=============== callgrind ==============="
OUT="/tmp/callgrind.qcdsl.$$"
valgrind --tool=callgrind --callgrind-out-file="$OUT" \
         --cache-sim=yes --branch-sim=yes \
         "$BIN" "$PROFILE_QUBITS" 2>&1 | grep -E "refs|misses|rate|Collected" || true

echo
echo "=============== hottest functions ==============="
if have callgrind_annotate; then
  callgrind_annotate --auto=no "$OUT" 2>/dev/null | sed -n '15,38p'
else
  echo "callgrind_annotate not found; raw profile is at $OUT"
fi
