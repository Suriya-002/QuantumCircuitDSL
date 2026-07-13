// Reproduces the routing table in the README.
//
//   cmake --build build --target qcdsl_bench_route &&
//   ./build/bench/qcdsl_bench_route
//
// Reports the SWAPs SABRE inserts on each topology, with a trivial layout and
// with the layout search. The head-to-head against Qiskit's transpiler lives in
// python/tests/test_routing_vs_qiskit.py, which is where the comparison
// belongs: Qiskit is the reference, and a benchmark that only ever measures
// itself is a benchmark that cannot lose.

#include <cstdint>
#include <cstdio>
#include <random>
#include <vector>

#include "qcdsl/qcdsl.hpp"

using namespace qcdsl;

namespace {

Circuit synth(std::size_t n, std::size_t m, std::uint64_t seed) {
  std::mt19937_64 rng(seed);
  std::uniform_real_distribution<double> u01(0.0, 1.0);
  Circuit qc(n);
  for (std::size_t i = 0; i < m; ++i) {
    if (u01(rng) < 0.45 && n >= 2) {
      std::uniform_int_distribution<std::size_t> pq(0, n - 1);
      Qubit a = pq(rng);
      Qubit b = pq(rng);
      while (b == a) {
        b = pq(rng);
      }
      qc.add(GateKind::CX, {a, b});
    } else {
      qc.add(GateKind::H, {rng() % n});
    }
  }
  return qc;
}

void row(const char* name, const CouplingMap& cm, std::size_t trials) {
  SabreOptions opt;
  opt.trials = trials;
  opt.seed = 0;
  const SabreRouter router(cm, opt);

  std::size_t two_q = 0;
  std::size_t trivial = 0;
  std::size_t with_layout = 0;

  for (std::uint64_t s = 0; s < 30; ++s) {
    const Circuit qc = synth(cm.num_qubits(), 60, 1000 + s);
    for (const Gate& g : qc.gates()) {
      if (g.qubits.size() == 2) {
        ++two_q;
      }
    }
    trivial += router.route(qc).swaps_added;
    with_layout += router.compile(qc).swaps_added;
  }

  if (trivial == 0) {
    // all-to-all: the control. Zero swaps in, zero swaps out, nothing to save.
    std::printf("  %-12s %8zu %11zu %13zu %12s\n", name, two_q, trivial,
                with_layout, "--");
    return;
  }
  std::printf("  %-12s %8zu %11zu %13zu %11.0f%%\n", name, two_q, trivial,
              with_layout,
              100.0 * (1.0 - static_cast<double>(with_layout) /
                                 static_cast<double>(trivial)));
}

}  // namespace

int main() {
  std::printf("30 circuits x 60 gates. SWAPs inserted; fewer is better.\n\n");
  for (const std::size_t trials : {std::size_t{1}, std::size_t{8}}) {
    std::printf("trials = %zu\n", trials);
    std::printf("  %-12s %8s %11s %13s %12s\n", "device", "2q in", "identity",
                "sabre-layout", "saved");
    row("line(7)", CouplingMap::line(7), trials);
    row("ring(7)", CouplingMap::ring(7), trials);
    row("grid(2x4)", CouplingMap::grid(2, 4), trials);
    row("all-to-all", CouplingMap::all_to_all(7), trials);
    std::printf("\n");
  }
  return 0;
}
