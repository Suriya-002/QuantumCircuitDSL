// Reproduces the optimisation table in the README.
//
//   cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release
//   -DQCDSL_BUILD_BENCH=ON cmake --build build --target qcdsl_bench_optimise
//   ./build/bench/qcdsl_bench_optimise
//
// Synthetic circuits, but not uniformly random ones: real unoptimised compiler
// output repeats itself, so a `redundancy` knob controls how often a gate is
// emitted twice in a row. At redundancy 0 there is essentially nothing to
// cancel, which is the worst case for an optimiser and the honest one to show.

#include <cstdint>
#include <cstdio>
#include <memory>
#include <random>
#include <vector>

#include "qcdsl/qcdsl.hpp"

using namespace qcdsl;

namespace {

Circuit synth(std::size_t n, std::size_t m, std::uint64_t seed,
              double redundancy) {
  std::mt19937_64 rng(seed);
  const std::vector<GateKind> one = {GateKind::H, GateKind::T,   GateKind::Tdg,
                                     GateKind::S, GateKind::Sdg, GateKind::X,
                                     GateKind::Y, GateKind::Z};
  const std::vector<GateKind> rot = {GateKind::RX, GateKind::RY, GateKind::RZ};
  const std::vector<GateKind> two = {GateKind::CX, GateKind::CZ,
                                     GateKind::SWAP};
  std::uniform_real_distribution<double> ang(-3.0, 3.0);
  std::uniform_real_distribution<double> u01(0.0, 1.0);

  Circuit qc(n);
  Gate last(GateKind::I, {0});
  bool have_last = false;
  while (qc.size() < m) {
    if (have_last && u01(rng) < redundancy) {
      qc.add(last);
      continue;
    }
    Gate g(GateKind::I, {0});
    const auto roll = rng() % 100;
    if (n >= 2 && roll < 30) {
      std::uniform_int_distribution<std::size_t> pq(0, n - 1);
      Qubit a = pq(rng);
      Qubit b = pq(rng);
      while (b == a) {
        b = pq(rng);
      }
      g = Gate(two[rng() % two.size()], {a, b});
    } else if (roll < 60) {
      g = Gate(rot[rng() % rot.size()], {rng() % n}, ang(rng));
    } else {
      g = Gate(one[rng() % one.size()], {rng() % n});
    }
    qc.add(g);
    last = g;
    have_last = true;
  }
  return qc;
}

PassManager optimise_only() {
  PassManager pm;
  pm.add(std::make_shared<MergeRotations>())
      .add(std::make_shared<RemoveIdentities>())
      .add(std::make_shared<CancelInversePairs>());
  return pm;
}

PassManager lower_first() {
  PassManager pm;
  pm.add(std::make_shared<DecomposeToCx>())
      .add(std::make_shared<MergeRotations>())
      .add(std::make_shared<RemoveIdentities>())
      .add(std::make_shared<CancelInversePairs>());
  return pm;
}

PassManager optimise_then_lower() {
  PassManager pm;
  pm.add(std::make_shared<MergeRotations>())
      .add(std::make_shared<RemoveIdentities>())
      .add(std::make_shared<CancelInversePairs>())
      .add(std::make_shared<DecomposeToCx>())
      .add(std::make_shared<MergeRotations>())
      .add(std::make_shared<RemoveIdentities>())
      .add(std::make_shared<CancelInversePairs>());
  return pm;
}

void bench(const char* label, PassManager (*make)(), double redundancy) {
  std::size_t gb = 0;
  std::size_t ga = 0;
  std::size_t db = 0;
  std::size_t da = 0;
  std::size_t illegal = 0;

  for (std::uint64_t s = 0; s < 300; ++s) {
    const Circuit qc = synth(6, 200, s, redundancy);
    PassManager pm = make();
    const Circuit out = pm.run_to_fixed_point(qc);
    for (const Gate& g : out.gates()) {
      if (g.kind == GateKind::CZ || g.kind == GateKind::SWAP) {
        ++illegal;
      }
    }
    gb += qc.size();
    ga += out.size();
    db += Dag(qc).depth();
    da += Dag(out).depth();
  }
  std::printf("  %-34s %8.1f%% %10.1f%%   %s\n", label,
              100.0 * (1.0 - static_cast<double>(ga) / static_cast<double>(gb)),
              100.0 * (1.0 - static_cast<double>(da) / static_cast<double>(db)),
              illegal == 0 ? "yes" : "no");
}

void table(const char* title, double redundancy) {
  std::printf("\n%s\n", title);
  std::printf("  %-34s %9s %11s   %s\n", "pipeline", "gates", "depth",
              "cx-only");
  bench("optimise only", optimise_only, redundancy);
  bench("lower first", lower_first, redundancy);
  bench("optimise -> lower -> optimise", optimise_then_lower, redundancy);
}

}  // namespace

int main() {
  std::printf(
      "300 circuits x 6 qubits x 200 gates each. Positive = reduction.\n");
  table("Redundant circuits (what unoptimised output looks like):", 0.35);
  table("Uniformly random circuits (nothing to cancel -- worst case):", 0.0);
  return 0;
}
