// Reproduces the simulation table in the README.
//
//   cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release
//   -DQCDSL_BUILD_BENCH=ON cmake --build build --target qcdsl_bench_simulate
//   ./build/bench/qcdsl_bench_simulate            # full sweep
//   ./build/bench/qcdsl_bench_simulate 18         # stop at 18 qubits (for
//   profilers)
//
// The scalar kernel is the baseline. It is also the kernel the other two are
// validated against, so "speedup" here always means speedup over code known to
// be correct.
//
// The GB/s column is the point. Each amplitude pair moves 64 bytes -- two
// 16-byte loads and two 16-byte stores -- so the achieved bandwidth is
// computable, not guessable. When it stops rising, the kernel has stopped being
// compute-bound and no amount of vector width or threads will help.

#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <vector>

#include "qcdsl/qcdsl.hpp"

#ifdef _OPENMP
#include <omp.h>
#endif

using namespace qcdsl;
using Clock = std::chrono::steady_clock;

namespace {

/// Single-qubit gates only, spread over every wire: this measures the KERNEL,
/// not the dispatch around it.
Circuit workload(std::size_t n, std::size_t gates, std::uint64_t seed) {
  std::mt19937_64 rng(seed);
  const std::vector<GateKind> kinds = {GateKind::H,  GateKind::T,
                                       GateKind::S,  GateKind::RX,
                                       GateKind::RY, GateKind::RZ};
  std::uniform_real_distribution<double> ang(-3.0, 3.0);

  Circuit qc(n);
  for (std::size_t i = 0; i < gates; ++i) {
    qc.add(kinds[rng() % kinds.size()], {rng() % n}, ang(rng));
  }
  return qc;
}

double seconds(const Circuit& qc, Kernel k, int reps) {
  double best = 1e18;
  for (int r = 0; r < reps; ++r) {
    Statevector<> sv(qc.num_qubits());
    sv.set_kernel(k);
    const auto t0 = Clock::now();
    sv.run(qc);
    const auto t1 = Clock::now();
    if (sv.amplitude(0).real() == 12345.678) {  // keep it from being elided
      std::printf(" ");
    }
    const double s = std::chrono::duration<double>(t1 - t0).count();
    best = s < best ? s : best;  // best-of-N: fewest interruptions, least noise
  }
  return best;
}

/// Two 16-byte loads and two 16-byte stores per pair, dim/2 pairs per gate.
double gigabytes(std::size_t n, std::size_t gates) {
  const double dim = static_cast<double>(std::size_t(1) << n);
  return static_cast<double>(gates) * dim * 32.0 / 1e9;
}

}  // namespace

int main(int argc, char** argv) {
  const std::size_t max_qubits =
      (argc > 1) ? static_cast<std::size_t>(std::atoi(argv[1])) : 24;
  const std::size_t gates = 100;
  const int reps = 3;

  std::printf("qcdsl gate-kernel benchmark\n");
  std::printf("  AVX-512 available : %s\n",
              kernel::has_avx512() ? "yes" : "no");
#ifdef _OPENMP
  std::printf("  OpenMP threads    : %d\n", omp_get_max_threads());
#else
  std::printf("  OpenMP threads    : 1 (not compiled in)\n");
#endif
  std::printf("  workload          : %zu single-qubit gates, best of %d\n\n",
              gates, reps);

  std::printf("%6s %11s %9s %9s %9s   %6s %6s   %9s\n", "qubits", "amplitudes",
              "scalar", "avx512", "+omp", "simd", "omp", "GB/s");
  std::printf("%6s %11s %9s %9s %9s   %6s %6s   %9s\n", "", "", "(ms)", "(ms)",
              "(ms)", "x", "x", "(best)");

  for (std::size_t n = 14; n <= max_qubits; n += 2) {
    const Circuit qc = workload(n, gates, 1234 + n);
    const double s = seconds(qc, Kernel::Scalar, reps);
    const double v = seconds(qc, Kernel::Simd, reps);
    const double p = seconds(qc, Kernel::Parallel, reps);

    const double fastest = (p < v) ? p : v;
    const double bw = gigabytes(n, gates) / fastest;

    std::printf("%6zu %11zu %9.2f %9.2f %9.2f   %6.2f %6.2f   %9.1f\n", n,
                std::size_t(1) << n, s * 1e3, v * 1e3, p * 1e3, s / v, s / p,
                bw);
  }

  std::printf(
      "\nGB/s flattening = the kernel has left cache and is bandwidth-bound.\n"
      "Vector registers and threads cannot make DRAM faster.\n");
  return 0;
}
