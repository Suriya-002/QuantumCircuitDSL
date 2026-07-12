#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <random>
#include <string>
#include <vector>

#include "qcdsl/circuit.hpp"
#include "qcdsl/sim/kernels.hpp"
#include "qcdsl/sim/statevector.hpp"

namespace {

using qcdsl::Circuit;
using qcdsl::GateKind;
using qcdsl::Kernel;
using qcdsl::Qubit;
using qcdsl::Statevector;

// FMA fuses a multiply and an add into ONE rounding step, so the vector kernel
// does not reproduce the scalar kernel bit for bit -- it is very slightly MORE
// accurate. Equality is therefore numerical, not exact, and the tolerance is
// tight enough that a real bug cannot hide underneath it.
constexpr double kTol = 1e-13;

Circuit random_circuit(std::size_t n, std::size_t m, std::uint64_t seed) {
  std::mt19937_64 rng(seed);
  const std::vector<GateKind> one = {GateKind::H, GateKind::T, GateKind::S,
                                     GateKind::X, GateKind::Y, GateKind::Z};
  const std::vector<GateKind> rot = {GateKind::RX, GateKind::RY, GateKind::RZ};
  const std::vector<GateKind> two = {GateKind::CX, GateKind::CZ,
                                     GateKind::SWAP};
  std::uniform_real_distribution<double> ang(-3.0, 3.0);

  Circuit qc(n);
  for (std::size_t i = 0; i < m; ++i) {
    const auto roll = rng() % 100;
    if (n >= 2 && roll < 30) {
      std::uniform_int_distribution<std::size_t> pq(0, n - 1);
      Qubit a = pq(rng);
      Qubit b = pq(rng);
      while (b == a) {
        b = pq(rng);
      }
      qc.add(two[rng() % two.size()], {a, b});
    } else if (roll < 60) {
      qc.add(rot[rng() % rot.size()], {rng() % n}, ang(rng));
    } else {
      qc.add(one[rng() % one.size()], {rng() % n});
    }
  }
  return qc;
}

void ExpectAgrees(const Statevector<>& a, const Statevector<>& b,
                  const char* what) {
  ASSERT_EQ(a.dim(), b.dim());
  double worst = 0.0;
  for (std::size_t i = 0; i < a.dim(); ++i) {
    worst = std::max(worst, std::abs(a.amplitude(i) - b.amplitude(i)));
  }
  EXPECT_LT(worst, kTol) << std::string(what) << ": worst amplitude error "
                         << worst;
}

Statevector<> run_with(const Circuit& qc, Kernel k) {
  Statevector<> sv(qc.num_qubits());
  sv.set_kernel(k);
  sv.run(qc);
  return sv;
}

TEST(Kernels, TheDefaultIsAuto) {
  const Statevector<> sv(4);
  EXPECT_EQ(sv.kernel(), Kernel::Auto);
}

TEST(Kernels, ReportsWhetherTheVectorPathApplies) {
  // Targets 0 and 1 cannot use contiguous 512-bit loads: the pair partners
  // interleave inside a single register. Everything from target 2 up can.
  if (qcdsl::kernel::has_avx512()) {
    EXPECT_FALSE(qcdsl::kernel::simd_applies<double>(1024, 0));
    EXPECT_FALSE(qcdsl::kernel::simd_applies<double>(1024, 1));
    EXPECT_TRUE(qcdsl::kernel::simd_applies<double>(1024, 2));
    EXPECT_TRUE(qcdsl::kernel::simd_applies<double>(1024, 9));
  }
  // float has no vector kernel yet, whatever the CPU offers.
  EXPECT_FALSE(qcdsl::kernel::simd_applies<float>(1024, 4));
}

// THE TEST THAT MATTERS. Every kernel must produce what the scalar reference
// produces -- on every target qubit, including the two that fall back.
TEST(Kernels, EveryKernelAgreesWithScalarOnEveryTarget) {
  for (std::size_t n = 3; n <= 12; ++n) {
    for (Qubit t = 0; t < n; ++t) {
      Circuit qc(n);
      for (std::size_t q = 0; q < n; ++q) {
        qc.add(GateKind::H, {q});
        qc.add(GateKind::T, {q});
      }
      qc.add(GateKind::RY, {t}, 0.83);
      qc.add(GateKind::RZ, {t}, -1.7);
      qc.add(GateKind::H, {t});

      const Statevector<> scalar = run_with(qc, Kernel::Scalar);
      ExpectAgrees(scalar, run_with(qc, Kernel::Simd), "simd");
      ExpectAgrees(scalar, run_with(qc, Kernel::Parallel), "parallel");
      ExpectAgrees(scalar, run_with(qc, Kernel::Auto), "auto");
    }
  }
}

TEST(Kernels, EveryKernelAgreesWithScalarOnRandomCircuits) {
  for (std::uint64_t seed = 0; seed < 20; ++seed) {
    const std::size_t n = 4 + (seed % 8);
    const Circuit qc = random_circuit(n, 60, seed);
    const Statevector<> scalar = run_with(qc, Kernel::Scalar);
    ExpectAgrees(scalar, run_with(qc, Kernel::Simd), "simd");
    ExpectAgrees(scalar, run_with(qc, Kernel::Parallel), "parallel");
    ExpectAgrees(scalar, run_with(qc, Kernel::Auto), "auto");
  }
}

// A vector far too big to fit in cache, where the parallel kernel is at its
// least useful -- and must still be exactly correct.
TEST(Kernels, AgreementHoldsOnALargeOutOfCacheVector) {
  const std::size_t n = 21;  // 2 M amplitudes, 32 MB
  const Circuit qc = random_circuit(n, 12, 99);
  const Statevector<> scalar = run_with(qc, Kernel::Scalar);
  ExpectAgrees(scalar, run_with(qc, Kernel::Parallel),
               "parallel, out of cache");
  ExpectAgrees(scalar, run_with(qc, Kernel::Auto), "auto, out of cache");
}

// The flat-index parallel kernel exists because the obvious block-loop version
// has NO iterations to hand out when the target qubit is high: blocks =
// dim >> (t+1), which is 1 at t = n-1. Pin the worst case explicitly.
TEST(Kernels, ParallelKernelIsCorrectForTheHighestTargets) {
  const std::size_t n = 12;
  for (const Qubit t : {n - 3, n - 2, n - 1}) {
    Circuit qc(n);
    for (std::size_t q = 0; q < n; ++q) {
      qc.add(GateKind::H, {q});
    }
    qc.add(GateKind::RY, {t}, 1.1);
    qc.add(GateKind::T, {t});
    ExpectAgrees(run_with(qc, Kernel::Scalar), run_with(qc, Kernel::Parallel),
                 "parallel at a high target");
  }
}

TEST(Kernels, NormIsPreservedByEveryKernel) {
  const Circuit qc = random_circuit(10, 80, 5);
  for (const Kernel k :
       {Kernel::Scalar, Kernel::Simd, Kernel::Parallel, Kernel::Auto}) {
    EXPECT_NEAR(run_with(qc, k).norm(), 1.0, 1e-10);
  }
}

// Asking for Simd on a machine that cannot run it, or on a target it cannot
// handle, must give the right answer anyway -- not a crash and not garbage.
TEST(Kernels, RequestingSimdIsAlwaysSafe) {
  Circuit qc(2);
  qc.add(GateKind::H, {0}).add(GateKind::CX, {0, 1});
  const Statevector<> sv = run_with(qc, Kernel::Simd);
  EXPECT_NEAR(std::abs(sv.amplitude(0)), 1.0 / std::sqrt(2.0), 1e-12);
  EXPECT_NEAR(std::abs(sv.amplitude(3)), 1.0 / std::sqrt(2.0), 1e-12);
  EXPECT_NEAR(sv.norm(), 1.0, 1e-12);
}

TEST(Kernels, FloatAlwaysUsesTheScalarPath) {
  Circuit qc(6);
  qc.add(GateKind::H, {0})
      .add(GateKind::CX, {0, 1})
      .add(GateKind::RZ, {4}, 0.5);
  Statevector<float> sv(6);
  sv.set_kernel(Kernel::Simd);  // ignored: no float vector kernel
  sv.run(qc);
  EXPECT_NEAR(sv.norm(), 1.0F, 1e-6F);
}

}  // namespace
