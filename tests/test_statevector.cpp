#include <gtest/gtest.h>

#include <cmath>
#include <complex>
#include <stdexcept>
#include <vector>

#include "qcdsl/circuit.hpp"
#include "qcdsl/sim/statevector.hpp"

namespace {

using qcdsl::Circuit;
using qcdsl::GateKind;
using qcdsl::Statevector;

using Cd = std::complex<double>;

constexpr double kTol = 1e-12;
const double kR2 = 1.0 / std::sqrt(2.0);

void ExpectAmp(const Statevector<>& sv, std::size_t i, Cd want) {
  const Cd got = sv.amplitude(i);
  EXPECT_NEAR(got.real(), want.real(), kTol) << "real part at index " << i;
  EXPECT_NEAR(got.imag(), want.imag(), kTol) << "imag part at index " << i;
}

TEST(Statevector, InitialisesToTheAllZeroState) {
  const Statevector<> sv(3);
  EXPECT_EQ(sv.num_qubits(), 3u);
  EXPECT_EQ(sv.dim(), 8u);
  ExpectAmp(sv, 0, Cd(1, 0));
  for (std::size_t i = 1; i < 8; ++i) {
    ExpectAmp(sv, i, Cd(0, 0));
  }
  EXPECT_NEAR(sv.norm(), 1.0, kTol);
}

TEST(Statevector, RejectsZeroQubitsAndAbsurdSizes) {
  EXPECT_THROW(Statevector<>(0), std::invalid_argument);
  EXPECT_THROW(Statevector<>(31), std::invalid_argument);
}

TEST(Statevector, RejectsOutOfRangeQubitAndAmplitude) {
  Statevector<> sv(2);
  EXPECT_THROW(sv.apply_1q(qcdsl::matrix_of<double>(GateKind::X), 2),
               std::out_of_range);
  EXPECT_THROW((void)sv.amplitude(4), std::out_of_range);
}

// THE ENDIANNESS TEST. Qubit q is bit q of the index, so qubit 0 is the LEAST
// significant bit -- Qiskit's convention. X on qubit 1 must land at index 2
// (binary 10), NOT index 1. If this fails, nothing will ever agree with Qiskit.
TEST(Statevector, QubitZeroIsTheLeastSignificantBit) {
  Statevector<> a(2);
  a.apply_1q(qcdsl::matrix_of<double>(GateKind::X), 0);
  ExpectAmp(a, 1, Cd(1, 0));
  ExpectAmp(a, 2, Cd(0, 0));

  Statevector<> b(2);
  b.apply_1q(qcdsl::matrix_of<double>(GateKind::X), 1);
  ExpectAmp(b, 1, Cd(0, 0));
  ExpectAmp(b, 2, Cd(1, 0));
}

// Applying a gate to qubit 1 of a 3-qubit register must couple the pairs
// (0,2), (1,3), (4,6), (5,7) -- and leave the stride-2 structure intact.
TEST(Statevector, StridedPairsAreCoupledCorrectly) {
  Circuit qc(3);
  qc.add(GateKind::X, {0}).add(GateKind::X, {2});  // -> index 5 (binary 101)
  Statevector<> sv = qcdsl::simulate(qc);
  ExpectAmp(sv, 5, Cd(1, 0));

  sv.apply_1q(qcdsl::matrix_of<double>(GateKind::X), 1);  // 5 -> 7
  ExpectAmp(sv, 5, Cd(0, 0));
  ExpectAmp(sv, 7, Cd(1, 0));
}

TEST(Statevector, HadamardMakesAnEvenSuperposition) {
  Circuit qc(1);
  qc.add(GateKind::H, {0});
  const Statevector<> sv = qcdsl::simulate(qc);
  ExpectAmp(sv, 0, Cd(kR2, 0));
  ExpectAmp(sv, 1, Cd(kR2, 0));
  EXPECT_NEAR(sv.norm(), 1.0, kTol);
}

TEST(Statevector, BellPair) {
  Circuit qc(2);
  qc.add(GateKind::H, {0}).add(GateKind::CX, {0, 1});
  const Statevector<> sv = qcdsl::simulate(qc);
  ExpectAmp(sv, 0, Cd(kR2, 0));
  ExpectAmp(sv, 1, Cd(0, 0));
  ExpectAmp(sv, 2, Cd(0, 0));
  ExpectAmp(sv, 3, Cd(kR2, 0));
}

TEST(Statevector, GhzState) {
  Circuit qc(3);
  qc.add(GateKind::H, {0}).add(GateKind::CX, {0, 1}).add(GateKind::CX, {1, 2});
  const Statevector<> sv = qcdsl::simulate(qc);
  ExpectAmp(sv, 0, Cd(kR2, 0));
  ExpectAmp(sv, 7, Cd(kR2, 0));
  EXPECT_NEAR(sv.norm(), 1.0, kTol);
}

TEST(Statevector, ControlledGateIsInertWhenControlIsZero) {
  Circuit qc(2);
  qc.add(GateKind::CX, {0, 1});  // control q0 is |0> -> no-op
  const Statevector<> sv = qcdsl::simulate(qc);
  ExpectAmp(sv, 0, Cd(1, 0));
  ExpectAmp(sv, 3, Cd(0, 0));
}

TEST(Statevector, CzFlipsThePhaseOfTheOneOneAmplitudeOnly) {
  Circuit qc(2);
  qc.add(GateKind::X, {0}).add(GateKind::X, {1}).add(GateKind::CZ, {0, 1});
  const Statevector<> sv = qcdsl::simulate(qc);
  ExpectAmp(sv, 3, Cd(-1, 0));
}

TEST(Statevector, SwapExchangesTheWires) {
  Circuit qc(2);
  qc.add(GateKind::X, {0}).add(GateKind::SWAP, {0, 1});
  const Statevector<> sv = qcdsl::simulate(qc);
  ExpectAmp(sv, 1, Cd(0, 0));
  ExpectAmp(sv, 2, Cd(1, 0));
}

TEST(Statevector, RzCarriesTheQiskitGlobalPhase) {
  const double theta = 0.7;
  Circuit qc(1);
  qc.add(GateKind::RZ, {0}, theta);
  const Statevector<> sv = qcdsl::simulate(qc);
  // RZ|0> = e^{-i*theta/2}|0>, matching qiskit's RZGate.
  ExpectAmp(sv, 0, Cd(std::cos(theta / 2), -std::sin(theta / 2)));
  ExpectAmp(sv, 1, Cd(0, 0));
}

TEST(Statevector, EveryGateFollowedByItsInverseIsTheIdentity) {
  const std::vector<std::pair<GateKind, GateKind>> pairs = {
      {GateKind::X, GateKind::X},   {GateKind::Y, GateKind::Y},
      {GateKind::Z, GateKind::Z},   {GateKind::H, GateKind::H},
      {GateKind::S, GateKind::Sdg}, {GateKind::T, GateKind::Tdg},
      {GateKind::Sdg, GateKind::S}, {GateKind::Tdg, GateKind::T},
  };
  for (const auto& [g, ginv] : pairs) {
    Circuit qc(2);
    qc.add(GateKind::H, {0}).add(GateKind::CX, {0, 1});  // non-trivial start
    Statevector<> before = qcdsl::simulate(qc);
    Statevector<> after = before;
    after.apply_1q(qcdsl::matrix_of<double>(g), 1);
    after.apply_1q(qcdsl::matrix_of<double>(ginv), 1);
    for (std::size_t i = 0; i < before.dim(); ++i) {
      EXPECT_NEAR(std::abs(after.amplitude(i) - before.amplitude(i)), 0.0, kTol)
          << "gate " << qcdsl::to_string(g) << " at index " << i;
    }
  }
}

TEST(Statevector, RotationsAreUnitaryAndPreserveNorm) {
  Circuit qc(4);
  qc.add(GateKind::H, {0})
      .add(GateKind::RX, {1}, 0.3)
      .add(GateKind::CX, {0, 1})
      .add(GateKind::RY, {2}, -1.1)
      .add(GateKind::CX, {1, 2})
      .add(GateKind::RZ, {3}, 2.4)
      .add(GateKind::T, {0})
      .add(GateKind::CZ, {2, 3})
      .add(GateKind::SWAP, {0, 3})
      .add(GateKind::H, {2});
  const Statevector<> sv = qcdsl::simulate(qc);
  EXPECT_NEAR(sv.norm(), 1.0, 1e-10);
  EXPECT_EQ(sv.dim(), 16u);
}

TEST(Statevector, ProbabilitiesSumToOne) {
  Circuit qc(3);
  qc.add(GateKind::H, {0}).add(GateKind::H, {1}).add(GateKind::H, {2});
  const Statevector<> sv = qcdsl::simulate(qc);
  const std::vector<double> p = sv.probabilities();
  ASSERT_EQ(p.size(), 8u);
  double total = 0;
  for (const double x : p) {
    EXPECT_NEAR(x, 0.125, kTol);
    total += x;
  }
  EXPECT_NEAR(total, 1.0, kTol);
}

TEST(Statevector, MeasurementIsRejected) {
  Circuit qc(1);
  qc.add(GateKind::MEASURE, {0});
  EXPECT_THROW((void)qcdsl::simulate(qc), std::invalid_argument);
}

TEST(Statevector, RunRejectsAQubitCountMismatch) {
  Circuit qc(3);
  qc.add(GateKind::H, {0});
  Statevector<> sv(2);
  EXPECT_THROW(sv.run(qc), std::invalid_argument);
}

TEST(Statevector, MatrixOfRejectsTwoQubitKinds) {
  EXPECT_THROW((void)qcdsl::matrix_of<double>(GateKind::CX),
               std::invalid_argument);
  EXPECT_THROW((void)qcdsl::matrix_of<double>(GateKind::MEASURE),
               std::invalid_argument);
}

TEST(Statevector, ControlAndTargetMustDiffer) {
  Statevector<> sv(2);
  EXPECT_THROW(
      sv.apply_controlled_1q(qcdsl::matrix_of<double>(GateKind::X), 1, 1),
      std::invalid_argument);
}

TEST(Statevector, SwapWithItselfIsANoOp) {
  Statevector<> sv(2);
  sv.apply_1q(qcdsl::matrix_of<double>(GateKind::X), 0);
  sv.apply_swap(0, 0);
  ExpectAmp(sv, 1, Cd(1, 0));
}

// The vector is templated on the scalar type. float halves the memory and is
// what the SIMD kernel will want; it must still be correct.
TEST(Statevector, WorksInSinglePrecision) {
  Circuit qc(2);
  qc.add(GateKind::H, {0}).add(GateKind::CX, {0, 1});
  Statevector<float> sv(2);
  sv.run(qc);
  EXPECT_NEAR(sv.amplitude(0).real(), static_cast<float>(kR2), 1e-6F);
  EXPECT_NEAR(sv.amplitude(3).real(), static_cast<float>(kR2), 1e-6F);
  EXPECT_NEAR(sv.norm(), 1.0F, 1e-6F);
}

}  // namespace
