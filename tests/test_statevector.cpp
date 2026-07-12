#include <gtest/gtest.h>

#include <cmath>
#include <complex>
#include <cstddef>
#include <stdexcept>
#include <type_traits>
#include <utility>
#include <vector>

#include "qcdsl/circuit.hpp"
#include "qcdsl/sim/statevector.hpp"

namespace {

using qcdsl::Circuit;
using qcdsl::GateKind;

// The simulator is a template. Both instantiations are shipped, so both are
// tested -- the whole battery runs against double AND float. float halves the
// memory and is what the SIMD kernel will want; an untested float path is an
// untested half of the library.
template <typename Real>
class Sim : public ::testing::Test {
 protected:
  using SV = qcdsl::Statevector<Real>;
  using C = std::complex<Real>;

  static Real tol() {
    return std::is_same_v<Real, float> ? Real(2e-6) : Real(1e-12);
  }
  static Real r2() { return Real(1) / std::sqrt(Real(2)); }

  static void ExpectAmp(const SV& sv, std::size_t i, Real re, Real im) {
    const C got = sv.amplitude(i);
    EXPECT_NEAR(got.real(), re, tol()) << "real part at index " << i;
    EXPECT_NEAR(got.imag(), im, tol()) << "imag part at index " << i;
  }

  static std::vector<GateKind> one_qubit_kinds() {
    return {GateKind::I,   GateKind::X,  GateKind::Y,   GateKind::Z,
            GateKind::H,   GateKind::S,  GateKind::Sdg, GateKind::T,
            GateKind::Tdg, GateKind::RX, GateKind::RY,  GateKind::RZ};
  }
};

using ScalarTypes = ::testing::Types<double, float>;
TYPED_TEST_SUITE(Sim, ScalarTypes);

TYPED_TEST(Sim, InitialisesToTheAllZeroState) {
  const qcdsl::Statevector<TypeParam> sv(3);
  EXPECT_EQ(sv.num_qubits(), 3u);
  EXPECT_EQ(sv.dim(), 8u);
  TestFixture::ExpectAmp(sv, 0, TypeParam(1), TypeParam(0));
  for (std::size_t i = 1; i < 8; ++i) {
    TestFixture::ExpectAmp(sv, i, TypeParam(0), TypeParam(0));
  }
  EXPECT_NEAR(sv.norm(), TypeParam(1), TestFixture::tol());
}

TYPED_TEST(Sim, RejectsZeroQubitsAndAbsurdSizes) {
  EXPECT_THROW(qcdsl::Statevector<TypeParam>(0), std::invalid_argument);
  EXPECT_THROW(qcdsl::Statevector<TypeParam>(31), std::invalid_argument);
}

TYPED_TEST(Sim, RejectsOutOfRangeQubitAndAmplitude) {
  qcdsl::Statevector<TypeParam> sv(2);
  EXPECT_THROW(sv.apply_1q(qcdsl::matrix_of<TypeParam>(GateKind::X), 2),
               std::out_of_range);
  EXPECT_THROW((void)sv.amplitude(4), std::out_of_range);
}

// THE ENDIANNESS TEST. Qubit q is bit q of the index, so qubit 0 is the LEAST
// significant bit -- Qiskit's convention. X on qubit 1 must land at index 2
// (binary 10), NOT index 1. If this fails, nothing will agree with Qiskit.
TYPED_TEST(Sim, QubitZeroIsTheLeastSignificantBit) {
  qcdsl::Statevector<TypeParam> a(2);
  a.apply_1q(qcdsl::matrix_of<TypeParam>(GateKind::X), 0);
  TestFixture::ExpectAmp(a, 1, TypeParam(1), TypeParam(0));
  TestFixture::ExpectAmp(a, 2, TypeParam(0), TypeParam(0));

  qcdsl::Statevector<TypeParam> b(2);
  b.apply_1q(qcdsl::matrix_of<TypeParam>(GateKind::X), 1);
  TestFixture::ExpectAmp(b, 1, TypeParam(0), TypeParam(0));
  TestFixture::ExpectAmp(b, 2, TypeParam(1), TypeParam(0));
}

// A gate on qubit 1 of a 3-qubit register couples the strided pairs
// (0,2), (1,3), (4,6), (5,7).
TYPED_TEST(Sim, StridedPairsAreCoupledCorrectly) {
  Circuit qc(3);
  qc.add(GateKind::X, {0}).add(GateKind::X, {2});  // -> index 5 (binary 101)
  qcdsl::Statevector<TypeParam> sv = qcdsl::simulate<TypeParam>(qc);
  TestFixture::ExpectAmp(sv, 5, TypeParam(1), TypeParam(0));

  sv.apply_1q(qcdsl::matrix_of<TypeParam>(GateKind::X), 1);  // 5 -> 7
  TestFixture::ExpectAmp(sv, 5, TypeParam(0), TypeParam(0));
  TestFixture::ExpectAmp(sv, 7, TypeParam(1), TypeParam(0));
}

TYPED_TEST(Sim, HadamardMakesAnEvenSuperposition) {
  Circuit qc(1);
  qc.add(GateKind::H, {0});
  const qcdsl::Statevector<TypeParam> sv = qcdsl::simulate<TypeParam>(qc);
  TestFixture::ExpectAmp(sv, 0, TestFixture::r2(), TypeParam(0));
  TestFixture::ExpectAmp(sv, 1, TestFixture::r2(), TypeParam(0));
  EXPECT_NEAR(sv.norm(), TypeParam(1), TestFixture::tol());
}

TYPED_TEST(Sim, BellPair) {
  Circuit qc(2);
  qc.add(GateKind::H, {0}).add(GateKind::CX, {0, 1});
  const qcdsl::Statevector<TypeParam> sv = qcdsl::simulate<TypeParam>(qc);
  TestFixture::ExpectAmp(sv, 0, TestFixture::r2(), TypeParam(0));
  TestFixture::ExpectAmp(sv, 1, TypeParam(0), TypeParam(0));
  TestFixture::ExpectAmp(sv, 2, TypeParam(0), TypeParam(0));
  TestFixture::ExpectAmp(sv, 3, TestFixture::r2(), TypeParam(0));
}

TYPED_TEST(Sim, GhzState) {
  Circuit qc(3);
  qc.add(GateKind::H, {0}).add(GateKind::CX, {0, 1}).add(GateKind::CX, {1, 2});
  const qcdsl::Statevector<TypeParam> sv = qcdsl::simulate<TypeParam>(qc);
  TestFixture::ExpectAmp(sv, 0, TestFixture::r2(), TypeParam(0));
  TestFixture::ExpectAmp(sv, 7, TestFixture::r2(), TypeParam(0));
  EXPECT_NEAR(sv.norm(), TypeParam(1), TestFixture::tol());
}

TYPED_TEST(Sim, ControlledGateIsInertWhenControlIsZero) {
  Circuit qc(2);
  qc.add(GateKind::CX, {0, 1});  // control q0 is |0> -> no-op
  const qcdsl::Statevector<TypeParam> sv = qcdsl::simulate<TypeParam>(qc);
  TestFixture::ExpectAmp(sv, 0, TypeParam(1), TypeParam(0));
  TestFixture::ExpectAmp(sv, 3, TypeParam(0), TypeParam(0));
}

TYPED_TEST(Sim, CzFlipsThePhaseOfTheOneOneAmplitudeOnly) {
  Circuit qc(2);
  qc.add(GateKind::X, {0}).add(GateKind::X, {1}).add(GateKind::CZ, {0, 1});
  const qcdsl::Statevector<TypeParam> sv = qcdsl::simulate<TypeParam>(qc);
  TestFixture::ExpectAmp(sv, 3, TypeParam(-1), TypeParam(0));
}

TYPED_TEST(Sim, SwapExchangesTheWires) {
  Circuit qc(2);
  qc.add(GateKind::X, {0}).add(GateKind::SWAP, {0, 1});
  const qcdsl::Statevector<TypeParam> sv = qcdsl::simulate<TypeParam>(qc);
  TestFixture::ExpectAmp(sv, 1, TypeParam(0), TypeParam(0));
  TestFixture::ExpectAmp(sv, 2, TypeParam(1), TypeParam(0));
}

TYPED_TEST(Sim, SwapWithItselfIsANoOp) {
  qcdsl::Statevector<TypeParam> sv(2);
  sv.apply_1q(qcdsl::matrix_of<TypeParam>(GateKind::X), 0);
  sv.apply_swap(0, 0);
  TestFixture::ExpectAmp(sv, 1, TypeParam(1), TypeParam(0));
}

TYPED_TEST(Sim, RzCarriesTheQiskitGlobalPhase) {
  const TypeParam theta = TypeParam(0.7);
  Circuit qc(1);
  qc.add(GateKind::RZ, {0}, static_cast<double>(theta));
  const qcdsl::Statevector<TypeParam> sv = qcdsl::simulate<TypeParam>(qc);
  // RZ|0> = e^{-i*theta/2}|0>, matching qiskit's RZGate.
  TestFixture::ExpectAmp(sv, 0, std::cos(theta / 2), -std::sin(theta / 2));
  TestFixture::ExpectAmp(sv, 1, TypeParam(0), TypeParam(0));
}

// Every single-qubit matrix the library can produce must be unitary: U^dag U =
// I. This walks every arm of the matrix_of switch, for every scalar type.
TYPED_TEST(Sim, EverySingleQubitMatrixIsUnitary) {
  using C = std::complex<TypeParam>;
  for (const GateKind k : TestFixture::one_qubit_kinds()) {
    const qcdsl::Matrix2<TypeParam> u =
        qcdsl::matrix_of<TypeParam>(k, TypeParam(0.83));
    // (U^dag U)_{rc} = sum_k conj(U_{kr}) * U_{kc}
    const C m00 = std::conj(u[0]) * u[0] + std::conj(u[2]) * u[2];
    const C m01 = std::conj(u[0]) * u[1] + std::conj(u[2]) * u[3];
    const C m10 = std::conj(u[1]) * u[0] + std::conj(u[3]) * u[2];
    const C m11 = std::conj(u[1]) * u[1] + std::conj(u[3]) * u[3];
    const TypeParam t = TestFixture::tol();
    EXPECT_NEAR(std::abs(m00 - C(1, 0)), TypeParam(0), t)
        << qcdsl::to_string(k);
    EXPECT_NEAR(std::abs(m01), TypeParam(0), t) << qcdsl::to_string(k);
    EXPECT_NEAR(std::abs(m10), TypeParam(0), t) << qcdsl::to_string(k);
    EXPECT_NEAR(std::abs(m11 - C(1, 0)), TypeParam(0), t)
        << qcdsl::to_string(k);
  }
}

// Every single-qubit gate, applied to a non-trivial entangled state, must
// preserve the norm. Exercises the full matrix_of switch through apply_1q.
TYPED_TEST(Sim, EverySingleQubitGatePreservesTheNorm) {
  for (const GateKind k : TestFixture::one_qubit_kinds()) {
    Circuit qc(3);
    qc.add(GateKind::H, {0}).add(GateKind::CX, {0, 1}).add(GateKind::T, {2});
    qcdsl::Statevector<TypeParam> sv = qcdsl::simulate<TypeParam>(qc);
    sv.apply(qcdsl::Gate(k, {1}, 0.41));
    EXPECT_NEAR(sv.norm(), TypeParam(1), TestFixture::tol())
        << "gate " << qcdsl::to_string(k);
  }
}

TYPED_TEST(Sim, EveryGateFollowedByItsInverseIsTheIdentity) {
  const std::vector<std::pair<GateKind, GateKind>> pairs = {
      {GateKind::X, GateKind::X},   {GateKind::Y, GateKind::Y},
      {GateKind::Z, GateKind::Z},   {GateKind::H, GateKind::H},
      {GateKind::S, GateKind::Sdg}, {GateKind::T, GateKind::Tdg},
      {GateKind::Sdg, GateKind::S}, {GateKind::Tdg, GateKind::T},
  };
  for (const auto& p : pairs) {
    Circuit qc(2);
    qc.add(GateKind::H, {0}).add(GateKind::CX, {0, 1});
    qcdsl::Statevector<TypeParam> before = qcdsl::simulate<TypeParam>(qc);
    qcdsl::Statevector<TypeParam> after = before;
    after.apply_1q(qcdsl::matrix_of<TypeParam>(p.first), 1);
    after.apply_1q(qcdsl::matrix_of<TypeParam>(p.second), 1);
    for (std::size_t i = 0; i < before.dim(); ++i) {
      EXPECT_NEAR(std::abs(after.amplitude(i) - before.amplitude(i)),
                  TypeParam(0), TestFixture::tol())
          << "gate " << qcdsl::to_string(p.first) << " at index " << i;
    }
  }
}

TYPED_TEST(Sim, RotationsAreUnitaryAndPreserveNorm) {
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
  const qcdsl::Statevector<TypeParam> sv = qcdsl::simulate<TypeParam>(qc);
  EXPECT_NEAR(sv.norm(), TypeParam(1), TestFixture::tol());
  EXPECT_EQ(sv.dim(), 16u);
}

TYPED_TEST(Sim, ProbabilitiesSumToOne) {
  Circuit qc(3);
  qc.add(GateKind::H, {0}).add(GateKind::H, {1}).add(GateKind::H, {2});
  const qcdsl::Statevector<TypeParam> sv = qcdsl::simulate<TypeParam>(qc);
  const std::vector<TypeParam> p = sv.probabilities();
  ASSERT_EQ(p.size(), 8u);
  TypeParam total = 0;
  for (const TypeParam x : p) {
    EXPECT_NEAR(x, TypeParam(0.125), TestFixture::tol());
    total += x;
  }
  EXPECT_NEAR(total, TypeParam(1), TestFixture::tol());
}

TYPED_TEST(Sim, AmplitudesAreExposedInIndexOrder) {
  Circuit qc(2);
  qc.add(GateKind::X, {1});
  const qcdsl::Statevector<TypeParam> sv = qcdsl::simulate<TypeParam>(qc);
  const auto& a = sv.amplitudes();
  ASSERT_EQ(a.size(), 4u);
  EXPECT_NEAR(std::abs(a[2]), TypeParam(1), TestFixture::tol());
}

TYPED_TEST(Sim, MeasurementIsRejected) {
  Circuit qc(1);
  qc.add(GateKind::MEASURE, {0});
  EXPECT_THROW((void)qcdsl::simulate<TypeParam>(qc), std::invalid_argument);
}

TYPED_TEST(Sim, RunRejectsAQubitCountMismatch) {
  Circuit qc(3);
  qc.add(GateKind::H, {0});
  qcdsl::Statevector<TypeParam> sv(2);
  EXPECT_THROW(sv.run(qc), std::invalid_argument);
}

TYPED_TEST(Sim, MatrixOfRejectsTwoQubitKinds) {
  EXPECT_THROW((void)qcdsl::matrix_of<TypeParam>(GateKind::CX),
               std::invalid_argument);
  EXPECT_THROW((void)qcdsl::matrix_of<TypeParam>(GateKind::CZ),
               std::invalid_argument);
  EXPECT_THROW((void)qcdsl::matrix_of<TypeParam>(GateKind::SWAP),
               std::invalid_argument);
  EXPECT_THROW((void)qcdsl::matrix_of<TypeParam>(GateKind::MEASURE),
               std::invalid_argument);
}

TYPED_TEST(Sim, ControlAndTargetMustDiffer) {
  qcdsl::Statevector<TypeParam> sv(2);
  EXPECT_THROW(
      sv.apply_controlled_1q(qcdsl::matrix_of<TypeParam>(GateKind::X), 1, 1),
      std::invalid_argument);
}

TYPED_TEST(Sim, ControlledGateRejectsAnOutOfRangeWire) {
  qcdsl::Statevector<TypeParam> sv(2);
  EXPECT_THROW(
      sv.apply_controlled_1q(qcdsl::matrix_of<TypeParam>(GateKind::X), 5, 0),
      std::out_of_range);
  EXPECT_THROW(sv.apply_swap(0, 9), std::out_of_range);
}

}  // namespace
