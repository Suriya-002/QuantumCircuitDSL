#include <gtest/gtest.h>

#include <stdexcept>
#include <vector>

#include "qcdsl/qcdsl.hpp"

namespace {

using qcdsl::Gate;
using qcdsl::GateKind;

const std::vector<GateKind>& all_kinds() {
  static const std::vector<GateKind> kinds = {
      GateKind::I,   GateKind::X,  GateKind::Y,    GateKind::Z,
      GateKind::H,   GateKind::S,  GateKind::Sdg,  GateKind::T,
      GateKind::Tdg, GateKind::RX, GateKind::RY,   GateKind::RZ,
      GateKind::CX,  GateKind::CZ, GateKind::SWAP, GateKind::MEASURE};
  return kinds;
}

TEST(Version, IsExposed) { EXPECT_STREQ(qcdsl::version(), "0.1.0"); }

// Every enumerator must be handled by the QASM writer and by the arity table.
// If someone adds a GateKind and forgets the switch, this fails.
TEST(Gate, EveryKindHasANameAndAnArity) {
  for (const GateKind k : all_kinds()) {
    EXPECT_STRNE(qcdsl::to_string(k), "unknown");
    EXPECT_GE(qcdsl::arity(k), 1u);
    EXPECT_LE(qcdsl::arity(k), 2u);
  }
}

TEST(Gate, EveryKindConstructsAtItsDeclaredArity) {
  for (const GateKind k : all_kinds()) {
    const std::vector<qcdsl::Qubit> qs = (qcdsl::arity(k) == 2)
                                             ? std::vector<qcdsl::Qubit>{0, 1}
                                             : std::vector<qcdsl::Qubit>{0};
    const Gate g(k, qs, 0.25);
    EXPECT_EQ(g.kind, k);
    EXPECT_EQ(g.width(), qcdsl::arity(k));
  }
}

TEST(Gate, ArityMatchesGateKind) {
  EXPECT_EQ(qcdsl::arity(GateKind::H), 1u);
  EXPECT_EQ(qcdsl::arity(GateKind::CX), 2u);
  EXPECT_EQ(qcdsl::arity(GateKind::SWAP), 2u);
}

TEST(Gate, ParametricGatesAreFlagged) {
  EXPECT_TRUE(qcdsl::is_parametric(GateKind::RX));
  EXPECT_TRUE(qcdsl::is_parametric(GateKind::RY));
  EXPECT_TRUE(qcdsl::is_parametric(GateKind::RZ));
  EXPECT_FALSE(qcdsl::is_parametric(GateKind::T));
  EXPECT_FALSE(qcdsl::is_parametric(GateKind::CX));
}

TEST(Gate, StoresQubitsAndAngle) {
  const Gate g(GateKind::RX, {2}, 1.5707963267948966);
  EXPECT_EQ(g.kind, GateKind::RX);
  EXPECT_EQ(g.width(), 1u);
  EXPECT_EQ(g.qubits[0], 2u);
  EXPECT_DOUBLE_EQ(g.param, 1.5707963267948966);
}

TEST(Gate, RejectsWrongQubitCount) {
  EXPECT_THROW(Gate(GateKind::CX, {0}), std::invalid_argument);
  EXPECT_THROW(Gate(GateKind::H, {0, 1}), std::invalid_argument);
}

TEST(Gate, RejectsSelfControlledTwoQubitGate) {
  EXPECT_THROW(Gate(GateKind::CX, {1, 1}), std::invalid_argument);
  EXPECT_THROW(Gate(GateKind::SWAP, {3, 3}), std::invalid_argument);
}

TEST(Gate, NameRoundTripsForQasm) {
  EXPECT_STREQ(qcdsl::to_string(GateKind::Sdg), "sdg");
  EXPECT_STREQ(qcdsl::to_string(GateKind::CX), "cx");
  EXPECT_STREQ(qcdsl::to_string(GateKind::MEASURE), "measure");
}

}  // namespace