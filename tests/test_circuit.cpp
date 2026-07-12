#include <gtest/gtest.h>

#include <stdexcept>

#include "qcdsl/circuit.hpp"

namespace {

using qcdsl::Circuit;
using qcdsl::Gate;
using qcdsl::GateKind;

TEST(Circuit, RejectsZeroQubits) {
  EXPECT_THROW(Circuit(0), std::invalid_argument);
}

TEST(Circuit, RejectsOutOfRangeQubit) {
  Circuit qc(2);
  EXPECT_THROW(qc.add(GateKind::X, {2}), std::out_of_range);
}

TEST(Circuit, EmptyCircuitHasZeroDepth) {
  const Circuit qc(3);
  EXPECT_EQ(qc.num_qubits(), 3u);
  EXPECT_EQ(qc.size(), 0u);
  EXPECT_EQ(qc.depth(), 0u);
}

// Bell pair: h q[0]; cx q[0],q[1];  -- strictly sequential.
TEST(Circuit, BellPairHasSizeTwoDepthTwo) {
  Circuit qc(2);
  qc.add(GateKind::H, {0}).add(GateKind::CX, {0, 1});
  EXPECT_EQ(qc.size(), 2u);
  EXPECT_EQ(qc.depth(), 2u);
}

// Disjoint wires share a layer: 3 gates, depth 1. This is the property that
// makes depth different from size, and the one people get wrong.
TEST(Circuit, DisjointGatesShareOneLayer) {
  Circuit qc(3);
  qc.add(GateKind::H, {0}).add(GateKind::H, {1}).add(GateKind::H, {2});
  EXPECT_EQ(qc.size(), 3u);
  EXPECT_EQ(qc.depth(), 1u);
}

// A two-qubit gate synchronises both wires: q0 is idle, q1 has two gates, so
// the cx cannot be placed before layer 3.
//   q0: -----------@--
//   q1: --X----X---X--
TEST(Circuit, TwoQubitGateSynchronisesBothWires) {
  Circuit qc(2);
  qc.add(GateKind::X, {1}).add(GateKind::X, {1}).add(GateKind::CX, {0, 1});
  EXPECT_EQ(qc.size(), 3u);
  EXPECT_EQ(qc.depth(), 3u);
}

TEST(Circuit, ExposesItsGateList) {
  Circuit qc(2);
  qc.add(GateKind::H, {0}).add(GateKind::CX, {0, 1});
  const auto& gs = qc.gates();
  ASSERT_EQ(gs.size(), 2u);
  EXPECT_EQ(gs[0].kind, GateKind::H);
  EXPECT_EQ(gs[1].kind, GateKind::CX);
}

TEST(Circuit, AcceptsPrebuiltGateObjects) {
  Circuit qc(1);
  qc.add(Gate(GateKind::RZ, {0}, 0.75));
  ASSERT_EQ(qc.size(), 1u);
  EXPECT_DOUBLE_EQ(qc.gates()[0].param, 0.75);
}

TEST(Circuit, CountsGatesPerWire) {
  Circuit qc(2);
  qc.add(GateKind::H, {0}).add(GateKind::CX, {0, 1}).add(GateKind::T, {1});
  EXPECT_EQ(qc.gate_count_on(0), 2u);
  EXPECT_EQ(qc.gate_count_on(1), 2u);
  EXPECT_THROW((void)qc.gate_count_on(5), std::out_of_range);
}

}  // namespace