#include <gtest/gtest.h>

#include <algorithm>
#include <cstddef>
#include <random>
#include <stdexcept>
#include <vector>

#include "qcdsl/circuit.hpp"
#include "qcdsl/ir/dag.hpp"
#include "qcdsl/sim/statevector.hpp"

namespace {

using qcdsl::Circuit;
using qcdsl::Dag;
using qcdsl::GateKind;
using qcdsl::Qubit;

constexpr double kTol = 1e-12;

Circuit random_circuit(std::size_t n, std::size_t m, std::uint64_t seed) {
  std::mt19937_64 rng(seed);
  const std::vector<GateKind> one = {GateKind::H, GateKind::T, GateKind::S,
                                     GateKind::X, GateKind::Z, GateKind::Y};
  const std::vector<GateKind> two = {GateKind::CX, GateKind::CZ,
                                     GateKind::SWAP};
  Circuit qc(n);
  for (std::size_t i = 0; i < m; ++i) {
    if (n >= 2 && (rng() % 100) < 40) {
      std::uniform_int_distribution<std::size_t> pq(0, n - 1);
      Qubit a = pq(rng);
      Qubit b = pq(rng);
      while (b == a) {
        b = pq(rng);
      }
      qc.add(two[rng() % two.size()], {a, b});
    } else {
      qc.add(one[rng() % one.size()], {rng() % n});
    }
  }
  return qc;
}

void ExpectSameState(const Circuit& a, const Circuit& b) {
  const qcdsl::Statevector<> sa = qcdsl::simulate(a);
  const qcdsl::Statevector<> sb = qcdsl::simulate(b);
  ASSERT_EQ(sa.dim(), sb.dim());
  for (std::size_t i = 0; i < sa.dim(); ++i) {
    EXPECT_NEAR(std::abs(sa.amplitude(i) - sb.amplitude(i)), 0.0, kTol)
        << "amplitude " << i;
  }
}

// --------------------------------------------------------------------------
// Structure
// --------------------------------------------------------------------------

TEST(Dag, EmptyCircuitGivesAnEmptyGraph) {
  const Dag d{Circuit(3)};
  EXPECT_EQ(d.num_qubits(), 3u);
  EXPECT_EQ(d.size(), 0u);
  EXPECT_EQ(d.num_edges(), 0u);
  EXPECT_EQ(d.depth(), 0u);
  EXPECT_TRUE(d.layers().empty());
  EXPECT_TRUE(d.frontier().empty());
}

// Gates on disjoint wires are NOT ordered by the DAG, however they were
// written.
TEST(Dag, IndependentGatesHaveNoEdge) {
  Circuit qc(2);
  qc.add(GateKind::H, {0}).add(GateKind::T, {1});
  const Dag d(qc);
  EXPECT_EQ(d.size(), 2u);
  EXPECT_EQ(d.num_edges(), 0u);
  EXPECT_EQ(d.depth(), 1u);
  EXPECT_EQ(d.frontier().size(), 2u);
}

TEST(Dag, GatesSharingAWireAreOrdered) {
  Circuit qc(1);
  qc.add(GateKind::H, {0}).add(GateKind::T, {0});
  const Dag d(qc);
  EXPECT_EQ(d.num_edges(), 1u);
  EXPECT_TRUE(d.node(0).preds.empty());
  ASSERT_EQ(d.node(1).preds.size(), 1u);
  EXPECT_EQ(d.node(1).preds[0], 0u);
  EXPECT_EQ(d.frontier(), std::vector<std::size_t>{0});
}

TEST(Dag, ATwoQubitGateJoinsBothWires) {
  Circuit qc(2);
  qc.add(GateKind::H, {0}).add(GateKind::T, {1}).add(GateKind::CX, {0, 1});
  const Dag d(qc);
  EXPECT_EQ(d.num_edges(), 2u);  // 0 -> 2 and 1 -> 2
  EXPECT_EQ(d.node(2).preds.size(), 2u);
  EXPECT_EQ(d.depth(), 2u);
}

// Back-to-back two-qubit gates on the SAME pair share one dependency, not two.
// A naive builder records the edge once per wire and double-counts it.
TEST(Dag, ARepeatedDependencyIsRecordedOnce) {
  Circuit qc(2);
  qc.add(GateKind::CX, {0, 1}).add(GateKind::CZ, {0, 1});
  const Dag d(qc);
  EXPECT_EQ(d.num_edges(), 1u);
  ASSERT_EQ(d.node(1).preds.size(), 1u);
  EXPECT_EQ(d.node(1).preds[0], 0u);
}

TEST(Dag, RejectsAnOutOfRangeNodeId) {
  const Dag d{Circuit(1)};
  EXPECT_THROW((void)d.node(0), std::out_of_range);
}

// --------------------------------------------------------------------------
// Scheduling
// --------------------------------------------------------------------------

TEST(Dag, NoTwoGatesInALayerShareAWire) {
  for (std::uint64_t seed = 0; seed < 20; ++seed) {
    const Circuit qc = random_circuit(5, 40, seed);
    const Dag d(qc);
    for (const std::vector<std::size_t>& layer : d.layers()) {
      std::vector<bool> used(qc.num_qubits(), false);
      for (const std::size_t id : layer) {
        for (const Qubit q : d.node(id).gate.qubits) {
          EXPECT_FALSE(used[q]) << "wire " << q << " used twice in one layer";
          used[q] = true;
        }
      }
    }
  }
}

TEST(Dag, TopologicalOrderIsAValidSchedule) {
  for (std::uint64_t seed = 0; seed < 20; ++seed) {
    const Dag d(random_circuit(4, 40, seed));
    EXPECT_TRUE(d.is_valid_schedule(d.topological_order()));
    EXPECT_TRUE(d.is_valid_schedule(d.random_topological_order(seed)));
  }
}

TEST(Dag, RejectsSchedulesThatAreNotTopological) {
  Circuit qc(1);
  qc.add(GateKind::H, {0}).add(GateKind::T, {0});
  const Dag d(qc);
  EXPECT_FALSE(d.is_valid_schedule({1, 0}));  // violates 0 -> 1
  EXPECT_FALSE(d.is_valid_schedule({0}));     // too short
  EXPECT_FALSE(d.is_valid_schedule({0, 0}));  // duplicate
  EXPECT_FALSE(d.is_valid_schedule({0, 7}));  // out of range
  EXPECT_THROW((void)d.to_circuit({1, 0}), std::invalid_argument);
}

// THE FIRST OF THE TWO TESTS THAT PIN THE EDGE SET.
//
// Circuit::depth walks the wires and tracks a per-qubit layer counter.
// Dag::depth is the longest path through a graph. Two entirely different
// algorithms over two entirely different structures. If the DAG carries a
// SPURIOUS edge -- an ordering constraint that does not really exist -- its
// critical path gets longer and these two numbers diverge. This is what catches
// an over-constrained IR.
TEST(Dag, DepthAgreesWithTheCircuitCriticalPath) {
  for (std::uint64_t seed = 0; seed < 50; ++seed) {
    const Circuit qc = random_circuit(1 + (seed % 6), 60, seed);
    const Dag d(qc);
    EXPECT_EQ(d.depth(), qc.depth()) << "seed " << seed;
  }
}

// --------------------------------------------------------------------------
// Semantics
// --------------------------------------------------------------------------

TEST(Dag, RoundTripsBackToAnEquivalentCircuit) {
  for (std::uint64_t seed = 0; seed < 20; ++seed) {
    const Circuit qc = random_circuit(4, 40, seed);
    const Circuit back = Dag(qc).to_circuit();
    EXPECT_EQ(back.size(), qc.size());
    EXPECT_EQ(back.depth(), qc.depth());
    ExpectSameState(qc, back);
  }
}

// THE SECOND, AND THE POINT OF THE WHOLE IR.
//
// Every topological order of the DAG is a legal execution schedule, so every
// one must produce the identical state vector. If the DAG is MISSING an edge --
// two gates that really do depend on each other but were left unordered -- then
// some random schedule will run them in the wrong order and the state will
// change.
//
// Together with the depth test above this pins the edge set exactly: too many
// edges and depth diverges, too few and the state diverges. There is exactly
// one edge set that survives both.
TEST(Dag, EveryTopologicalOrderProducesTheSameState) {
  for (std::uint64_t seed = 0; seed < 15; ++seed) {
    const Circuit qc = random_circuit(4, 35, seed);
    const Dag d(qc);
    const qcdsl::Statevector<> reference = qcdsl::simulate(qc);

    for (std::uint64_t shuffle = 0; shuffle < 8; ++shuffle) {
      const std::vector<std::size_t> order =
          d.random_topological_order(1000 * seed + shuffle);
      const Circuit rescheduled = d.to_circuit(order);
      const qcdsl::Statevector<> got = qcdsl::simulate(rescheduled);
      for (std::size_t i = 0; i < reference.dim(); ++i) {
        EXPECT_NEAR(std::abs(got.amplitude(i) - reference.amplitude(i)), 0.0,
                    kTol)
            << "seed " << seed << " shuffle " << shuffle << " amplitude " << i;
      }
    }
  }
}

// The reschedules really are different orderings -- otherwise the test above
// would be vacuous.
TEST(Dag, RandomSchedulesActuallyDiffer) {
  const Dag d(random_circuit(5, 60, 42));
  const std::vector<std::size_t> canonical = d.topological_order();
  std::size_t distinct = 0;
  for (std::uint64_t s = 0; s < 20; ++s) {
    if (d.random_topological_order(s) != canonical) {
      ++distinct;
    }
  }
  EXPECT_GT(distinct, 10u) << "random_topological_order is not exploring";
}

}  // namespace
