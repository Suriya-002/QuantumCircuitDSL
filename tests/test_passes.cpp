#include <gtest/gtest.h>

#include <cstddef>
#include <cstdint>
#include <memory>
#include <random>
#include <vector>

#include "qcdsl/circuit.hpp"
#include "qcdsl/ir/dag.hpp"
#include "qcdsl/pass/pass.hpp"
#include "qcdsl/pass/passes.hpp"
#include "qcdsl/sim/statevector.hpp"

namespace {

using qcdsl::CancelInversePairs;
using qcdsl::Circuit;
using qcdsl::Dag;
using qcdsl::DecomposeToCx;
using qcdsl::Gate;
using qcdsl::GateKind;
using qcdsl::MergeRotations;
using qcdsl::PassManager;
using qcdsl::Qubit;
using qcdsl::RemoveIdentities;

constexpr double kTol = 1e-10;

/// THE CONTRACT. Every pass must leave the state vector untouched -- amplitude
/// for amplitude, global phase included. Not "up to phase": exactly.
void ExpectSameState(const Circuit& before, const Circuit& after) {
  ASSERT_EQ(before.num_qubits(), after.num_qubits());
  const qcdsl::Statevector<> a = qcdsl::simulate(before);
  const qcdsl::Statevector<> b = qcdsl::simulate(after);
  for (std::size_t i = 0; i < a.dim(); ++i) {
    EXPECT_NEAR(std::abs(a.amplitude(i) - b.amplitude(i)), 0.0, kTol)
        << "amplitude " << i;
  }
}

Circuit random_circuit(std::size_t n, std::size_t m, std::uint64_t seed) {
  std::mt19937_64 rng(seed);
  const std::vector<GateKind> one = {GateKind::H, GateKind::T,   GateKind::Tdg,
                                     GateKind::S, GateKind::Sdg, GateKind::X,
                                     GateKind::Y, GateKind::Z,   GateKind::I};
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
    } else if (roll < 55) {
      qc.add(rot[rng() % rot.size()], {rng() % n}, ang(rng));
    } else {
      qc.add(one[rng() % one.size()], {rng() % n});
    }
  }
  return qc;
}

// --------------------------------------------------------------------------
// CancelInversePairs
// --------------------------------------------------------------------------

TEST(CancelInversePairs, CancelsAnAdjacentPair) {
  Circuit qc(1);
  qc.add(GateKind::H, {0}).add(GateKind::H, {0});
  const Circuit out = CancelInversePairs()(qc);
  EXPECT_EQ(out.size(), 0u);
  ExpectSameState(qc, out);
}

TEST(CancelInversePairs, CancelsDaggerPairs) {
  for (const auto& p :
       std::vector<std::pair<GateKind, GateKind>>{{GateKind::T, GateKind::Tdg},
                                                  {GateKind::S, GateKind::Sdg},
                                                  {GateKind::Sdg, GateKind::S},
                                                  {GateKind::X, GateKind::X},
                                                  {GateKind::Y, GateKind::Y},
                                                  {GateKind::Z, GateKind::Z}}) {
    Circuit qc(1);
    qc.add(p.first, {0}).add(p.second, {0});
    const Circuit out = CancelInversePairs()(qc);
    EXPECT_EQ(out.size(), 0u) << qcdsl::to_string(p.first);
    ExpectSameState(qc, out);
  }
}

// THE CASE THAT MAKES THIS A DAG PASS AND NOT A STRING MATCH.
// h q0; x q0; h q0 -- the two H gates are inverses of each other and they are
// NOT cancellable, because the X is between them. A pass that scanned the gate
// list for "H followed by H on the same wire" would get this wrong.
TEST(CancelInversePairs, RefusesToCancelAcrossAnInterveningGate) {
  Circuit qc(1);
  qc.add(GateKind::H, {0}).add(GateKind::X, {0}).add(GateKind::H, {0});
  const Circuit out = CancelInversePairs()(qc);
  EXPECT_EQ(out.size(), 3u);
  ExpectSameState(qc, out);
}

// Same idea on two wires: the CX gates are separated on wire 1 only.
TEST(CancelInversePairs, RefusesToCancelWhenOnlyOneWireIsClear) {
  Circuit qc(2);
  qc.add(GateKind::CX, {0, 1}).add(GateKind::T, {1}).add(GateKind::CX, {0, 1});
  const Circuit out = CancelInversePairs()(qc);
  EXPECT_EQ(out.size(), 3u);
  ExpectSameState(qc, out);
}

TEST(CancelInversePairs, CancelsAdjacentTwoQubitGates) {
  Circuit qc(2);
  qc.add(GateKind::CX, {0, 1}).add(GateKind::CX, {0, 1});
  EXPECT_EQ(CancelInversePairs()(qc).size(), 0u);

  Circuit qc2(2);
  qc2.add(GateKind::SWAP, {0, 1}).add(GateKind::SWAP, {0, 1});
  EXPECT_EQ(CancelInversePairs()(qc2).size(), 0u);
}

// CZ is symmetric in its wires, so cz(0,1) . cz(1,0) is a cancelling pair.
// CX is NOT symmetric: cx(0,1) . cx(1,0) is not the identity and must survive.
TEST(CancelInversePairs, RespectsWireSymmetry) {
  Circuit cz(2);
  cz.add(GateKind::CZ, {0, 1}).add(GateKind::CZ, {1, 0});
  const Circuit cz_out = CancelInversePairs()(cz);
  EXPECT_EQ(cz_out.size(), 0u);
  ExpectSameState(cz, cz_out);

  Circuit cx(2);
  cx.add(GateKind::CX, {0, 1}).add(GateKind::CX, {1, 0});
  const Circuit cx_out = CancelInversePairs()(cx);
  EXPECT_EQ(cx_out.size(), 2u) << "cx(0,1) . cx(1,0) is not the identity";
  ExpectSameState(cx, cx_out);
}

TEST(CancelInversePairs, DoesNotTouchRotations) {
  Circuit qc(1);
  qc.add(GateKind::RZ, {0}, 0.5).add(GateKind::RZ, {0}, -0.5);
  EXPECT_EQ(CancelInversePairs()(qc).size(), 2u);
}

// --------------------------------------------------------------------------
// MergeRotations
// --------------------------------------------------------------------------

TEST(MergeRotations, FoldsARunOfSameAxisRotations) {
  Circuit qc(1);
  qc.add(GateKind::RZ, {0}, 0.3)
      .add(GateKind::RZ, {0}, 0.4)
      .add(GateKind::RZ, {0}, -0.2);
  const Circuit out = MergeRotations()(qc);
  ASSERT_EQ(out.size(), 1u);
  EXPECT_EQ(out.gates()[0].kind, GateKind::RZ);
  EXPECT_NEAR(out.gates()[0].param, 0.5, 1e-15);
  ExpectSameState(qc, out);
}

TEST(MergeRotations, DoesNotFoldDifferentAxes) {
  Circuit qc(1);
  qc.add(GateKind::RZ, {0}, 0.3).add(GateKind::RX, {0}, 0.4);
  EXPECT_EQ(MergeRotations()(qc).size(), 2u);
}

TEST(MergeRotations, DoesNotFoldAcrossAnInterveningGate) {
  Circuit qc(1);
  qc.add(GateKind::RZ, {0}, 0.3)
      .add(GateKind::H, {0})
      .add(GateKind::RZ, {0}, 0.4);
  EXPECT_EQ(MergeRotations()(qc).size(), 3u);
}

// RZ(2*pi) is -I, not I. Reducing the angle modulo 2*pi would silently flip the
// sign of the whole state, which is exactly the kind of "harmless" phase bug
// that survives a test suite comparing probabilities instead of amplitudes.
TEST(MergeRotations, DoesNotReduceAnglesModuloTwoPi) {
  const double pi = 3.14159265358979323846;
  Circuit qc(1);
  qc.add(GateKind::H, {0})
      .add(GateKind::RZ, {0}, pi)
      .add(GateKind::RZ, {0}, pi);
  const Circuit out = MergeRotations()(qc);
  ASSERT_EQ(out.size(), 2u);
  EXPECT_NEAR(out.gates()[1].param, 2 * pi, 1e-15);
  ExpectSameState(qc, out);
}

// --------------------------------------------------------------------------
// RemoveIdentities
// --------------------------------------------------------------------------

TEST(RemoveIdentities, DropsIGatesAndZeroRotations) {
  Circuit qc(1);
  qc.add(GateKind::I, {0})
      .add(GateKind::H, {0})
      .add(GateKind::RZ, {0}, 0.0)
      .add(GateKind::RX, {0}, 1e-18);
  const Circuit out = RemoveIdentities()(qc);
  ASSERT_EQ(out.size(), 1u);
  EXPECT_EQ(out.gates()[0].kind, GateKind::H);
  ExpectSameState(qc, out);
}

TEST(RemoveIdentities, KeepsATwoPiRotation) {
  Circuit qc(1);
  qc.add(GateKind::RZ, {0}, 6.283185307179586);
  EXPECT_EQ(RemoveIdentities()(qc).size(), 1u);
}

// --------------------------------------------------------------------------
// DecomposeToCx
// --------------------------------------------------------------------------

TEST(DecomposeToCx, RewritesCzAndSwapAndNothingElse) {
  Circuit qc(2);
  qc.add(GateKind::H, {0})
      .add(GateKind::CZ, {0, 1})
      .add(GateKind::SWAP, {0, 1});
  const Circuit out = DecomposeToCx()(qc);
  // h + (h cx h) + (cx cx cx) = 7
  EXPECT_EQ(out.size(), 7u);
  for (const Gate& g : out.gates()) {
    EXPECT_NE(g.kind, GateKind::CZ);
    EXPECT_NE(g.kind, GateKind::SWAP);
  }
  ExpectSameState(qc, out);
}

TEST(DecomposeToCx, LeavesNoNonCxTwoQubitGateOnRandomCircuits) {
  for (std::uint64_t seed = 0; seed < 25; ++seed) {
    const Circuit qc = random_circuit(4, 40, seed);
    const Circuit out = DecomposeToCx()(qc);
    for (const Gate& g : out.gates()) {
      EXPECT_NE(g.kind, GateKind::CZ);
      EXPECT_NE(g.kind, GateKind::SWAP);
    }
    ExpectSameState(qc, out);
  }
}

TEST(DecomposeToCx, IsIdempotent) {
  const Circuit qc = random_circuit(4, 40, 7);
  const Circuit once = DecomposeToCx()(qc);
  const Circuit twice = DecomposeToCx()(once);
  EXPECT_EQ(once.size(), twice.size());
}

// --------------------------------------------------------------------------
// Every pass, on every random circuit, must preserve the state. This is the
// test that matters; everything above is a specific case of it.
// --------------------------------------------------------------------------

TEST(Pass, EveryPassPreservesTheStateOnRandomCircuits) {
  for (std::uint64_t seed = 0; seed < 40; ++seed) {
    const Circuit qc = random_circuit(1 + (seed % 5), 50, seed);
    ExpectSameState(qc, CancelInversePairs()(qc));
    ExpectSameState(qc, MergeRotations()(qc));
    ExpectSameState(qc, RemoveIdentities()(qc));
    ExpectSameState(qc, DecomposeToCx()(qc));
  }
}

// --------------------------------------------------------------------------
// PassManager
// --------------------------------------------------------------------------

PassManager standard_pipeline() {
  PassManager pm;
  pm.add(std::make_shared<DecomposeToCx>())
      .add(std::make_shared<MergeRotations>())
      .add(std::make_shared<RemoveIdentities>())
      .add(std::make_shared<CancelInversePairs>());
  return pm;
}

TEST(PassManager, RecordsStatsForEveryPass) {
  PassManager pm = standard_pipeline();
  const Circuit qc = random_circuit(4, 60, 11);
  const Circuit out = pm.run(qc);
  ASSERT_EQ(pm.stats().size(), 4u);
  EXPECT_EQ(pm.stats()[0].pass, "decompose-to-cx");
  EXPECT_EQ(pm.stats()[0].gates_before, qc.size());
  EXPECT_EQ(pm.stats().back().gates_after, out.size());
  ExpectSameState(qc, out);
}

TEST(PassManager, PipelinePreservesTheStateOnRandomCircuits) {
  for (std::uint64_t seed = 0; seed < 40; ++seed) {
    PassManager pm = standard_pipeline();
    const Circuit qc = random_circuit(1 + (seed % 5), 60, 500 + seed);
    ExpectSameState(qc, pm.run_to_fixed_point(qc));
  }
}

// Passes enable one another: cancelling H . H can bring two rotations together,
// merging those can produce a zero angle, and removing that identity can expose
// a fresh inverse pair. One sweep is not enough.
TEST(PassManager, FixedPointFindsWhatOneSweepMisses) {
  Circuit qc(1);
  qc.add(GateKind::RZ, {0}, 0.5)
      .add(GateKind::H, {0})
      .add(GateKind::H, {0})
      .add(GateKind::RZ, {0}, -0.5)
      .add(GateKind::T, {0})
      .add(GateKind::Tdg, {0});

  PassManager one = standard_pipeline();
  PassManager fix = standard_pipeline();
  const Circuit after_one_sweep = one.run(qc);
  const Circuit after_fixed_point = fix.run_to_fixed_point(qc);

  EXPECT_LT(after_fixed_point.size(), after_one_sweep.size());
  EXPECT_EQ(after_fixed_point.size(), 0u);
  EXPECT_GT(fix.sweeps(), 1u);
  ExpectSameState(qc, after_one_sweep);
  ExpectSameState(qc, after_fixed_point);
}

TEST(PassManager, EmptyPipelineIsTheIdentity) {
  PassManager pm;
  EXPECT_EQ(pm.size(), 0u);
  const Circuit qc = random_circuit(3, 20, 3);
  EXPECT_EQ(pm.run(qc).size(), qc.size());
}

TEST(PassStats, ReportsWhetherAnythingChanged) {
  PassManager pm;
  pm.add(std::make_shared<CancelInversePairs>());
  Circuit qc(1);
  qc.add(GateKind::H, {0}).add(GateKind::H, {0});
  pm.run(qc);
  ASSERT_EQ(pm.stats().size(), 1u);
  EXPECT_TRUE(pm.stats()[0].changed());
  EXPECT_EQ(pm.stats()[0].gates_before, 2u);
  EXPECT_EQ(pm.stats()[0].gates_after, 0u);
}

}  // namespace
