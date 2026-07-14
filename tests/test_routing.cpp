#include <gtest/gtest.h>

#include <algorithm>
#include <cstdint>
#include <random>
#include <stdexcept>
#include <vector>

#include "qcdsl/circuit.hpp"
#include "qcdsl/hw/coupling_map.hpp"
#include "qcdsl/pass/routing.hpp"
#include "qcdsl/sim/statevector.hpp"

namespace {

using qcdsl::Circuit;
using qcdsl::CouplingMap;
using qcdsl::GateKind;
using qcdsl::Layout;
using qcdsl::Qubit;
using qcdsl::RoutingResult;
using qcdsl::SabreRouter;

constexpr double kTol = 1e-10;

// --------------------------------------------------------------------------
// CouplingMap
// --------------------------------------------------------------------------

TEST(CouplingMap, LineHasTheRightDistances) {
  const CouplingMap cm = CouplingMap::line(5);
  EXPECT_EQ(cm.num_qubits(), 5u);
  EXPECT_EQ(cm.num_edges(), 4u);
  EXPECT_TRUE(cm.are_connected(0, 1));
  EXPECT_TRUE(cm.are_connected(1, 0));  // undirected
  EXPECT_FALSE(cm.are_connected(0, 2));
  EXPECT_EQ(cm.distance(0, 0), 0u);
  EXPECT_EQ(cm.distance(0, 4), 4u);
  EXPECT_EQ(cm.distance(4, 0), 4u);
  EXPECT_TRUE(cm.is_connected());
}

TEST(CouplingMap, RingIsShorterThanLine) {
  const CouplingMap cm = CouplingMap::ring(6);
  EXPECT_EQ(cm.num_edges(), 6u);
  EXPECT_EQ(cm.distance(0, 5), 1u) << "the ring closes";
  EXPECT_EQ(cm.distance(0, 3), 3u);
  EXPECT_EQ(CouplingMap::line(6).distance(0, 5), 5u);
}

TEST(CouplingMap, GridUsesManhattanDistance) {
  const CouplingMap cm = CouplingMap::grid(3, 4);  // 12 qubits
  EXPECT_EQ(cm.num_qubits(), 12u);
  EXPECT_EQ(cm.degree(0), 2u);        // corner
  EXPECT_EQ(cm.degree(5), 4u);        // interior
  EXPECT_EQ(cm.distance(0, 11), 5u);  // (0,0) -> (2,3): 2 + 3
}

TEST(CouplingMap, AllToAllHasDistanceOneEverywhere) {
  const CouplingMap cm = CouplingMap::all_to_all(5);
  EXPECT_EQ(cm.num_edges(), 10u);
  for (Qubit a = 0; a < 5; ++a) {
    for (Qubit b = 0; b < 5; ++b) {
      EXPECT_EQ(cm.distance(a, b), (a == b) ? 0u : 1u);
    }
  }
}

TEST(CouplingMap, DetectsDisconnection) {
  const CouplingMap cm(4, {{0, 1}, {2, 3}});
  EXPECT_FALSE(cm.is_connected());
  EXPECT_EQ(cm.distance(0, 2), CouplingMap::kUnreachable);
  EXPECT_THROW(SabreRouter{cm}, std::invalid_argument);
}

TEST(CouplingMap, RejectsBadEdges) {
  EXPECT_THROW(CouplingMap(3, {{0, 0}}), std::invalid_argument);
  EXPECT_THROW(CouplingMap(3, {{0, 9}}), std::out_of_range);
  EXPECT_THROW(CouplingMap(0, {}), std::invalid_argument);
  // a duplicate edge is sloppy input, not an error
  const CouplingMap cm(3, {{0, 1}, {1, 0}, {1, 2}});
  EXPECT_EQ(cm.num_edges(), 2u);
}

// --------------------------------------------------------------------------
// The verification machinery, and proof that it is not vacuous
// --------------------------------------------------------------------------

Circuit random_circuit(std::size_t n, std::size_t m, std::uint64_t seed) {
  std::mt19937_64 rng(seed);
  const std::vector<GateKind> one = {GateKind::H, GateKind::T, GateKind::S,
                                     GateKind::X, GateKind::Z};
  std::uniform_real_distribution<double> ang(-3.0, 3.0);
  Circuit qc(n);
  for (std::size_t i = 0; i < m; ++i) {
    const auto roll = rng() % 100;
    if (n >= 2 && roll < 45) {
      std::uniform_int_distribution<std::size_t> pq(0, n - 1);
      Qubit a = pq(rng);
      Qubit b = pq(rng);
      while (b == a) {
        b = pq(rng);
      }
      qc.add((rng() % 2 == 0) ? GateKind::CX : GateKind::CZ, {a, b});
    } else if (roll < 70) {
      qc.add(GateKind::RZ, {rng() % n}, ang(rng));
    } else {
      qc.add(one[rng() % one.size()], {rng() % n});
    }
  }
  return qc;
}

/// Pad a circuit out to the device width; the extra wires stay idle at |0>.
Circuit widen(const Circuit& qc, std::size_t n) {
  Circuit out(n);
  for (const qcdsl::Gate& g : qc.gates()) {
    out.add(g);
  }
  return out;
}

/// THE CHECK. The routed circuit does NOT produce the original state vector --
/// it produces the original state with its amplitude indices permuted by the
/// layout the router finished on. Compare them naively and a correct router
/// looks broken.
void ExpectRoutedStateMatches(const Circuit& original, const RoutingResult& r) {
  const std::size_t np = r.circuit.num_qubits();
  const qcdsl::Statevector<> want = qcdsl::simulate(widen(original, np));
  const qcdsl::Statevector<> got = qcdsl::simulate(r.circuit);
  ASSERT_EQ(want.dim(), got.dim());

  for (std::size_t i = 0; i < want.dim(); ++i) {
    const std::size_t j = qcdsl::permute_index(i, r.final_layout);
    EXPECT_NEAR(std::abs(want.amplitude(i) - got.amplitude(j)), 0.0, kTol)
        << "amplitude " << i << " -> " << j;
  }
}

// If the permutation is not actually needed, the check above proves nothing.
// Prove it IS needed: compare without permuting and it must fail.
TEST(Routing, TheStateReallyIsPermuted) {
  const SabreRouter router{CouplingMap::line(5)};
  Circuit qc(5);
  qc.add(GateKind::H, {0}).add(GateKind::CX, {0, 4});  // distance 4: must swap
  const RoutingResult r = router.route(qc);
  ASSERT_GT(r.swaps_added, 0u);
  EXPECT_NE(r.final_layout, router.trivial_layout(5))
      << "swaps were added but the layout did not move";

  ExpectRoutedStateMatches(qc, r);  // passes with the permutation

  // and fails without it
  const qcdsl::Statevector<> want = qcdsl::simulate(widen(qc, 5));
  const qcdsl::Statevector<> got = qcdsl::simulate(r.circuit);
  bool identical = true;
  for (std::size_t i = 0; i < want.dim(); ++i) {
    if (std::abs(want.amplitude(i) - got.amplitude(i)) > kTol) {
      identical = false;
      break;
    }
  }
  EXPECT_FALSE(identical)
      << "the unpermuted comparison passed, so it is testing nothing";
}

// --------------------------------------------------------------------------
// Routing
// --------------------------------------------------------------------------

TEST(Routing, AllToAllNeedsNoSwaps) {
  const SabreRouter router{CouplingMap::all_to_all(6)};
  for (std::uint64_t seed = 0; seed < 10; ++seed) {
    const Circuit qc = random_circuit(6, 40, seed);
    const RoutingResult r = router.route(qc);
    EXPECT_EQ(r.swaps_added, 0u);
    EXPECT_EQ(r.final_layout, router.trivial_layout(6));
    ExpectRoutedStateMatches(qc, r);
  }
}

TEST(Routing, AdjacentGatesOnALineNeedNoSwaps) {
  const SabreRouter router{CouplingMap::line(4)};
  Circuit qc(4);
  qc.add(GateKind::CX, {0, 1})
      .add(GateKind::CX, {1, 2})
      .add(GateKind::CX, {2, 3});
  const RoutingResult r = router.route(qc);
  EXPECT_EQ(r.swaps_added, 0u);
  EXPECT_TRUE(router.respects_device(r.circuit));
}

TEST(Routing, OutputAlwaysRespectsTheDevice) {
  for (const CouplingMap& cm :
       {CouplingMap::line(6), CouplingMap::ring(6), CouplingMap::grid(2, 4)}) {
    const SabreRouter router{cm};
    for (std::uint64_t seed = 0; seed < 12; ++seed) {
      const Circuit qc = random_circuit(cm.num_qubits(), 40, seed);
      const RoutingResult r = router.route(qc);
      EXPECT_TRUE(router.respects_device(r.circuit))
          << "a two-qubit gate landed on an uncoupled pair";
      ExpectRoutedStateMatches(qc, r);
    }
  }
}

TEST(Routing, PreservesSemanticsOnEveryTopology) {
  for (const CouplingMap& cm :
       {CouplingMap::line(5), CouplingMap::ring(5), CouplingMap::grid(2, 3)}) {
    const SabreRouter router{cm};
    for (std::uint64_t seed = 0; seed < 10; ++seed) {
      const Circuit qc = random_circuit(cm.num_qubits(), 30, 100 + seed);
      ExpectRoutedStateMatches(qc, router.route(qc));
      ExpectRoutedStateMatches(qc, router.compile(qc));
    }
  }
}

// --------------------------------------------------------------------------
// Does the layout search actually earn its keep?
// --------------------------------------------------------------------------

TEST(Routing, SabreLayoutBeatsTheIdentityLayout) {
  const SabreRouter router{CouplingMap::line(7)};
  std::size_t trivial_swaps = 0;
  std::size_t sabre_swaps = 0;

  for (std::uint64_t seed = 0; seed < 30; ++seed) {
    const Circuit qc = random_circuit(7, 60, 900 + seed);
    trivial_swaps += router.route(qc).swaps_added;
    sabre_swaps += router.compile(qc).swaps_added;
  }
  EXPECT_LT(sabre_swaps, trivial_swaps)
      << "trivial " << trivial_swaps << " vs sabre " << sabre_swaps;
}

TEST(Routing, MoreTrialsNeverMakeItWorse) {
  const Circuit qc = random_circuit(7, 80, 7);
  SabreRouter::Options one;
  one.trials = 1;
  SabreRouter::Options many;
  many.trials = 8;
  const std::size_t s1 =
      SabreRouter(CouplingMap::line(7), one).route(qc).swaps_added;
  const std::size_t s8 =
      SabreRouter(CouplingMap::line(7), many).route(qc).swaps_added;
  EXPECT_LE(s8, s1);
}

TEST(Routing, RandomLayoutRestartsNeverMakeItWorse) {
  // Trial 0 of the layout search starts from the identity, so whatever the
  // random restarts turn up, the answer cannot be worse than searching from the
  // identity alone. If this ever fails, the best-of-N bookkeeping is wrong.
  const CouplingMap device = CouplingMap::grid(3, 3);
  for (std::uint64_t seed = 0; seed < 12; ++seed) {
    const Circuit qc = random_circuit(9, 40, seed);

    SabreRouter::Options one;
    one.layout_trials = 1;
    one.seed = seed;

    SabreRouter::Options many = one;
    many.layout_trials = 8;

    const std::size_t few = SabreRouter(device, one).compile(qc).swaps_added;
    const std::size_t lots = SabreRouter(device, many).compile(qc).swaps_added;
    EXPECT_LE(lots, few) << "restarting the layout search made it worse";
  }
}

TEST(Routing, RandomLayoutRestartsStillProduceALegalCircuit) {
  // A restarted search hands back a layout that is NOT the identity. The routed
  // circuit still has to be a permutation of the logical qubits and still has
  // to respect the coupling map -- a faster answer that does not run is not an
  // answer.
  const CouplingMap device = CouplingMap::grid(3, 3);
  SabreRouter::Options opt;
  opt.layout_trials = 8;
  opt.trials = 4;
  const SabreRouter router(device, opt);

  const Circuit qc = random_circuit(9, 40, 7);
  const auto r = router.compile(qc);

  EXPECT_TRUE(router.respects_device(r.circuit));

  auto sorted = r.initial_layout;
  std::sort(sorted.begin(), sorted.end());
  for (std::size_t i = 0; i < sorted.size(); ++i) {
    EXPECT_EQ(sorted[i], i) << "the initial layout is not a permutation";
  }
}

TEST(Routing, LayoutTrialsAndSwapTrialsAreDifferentKnobs) {
  // The distinction that cost this router 5-27% against Qiskit. `trials`
  // re-rolls tie-breaks from a FIXED start, so every run descends into the same
  // basin. `layout_trials` restarts from a different permutation and can reach
  // a different basin. Raising the first is not a substitute for raising the
  // second, and on a grid the difference is large.
  const CouplingMap device = CouplingMap::grid(3, 3);

  std::size_t tie_break_wins = 0;
  std::size_t restart_wins = 0;
  for (std::uint64_t seed = 0; seed < 12; ++seed) {
    const Circuit qc = random_circuit(9, 40, seed);

    SabreRouter::Options tie_breaks;  // 8x the work, same starting layout
    tie_breaks.trials = 8;
    tie_breaks.layout_trials = 1;
    tie_breaks.seed = seed;

    SabreRouter::Options restarts;  // 8x the work, 8 different starting layouts
    restarts.trials = 1;
    restarts.layout_trials = 8;
    restarts.seed = seed;

    tie_break_wins += SabreRouter(device, tie_breaks).compile(qc).swaps_added;
    restart_wins += SabreRouter(device, restarts).compile(qc).swaps_added;
  }
  EXPECT_LT(restart_wins, tie_break_wins)
      << "on a grid, spending the budget on restarts must beat spending it on "
         "tie-breaks -- see bench/layout_ablation.py";
}

TEST(Routing, CheapRankingStillNeverMakesItWorse) {
  // `scoring_trials` ranks a candidate layout with fewer routing passes than
  // the final route will use. That is a cheap ranking, and a cheap ranking is
  // not authoritative: the layout that looks best under one pass need not be
  // the one that routes best under eight.
  //
  // So the promise -- raising `layout_trials` can never make the answer worse
  // -- is no longer free. `compile` keeps it by routing BOTH the cheap winner
  // and candidate 0 (the identity descent, which is exactly what
  // layout_trials=1 returns) at the full budget, and keeping the better.
  //
  // This test exists because without that play-off the promise breaks silently,
  // and a silent broken promise is how the last version of this code shipped a
  // layout chosen under one seed and evaluated under another.
  const CouplingMap device = CouplingMap::grid(3, 3);
  for (std::uint64_t seed = 0; seed < 12; ++seed) {
    const Circuit qc = random_circuit(9, 40, seed);

    SabreRouter::Options one;
    one.trials = 8;
    one.layout_trials = 1;
    one.scoring_trials = 1;
    one.seed = seed;

    SabreRouter::Options many = one;
    many.layout_trials = 8;

    const std::size_t few = SabreRouter(device, one).compile(qc).swaps_added;
    const std::size_t lots = SabreRouter(device, many).compile(qc).swaps_added;
    EXPECT_LE(lots, few) << "cheap ranking broke the layout_trials guarantee";
  }
}

TEST(Routing, CheapRankingCostsQualityButNotCorrectness) {
  // Ranking with one pass instead of eight is allowed to produce a slightly
  // worse circuit. It is not allowed to produce a wrong one.
  const CouplingMap device = CouplingMap::grid(3, 3);
  SabreRouter::Options opt;
  opt.trials = 8;
  opt.layout_trials = 8;
  opt.scoring_trials = 1;
  opt.seed = 3;
  const SabreRouter router(device, opt);

  const Circuit qc = random_circuit(9, 40, 11);
  const auto r = router.compile(qc);
  EXPECT_TRUE(router.respects_device(r.circuit));
}

TEST(Routing, RejectsCircuitsWiderThanTheDevice) {
  const SabreRouter router{CouplingMap::line(3)};
  EXPECT_THROW((void)router.route(Circuit(5)), std::invalid_argument);
  EXPECT_THROW((void)router.trivial_layout(9), std::invalid_argument);
}

TEST(Routing, RejectsALayoutThatIsNotAPermutation) {
  const SabreRouter router{CouplingMap::line(4)};
  Circuit qc(4);
  qc.add(GateKind::H, {0});
  EXPECT_THROW((void)router.route(qc, Layout{0, 0, 1, 2}),
               std::invalid_argument);
  EXPECT_THROW((void)router.route(qc, Layout{0, 1, 2}), std::invalid_argument);
  EXPECT_THROW((void)router.route(qc, Layout{0, 1, 2, 9}),
               std::invalid_argument);
}

TEST(Routing, ANarrowCircuitCanUseAWideDevice) {
  const SabreRouter router{CouplingMap::grid(2, 4)};  // 8 physical
  const Circuit qc = random_circuit(3, 20, 4);        // 3 logical
  const RoutingResult r = router.compile(qc);
  EXPECT_EQ(r.circuit.num_qubits(), 8u);
  EXPECT_TRUE(router.respects_device(r.circuit));
  ExpectRoutedStateMatches(qc, r);
}

}  // namespace
