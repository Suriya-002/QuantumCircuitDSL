#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <numeric>
#include <random>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "qcdsl/circuit.hpp"
#include "qcdsl/gate.hpp"
#include "qcdsl/hw/coupling_map.hpp"
#include "qcdsl/ir/dag.hpp"

namespace qcdsl {

/// layout[logical] = physical.
using Layout = std::vector<Qubit>;

struct RoutingResult {
  /// On PHYSICAL qubits, and executable: every two-qubit gate acts on a pair
  /// the device actually couples.
  Circuit circuit;

  /// Where each logical qubit started and where it ended up. The SWAPs move
  /// them, so these differ, and the difference is not cosmetic -- see below.
  Layout initial_layout;
  Layout final_layout;

  std::size_t swaps_added = 0;

  /// A Circuit has no default constructor -- a zero-qubit circuit is not a
  /// thing -- so a RoutingResult must be told the device width up front.
  explicit RoutingResult(std::size_t num_physical) : circuit(num_physical) {}
};

/// Tuning knobs for SABRE.
struct SabreOptions {
  /// W in the paper: how much the next few gates count relative to the ones
  /// that are ready now.
  double lookahead_weight = 0.5;
  /// |E|: how many gates beyond the front layer to look at.
  std::size_t lookahead_size = 20;
  /// Discourages the router from repeatedly moving the same qubit.
  double decay_step = 0.001;
  /// Best-of-N over random TIE-BREAKS, from a fixed starting layout. SABRE is a
  /// heuristic; running it more than once and keeping the best answer is the
  /// cheapest real improvement.
  std::size_t trials = 1;

  /// Best-of-N over random STARTING PERMUTATIONS for the layout search.
  ///
  /// This is not the same knob as `trials`, and confusing the two is what kept
  /// this router 5-27% behind Qiskit's. `trials` re-rolls the coin flips inside
  /// one run; every run still begins from the identity layout, so they all
  /// descend into the SAME basin and disagree only about the details. More
  /// trials therefore polish one local optimum and never look for a better one.
  ///
  /// `layout_trials` restarts the search from a different random permutation
  /// each time, landing in a DIFFERENT basin, and keeps whichever bottoms out
  /// lowest. That is what Qiskit's SabreLayout does (its `layout_trials`
  /// defaults to the CPU count), and it is the entire source of its advantage.
  ///
  /// Trial 0 always starts from the identity, so raising this can only help.
  std::size_t layout_trials = 1;

  /// How many routing passes to spend RANKING a candidate layout. 0 means "use
  /// `trials`".
  ///
  /// Ranking is not answering. To decide whether layout A beats layout B you do
  /// not need A's best-of-eight routing -- you need enough signal to order
  /// them. Measured on a 25-qubit grid: ranking with ONE pass instead of eight
  /// costs 3% in final swap count and returns 35% of the compile time.
  /// Ninety-five percent of a compile is the layout search, so that is where
  /// the time is.
  ///
  /// A cheap ranking on its own would break the promise that raising
  /// `layout_trials` can never make the answer worse -- the layout that ranks
  /// best under one pass need not be the one that routes best under eight. So
  /// `compile` evaluates BOTH the cheap winner and the identity candidate at
  /// the full budget and keeps the better. The promise survives; the cost does
  /// not.
  std::size_t scoring_trials = 0;

  std::uint64_t seed = 0;
};

/// SABRE: SWAP-based BidiREctional heuristic search (Li, Ding & Xie, ASPLOS
/// 2019). The algorithm Qiskit ships.
///
/// The idea is small. Keep a mapping from logical to physical qubits. Look at
/// the FRONT LAYER of the dependency graph -- the gates with nothing left
/// blocking them. Execute every one that is already legal (single-qubit gates
/// always are; a two-qubit gate is legal iff its qubits are coupled). If none
/// are legal, the mapping has to change: score every SWAP on an edge touching a
/// front-layer qubit, apply the best one, and look again.
///
/// The score is what makes it work:
///
///   H = mean distance between the qubits of the front-layer gates
///     + W * mean distance for the next few gates after that   (lookahead)
///     scaled by a decay on recently-swapped qubits            (parallelism)
///
/// The lookahead term stops the router from greedily fixing one gate while
/// wrecking the next. The decay term stops it from moving the same qubit over
/// and over when it could be making progress elsewhere.
///
/// THE THING THAT IS EASY TO GET WRONG: routing PERMUTES the qubits. The routed
/// circuit does not compute the same state vector as the original -- it
/// computes the same state with its amplitude indices permuted by
/// `final_layout`. Compare them naively and a correct router looks broken; fail
/// to compare at all and a broken router looks correct.
/// `RoutingResult::final_layout` is what closes that gap, and the tests use it.
class SabreRouter {
 public:
  /// Nested alias: a nested struct with default member initialisers cannot be
  /// used as a default argument in its own enclosing class, so Options lives at
  /// namespace scope and is aliased back in.
  using Options = SabreOptions;

  explicit SabreRouter(CouplingMap device, SabreOptions opt = SabreOptions())
      : device_(std::move(device)), opt_(opt) {
    if (!device_.is_connected()) {
      throw std::invalid_argument(
          "cannot route on a disconnected device: some qubits can never be "
          "brought together");
    }
    if (opt_.trials == 0) {
      throw std::invalid_argument("trials must be at least 1");
    }
    if (opt_.layout_trials == 0) {
      throw std::invalid_argument("layout_trials must be at least 1");
    }
  }

  [[nodiscard]] const CouplingMap& device() const noexcept { return device_; }

  /// Route with the identity layout: logical q starts on physical q.
  [[nodiscard]] RoutingResult route(const Circuit& qc) const {
    return route(qc, trivial_layout(qc.num_qubits()));
  }

  /// Route `initial` `opt_.trials` times with different tie-breaks and keep the
  /// best. The trials share nothing, so they run in parallel.
  ///
  /// Nested inside `search_layouts`' parallel region this collapses to serial,
  /// which is correct: the outer loop over candidates already has the cores.
  [[nodiscard]] RoutingResult route(const Circuit& qc,
                                    const Layout& initial) const {
    check_layout(qc, initial);
    const std::size_t trials = opt_.trials;

    std::vector<RoutingResult> results(trials,
                                       RoutingResult(device_.num_qubits()));
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
    for (std::ptrdiff_t t = 0; t < static_cast<std::ptrdiff_t>(trials); ++t) {
      results[static_cast<std::size_t>(t)] =
          route_once(qc, initial, opt_.seed + static_cast<std::uint64_t>(t));
    }

    // Ties to the lowest trial index, so the answer does not depend on the
    // order the threads happened to finish in.
    std::size_t best = 0;
    for (std::size_t t = 1; t < trials; ++t) {
      if (results[t].swaps_added < results[best].swaps_added) {
        best = t;
      }
    }
    return std::move(results[best]);
  }

  /// SabreLayout. Route forwards, take the mapping you ended on, route the
  /// REVERSED circuit from it, and take the mapping you end on there. That is
  /// the initial layout for the real run.
  ///
  /// It works because a good final mapping for a circuit is a good initial
  /// mapping for its reverse: the router has already discovered which qubits
  /// want to be near which. Two passes cost nothing and typically remove a
  /// large fraction of the SWAPs an identity layout would need.
  ///
  /// The refinement is a DESCENT: it improves whatever layout you hand it, but
  /// it cannot leave the basin that layout sits in. Starting always from the
  /// identity therefore explores exactly one basin, however many times you run
  /// it. `opt_.layout_trials` restarts from random permutations and keeps the
  /// best -- see SabreOptions, and see bench/layout_ablation.cpp for what that
  /// is worth against Qiskit.
  ///
  /// Returns EVERY candidate plus the index of the one that ranked best. The
  /// caller needs them all: the ranking is cheap (see
  /// SabreOptions::scoring_trials) and therefore not authoritative, so
  /// `compile` plays the winner off against candidate 0 -- the identity descent
  /// -- at the full routing budget before committing.
  [[nodiscard]] std::pair<std::vector<Layout>, std::size_t> search_layouts(
      const Circuit& qc, std::size_t iterations = 3) const {
    const std::size_t n = qc.num_qubits();
    const std::size_t trials = opt_.layout_trials;

    std::vector<Layout> cand(trials);
    std::vector<std::size_t> score(trials);

    // The restarts are independent BY CONSTRUCTION: each has its own starting
    // permutation, its own descent, and its own score, and none of them touch
    // each other. So they run in parallel. Qiskit parallelises its layout
    // trials too, which is a large part of why its transpiler is fast.
    //
    // Each trial's RNG is seeded from its INDEX, not drawn from a shared
    // stream. A shared stream would make the answer depend on the order the
    // threads happened to run in, which is a bug that only shows up on someone
    // else's machine.
    //
    // EVERY candidate is scored with `route`, the exact procedure the caller
    // will use on the winner. An earlier version scored each candidate under
    // its own routing seed and then routed the winner under a different one, so
    // the number used to CHOOSE was not the number you GOT -- and adding
    // restarts could make the result worse. A test caught it.
#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic)
#endif
    for (std::ptrdiff_t t = 0; t < static_cast<std::ptrdiff_t>(trials); ++t) {
      const auto ut = static_cast<std::uint64_t>(t);
      Layout start = trivial_layout(n);
      if (t > 0) {
        // Trial 0 stays the identity: the layout a caller gets for free.
        // Anything returned must be at least as good as it.
        std::mt19937_64 rng(opt_.seed + ut * 7919U);
        std::shuffle(start.begin(), start.end(), rng);
      }
      const std::size_t i = static_cast<std::size_t>(t);
      cand[i] = refine_layout(qc, std::move(start), iterations,
                              opt_.seed + ut * 977U);

      // RANK, do not answer. See SabreOptions::scoring_trials.
      const std::size_t st =
          opt_.scoring_trials == 0 ? opt_.trials : opt_.scoring_trials;
      std::size_t s_best = std::numeric_limits<std::size_t>::max();
      for (std::size_t k = 0; k < st; ++k) {
        s_best = std::min(s_best,
                          route_once(qc, cand[i], opt_.seed + k).swaps_added);
      }
      score[i] = s_best;
    }

    // Ties go to the lowest index, so trial 0 -- the identity -- wins them.
    std::size_t best = 0;
    for (std::size_t t = 1; t < trials; ++t) {
      if (score[t] < score[best]) {
        best = t;
      }
    }
    return {std::move(cand), best};
  }

  /// The forward-reverse descent, from a given starting layout.
  ///
  /// Returns the BEST layout the descent visited, not the last one.
  ///
  /// The distinction is not pedantic. The forward-reverse iteration is NOT
  /// monotone -- it can, and does, wander off a good layout onto a worse one.
  /// Measured on 40 random circuits (bench/layout_ablation.py):
  ///
  ///     grid2x4   iteration 2: 12.7 swaps   iteration 3: 13.4 swaps
  ///     ring7     iteration 2: 16.4 swaps   iteration 3: 16.7 swaps
  ///
  /// Returning the final iterate therefore throws away a layout the algorithm
  /// had already found. Qiskit's SabreLayout has the same shape (run
  /// max_iterations, take the last), so this is not a deviation from the
  /// reference so much as a free correction to it.
  ///
  /// The descent also SATURATES after one or two iterations on every topology
  /// tested, so the default of 3 is already past the point of return.
  /// The forward pass of each iteration ALREADY routes the current layout, so
  /// its swap count is that layout's score. Taking it costs nothing; asking for
  /// it again with a separate `route_once` cost three passes per candidate, and
  /// this loop is 95% of a compile.
  ///
  /// The seed is fixed across iterations rather than advanced. Scores compared
  /// under different seeds are not comparable, and selecting on one number
  /// while evaluating on another is exactly the bug a test caught here once
  /// already.
  [[nodiscard]] Layout refine_layout(const Circuit& qc, Layout layout,
                                     std::size_t iterations,
                                     std::uint64_t seed) const {
    Layout best = layout;
    std::size_t best_swaps = std::numeric_limits<std::size_t>::max();

    for (std::size_t i = 0; i < iterations; ++i) {
      RoutingResult fwd = route_once(qc, layout, seed);
      if (fwd.swaps_added < best_swaps) {
        best_swaps = fwd.swaps_added;
        best = layout;  // the layout that was ROUTED, not the one it produced
      }
      layout = std::move(fwd.final_layout);
      layout = route_once(reverse(qc), layout, seed).final_layout;
    }

    // The layout the last iteration produced has not been scored yet.
    if (route_once(qc, layout, seed).swaps_added < best_swaps) {
      best = std::move(layout);
    }
    return best;
  }

  /// The layout this compiler would use.
  [[nodiscard]] Layout find_layout(const Circuit& qc,
                                   std::size_t iterations = 3) const {
    return compile(qc, iterations).initial_layout;
  }

  /// Layout, then route. This is the whole compiler back end in one call.
  ///
  /// The candidate that RANKED best under a cheap score need not be the one
  /// that ROUTES best under the full budget. So both it and candidate 0 -- the
  /// identity descent, which is exactly what `layout_trials = 1` would have
  /// returned -- are routed properly, and the better wins.
  ///
  /// That play-off is what keeps the promise that raising `layout_trials` can
  /// never make the answer worse. Without it, a cheap ranking silently breaks
  /// that promise, which is the same class of bug as selecting a layout under
  /// one routing seed and then evaluating it under another. A test caught that
  /// one. This is the same trap wearing a different hat.
  [[nodiscard]] RoutingResult compile(const Circuit& qc,
                                      std::size_t iterations = 3) const {
    auto [cand, winner] = search_layouts(qc, iterations);

    RoutingResult best = route(qc, cand[0]);
    if (winner != 0) {
      RoutingResult challenger = route(qc, cand[winner]);
      if (challenger.swaps_added < best.swaps_added) {
        best = std::move(challenger);
      }
    }
    return best;
  }

  /// Every two-qubit gate on a coupled pair?
  [[nodiscard]] bool respects_device(const Circuit& qc) const {
    if (qc.num_qubits() != device_.num_qubits()) {
      return false;
    }
    return std::all_of(qc.gates().begin(), qc.gates().end(),
                       [this](const Gate& g) {
                         return g.qubits.size() != 2 ||
                                device_.are_connected(g.qubits[0], g.qubits[1]);
                       });
  }

  [[nodiscard]] Layout trivial_layout(std::size_t num_logical) const {
    if (num_logical > device_.num_qubits()) {
      throw std::invalid_argument(
          "circuit needs " + std::to_string(num_logical) +
          " qubits but the device has " + std::to_string(device_.num_qubits()));
    }
    Layout l(device_.num_qubits());
    std::iota(l.begin(), l.end(), Qubit{0});
    return l;
  }

 private:
  static Circuit reverse(const Circuit& qc) {
    Circuit out(qc.num_qubits());
    const std::vector<Gate>& gs = qc.gates();
    for (auto it = gs.rbegin(); it != gs.rend(); ++it) {
      out.add(*it);
    }
    return out;
  }

  void check_layout(const Circuit& qc, const Layout& l) const {
    if (qc.num_qubits() > device_.num_qubits()) {
      throw std::invalid_argument(
          "circuit needs " + std::to_string(qc.num_qubits()) +
          " qubits but the device has " + std::to_string(device_.num_qubits()));
    }
    if (l.size() != device_.num_qubits()) {
      throw std::invalid_argument(
          "a layout must name a physical qubit for every device qubit");
    }
    std::vector<bool> seen(device_.num_qubits(), false);
    for (const Qubit p : l) {
      if (p >= device_.num_qubits() || seen[p]) {
        throw std::invalid_argument("a layout must be a permutation");
      }
      seen[p] = true;
    }
  }

  [[nodiscard]] RoutingResult route_once(const Circuit& qc,
                                         const Layout& initial,
                                         std::uint64_t seed) const {
    const std::size_t np = device_.num_qubits();
    const Dag dag(qc);

    std::vector<Qubit> pi = initial;  // logical -> physical
    std::vector<Qubit> inv(np);       // physical -> logical
    for (std::size_t l = 0; l < np; ++l) {
      inv[pi[l]] = static_cast<Qubit>(l);
    }

    std::vector<std::size_t> indeg(dag.size());
    for (std::size_t i = 0; i < dag.size(); ++i) {
      indeg[i] = dag.node(i).preds.size();
    }
    std::vector<std::size_t> front;
    for (std::size_t i = 0; i < dag.size(); ++i) {
      if (indeg[i] == 0) {
        front.push_back(i);
      }
    }

    RoutingResult out(np);
    out.initial_layout = initial;

    std::vector<double> decay(np, 1.0);
    std::mt19937_64 rng(seed);

    // SABRE is a heuristic and a pathological input could in principle make it
    // thrash. Bound the work and fail loudly rather than hang.
    const std::size_t swap_budget = 100 * (qc.size() + 1) * np;
    std::size_t executed = 0;

    while (executed < dag.size()) {
      std::vector<std::size_t> ready;
      for (const std::size_t id : front) {
        if (is_executable(dag.node(id).gate, pi)) {
          ready.push_back(id);
        }
      }

      if (!ready.empty()) {
        for (const std::size_t id : ready) {
          emit(out.circuit, dag.node(id).gate, pi);
          ++executed;
          front.erase(std::find(front.begin(), front.end(), id));
          for (const std::size_t s : dag.node(id).succs) {
            if (--indeg[s] == 0) {
              front.push_back(s);
            }
          }
        }
        std::fill(decay.begin(), decay.end(), 1.0);
        continue;
      }

      if (out.swaps_added >= swap_budget) {
        throw std::runtime_error(
            "sabre failed to make progress; the coupling map may be malformed");
      }

      const auto [a, b] = best_swap(dag, front, indeg, pi, inv, decay, rng);
      out.circuit.add(GateKind::SWAP, {a, b});
      ++out.swaps_added;

      const Qubit la = inv[a];
      const Qubit lb = inv[b];
      std::swap(inv[a], inv[b]);
      pi[la] = b;
      pi[lb] = a;

      decay[a] += opt_.decay_step;
      decay[b] += opt_.decay_step;
    }

    out.final_layout = pi;
    return out;
  }

  [[nodiscard]] bool is_executable(const Gate& g,
                                   const std::vector<Qubit>& pi) const {
    if (g.qubits.size() < 2) {
      return true;  // single-qubit gates never care where they are
    }
    return device_.are_connected(pi[g.qubits[0]], pi[g.qubits[1]]);
  }

  static void emit(Circuit& out, const Gate& g, const std::vector<Qubit>& pi) {
    std::vector<Qubit> phys;
    phys.reserve(g.qubits.size());
    for (const Qubit q : g.qubits) {
      phys.push_back(pi[q]);
    }
    out.add(Gate(g.kind, phys, g.param));
  }

  /// The gates that would execute NEXT, once the front layer clears -- up to
  /// `lookahead_size` two-qubit gates. These are the gates the router should
  /// avoid making harder while it fixes the ones in front of it.
  ///
  /// A gate joins this set only when its LAST remaining predecessor has been
  /// traversed. That is the whole subtlety, and getting it wrong is expensive.
  ///
  /// The obvious implementation -- breadth-first search from the front layer,
  /// marking each gate the first time you reach it -- builds a BFS BALL, not a
  /// topological wavefront. It admits gates that are still waiting on other
  /// predecessors down a different branch of the DAG, and those gates may be
  /// many layers from executing. The lookahead term then spends its budget
  /// optimising qubit distances for gates that will not run for a long time.
  ///
  /// The symptom is diagnostic: with the BFS-ball version, this router's best
  /// `lookahead_size` was 10, and raising it to 20 made routing WORSE -- while
  /// Qiskit runs happily at 20, because its extended set is a true wavefront. A
  /// lookahead window that gets worse as you widen it is not a window, it is a
  /// leak.
  [[nodiscard]] std::vector<std::size_t> extended_set(
      const Dag& dag, const std::vector<std::size_t>& front,
      std::vector<std::size_t> remaining) const {
    std::vector<std::size_t> out;
    std::vector<std::size_t> frontier = front;

    while (!frontier.empty() && out.size() < opt_.lookahead_size) {
      std::vector<std::size_t> next;
      for (const std::size_t id : frontier) {
        for (const std::size_t s : dag.node(id).succs) {
          // Decrement, and admit only on the last predecessor. `remaining` is a
          // copy of the router's live in-degree, so gates already executed are
          // already accounted for and this cannot underflow.
          if (--remaining[s] != 0) {
            continue;
          }
          next.push_back(s);
          if (dag.node(s).gate.qubits.size() == 2) {
            out.push_back(s);
            if (out.size() >= opt_.lookahead_size) {
              return out;
            }
          }
        }
      }
      frontier = std::move(next);
    }
    return out;
  }

  [[nodiscard]] double mean_distance(const Dag& dag,
                                     const std::vector<std::size_t>& gates,
                                     const std::vector<Qubit>& pi) const {
    std::size_t total = 0;
    std::size_t count = 0;
    for (const std::size_t id : gates) {
      const Gate& g = dag.node(id).gate;
      if (g.qubits.size() != 2) {
        continue;
      }
      total += device_.distance(pi[g.qubits[0]], pi[g.qubits[1]]);
      ++count;
    }
    return (count == 0)
               ? 0.0
               : static_cast<double>(total) / static_cast<double>(count);
  }

  [[nodiscard]] std::pair<Qubit, Qubit> best_swap(
      const Dag& dag, const std::vector<std::size_t>& front,
      const std::vector<std::size_t>& indeg, const std::vector<Qubit>& pi,
      const std::vector<Qubit>& inv, const std::vector<double>& decay,
      std::mt19937_64& rng) const {
    // Only SWAPs that touch a qubit the front layer is waiting on can possibly
    // help. Everything else moves an irrelevant qubit.
    std::vector<bool> involved(device_.num_qubits(), false);
    for (const std::size_t id : front) {
      const Gate& g = dag.node(id).gate;
      if (g.qubits.size() == 2) {
        involved[pi[g.qubits[0]]] = true;
        involved[pi[g.qubits[1]]] = true;
      }
    }

    const std::vector<std::size_t> ext = extended_set(dag, front, indeg);

    double best_score = std::numeric_limits<double>::infinity();
    std::vector<std::pair<Qubit, Qubit>> best;

    for (const auto& e : device_.edges()) {
      const Qubit a = e.first;
      const Qubit b = e.second;
      if (!involved[a] && !involved[b]) {
        continue;
      }

      // Try it: swap the two logical qubits sitting on these physical ones.
      std::vector<Qubit> trial = pi;
      const Qubit la = inv[a];
      const Qubit lb = inv[b];
      trial[la] = b;
      trial[lb] = a;

      double score = mean_distance(dag, front, trial);
      if (!ext.empty()) {
        score += opt_.lookahead_weight * mean_distance(dag, ext, trial);
      }
      score *= std::max(decay[a], decay[b]);

      if (score < best_score - 1e-12) {
        best_score = score;
        best.clear();
        best.emplace_back(a, b);
      } else if (score < best_score + 1e-12) {
        best.emplace_back(a, b);
      }
    }

    if (best.empty()) {
      throw std::runtime_error("no candidate swap; the coupling map is broken");
    }
    // Break ties at random. This is what makes `trials` worth anything.
    std::uniform_int_distribution<std::size_t> pick(0, best.size() - 1);
    return best[pick(rng)];
  }

  CouplingMap device_;
  Options opt_;
};

/// Where amplitude index `i` of the UNROUTED state vector ends up in the ROUTED
/// one, given the layout the router finished with.
///
/// Logical qubit l is bit l of the original index. After routing it lives on
/// physical qubit final_layout[l], which is bit final_layout[l] of the routed
/// index. So the permutation just moves bits around -- and getting this wrong
/// is why routing looks broken when it is not.
inline std::size_t permute_index(std::size_t i, const Layout& final_layout) {
  std::size_t j = 0;
  for (std::size_t l = 0; l < final_layout.size(); ++l) {
    if (((i >> l) & 1U) != 0) {
      j |= std::size_t(1) << final_layout[l];
    }
  }
  return j;
}

}  // namespace qcdsl
