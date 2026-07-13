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
  /// Best-of-N over random tie-breaks. SABRE is a heuristic; running it more
  /// than once and keeping the best answer is the cheapest real improvement.
  std::size_t trials = 1;
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
  }

  [[nodiscard]] const CouplingMap& device() const noexcept { return device_; }

  /// Route with the identity layout: logical q starts on physical q.
  [[nodiscard]] RoutingResult route(const Circuit& qc) const {
    return route(qc, trivial_layout(qc.num_qubits()));
  }

  [[nodiscard]] RoutingResult route(const Circuit& qc,
                                    const Layout& initial) const {
    check_layout(qc, initial);

    RoutingResult best = route_once(qc, initial, opt_.seed);
    for (std::size_t trial = 1; trial < opt_.trials; ++trial) {
      RoutingResult r = route_once(qc, initial, opt_.seed + trial);
      if (r.swaps_added < best.swaps_added) {
        best = std::move(r);
      }
    }
    return best;
  }

  /// SabreLayout. Route forwards, take the mapping you ended on, route the
  /// REVERSED circuit from it, and take the mapping you end on there. That is
  /// the initial layout for the real run.
  ///
  /// It works because a good final mapping for a circuit is a good initial
  /// mapping for its reverse: the router has already discovered which qubits
  /// want to be near which. Two passes cost nothing and typically remove a
  /// large fraction of the SWAPs an identity layout would need.
  [[nodiscard]] Layout find_layout(const Circuit& qc,
                                   std::size_t iterations = 3) const {
    Layout layout = trivial_layout(qc.num_qubits());
    for (std::size_t i = 0; i < iterations; ++i) {
      layout = route_once(qc, layout, opt_.seed + i).final_layout;
      layout = route_once(reverse(qc), layout, opt_.seed + i).final_layout;
    }
    return layout;
  }

  /// Layout, then route. This is the whole compiler back end in one call.
  [[nodiscard]] RoutingResult compile(const Circuit& qc) const {
    return route(qc, find_layout(qc));
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

      const auto [a, b] = best_swap(dag, front, pi, inv, decay, rng);
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

  /// The next `lookahead_size` two-qubit gates beyond the front layer, in
  /// breadth-first order. These are the gates the router should avoid making
  /// harder while it fixes the ones in front of it.
  [[nodiscard]] std::vector<std::size_t> extended_set(
      const Dag& dag, const std::vector<std::size_t>& front) const {
    std::vector<std::size_t> out;
    std::vector<bool> seen(dag.size(), false);
    std::vector<std::size_t> frontier = front;
    for (const std::size_t id : front) {
      seen[id] = true;
    }

    while (!frontier.empty() && out.size() < opt_.lookahead_size) {
      std::vector<std::size_t> next;
      for (const std::size_t id : frontier) {
        for (const std::size_t s : dag.node(id).succs) {
          if (seen[s]) {
            continue;
          }
          seen[s] = true;
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
      const std::vector<Qubit>& pi, const std::vector<Qubit>& inv,
      const std::vector<double>& decay, std::mt19937_64& rng) const {
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

    const std::vector<std::size_t> ext = extended_set(dag, front);

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
