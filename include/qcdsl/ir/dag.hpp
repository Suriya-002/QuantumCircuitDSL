#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <queue>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

#include "qcdsl/circuit.hpp"
#include "qcdsl/gate.hpp"

namespace qcdsl {

/// Directed acyclic graph of true data dependencies between gates.
///
/// A `Circuit` records the order the gates were *written*. That order is mostly
/// arbitrary: in `h q0; t q1; cx q0,q1` nothing forces the h before the t --
/// they touch different wires. The DAG throws program order away and keeps only
/// the constraints that actually exist: an edge A -> B iff A and B share a
/// qubit and A is the last gate on that qubit before B.
///
/// This is the representation every pass operates on. Fusing two gates requires
/// proving they are adjacent on a wire; reordering requires proving two gates
/// are independent. Both are edge queries here and neither is answerable from a
/// flat gate list.
///
/// Quantum circuits make this unusually clean: a gate's only inputs and outputs
/// are its wires, so "shares a qubit" is the entire dependency relation. There
/// are no aliases, no memory, no side channels.
class Dag {
 public:
  static constexpr std::size_t kNone = static_cast<std::size_t>(-1);

  struct Node {
    Gate gate;
    std::vector<std::size_t> preds;
    std::vector<std::size_t> succs;
  };

  explicit Dag(const Circuit& qc) : n_(qc.num_qubits()) {
    nodes_.reserve(qc.size());
    std::vector<std::size_t> last(n_, kNone);

    for (const Gate& g : qc.gates()) {
      const std::size_t id = nodes_.size();
      nodes_.push_back(Node{g, {}, {}});

      for (const Qubit q : g.qubits) {
        const std::size_t prev = last[q];
        // A two-qubit gate immediately following another on BOTH of its wires
        // would otherwise record the same edge twice. One dependency, one edge.
        if (prev != kNone && !has_pred(id, prev)) {
          nodes_[id].preds.push_back(prev);
          nodes_[prev].succs.push_back(id);
        }
        last[q] = id;
      }
    }
  }

  [[nodiscard]] std::size_t num_qubits() const noexcept { return n_; }
  [[nodiscard]] std::size_t size() const noexcept { return nodes_.size(); }
  [[nodiscard]] const std::vector<Node>& nodes() const noexcept {
    return nodes_;
  }

  [[nodiscard]] const Node& node(std::size_t id) const {
    if (id >= nodes_.size()) {
      throw std::out_of_range("dag node id " + std::to_string(id) +
                              " out of range");
    }
    return nodes_[id];
  }

  [[nodiscard]] std::size_t num_edges() const {
    std::size_t e = 0;
    for (const Node& v : nodes_) {
      e += v.succs.size();
    }
    return e;
  }

  /// Gates with no unmet dependency -- the ones that could execute right now.
  [[nodiscard]] std::vector<std::size_t> frontier() const {
    std::vector<std::size_t> f;
    for (std::size_t i = 0; i < nodes_.size(); ++i) {
      if (nodes_[i].preds.empty()) {
        f.push_back(i);
      }
    }
    return f;
  }

  /// Kahn's algorithm, breaking ties by smallest id so the result is stable.
  /// Throws if the graph has a cycle, which would mean the builder is broken.
  [[nodiscard]] std::vector<std::size_t> topological_order() const {
    std::vector<std::size_t> indeg = indegrees();
    std::priority_queue<std::size_t, std::vector<std::size_t>, std::greater<>>
        ready;
    for (std::size_t i = 0; i < nodes_.size(); ++i) {
      if (indeg[i] == 0) {
        ready.push(i);
      }
    }

    std::vector<std::size_t> out;
    out.reserve(nodes_.size());
    while (!ready.empty()) {
      const std::size_t v = ready.top();
      ready.pop();
      out.push_back(v);
      for (const std::size_t s : nodes_[v].succs) {
        if (--indeg[s] == 0) {
          ready.push(s);
        }
      }
    }
    if (out.size() != nodes_.size()) {
      throw std::logic_error("dag contains a cycle");
    }
    return out;
  }

  /// A *different* valid topological order, chosen at random. Every one of
  /// these is a legal execution schedule, and every one must produce the same
  /// state -- that is the property the DAG exists to guarantee.
  [[nodiscard]] std::vector<std::size_t> random_topological_order(
      std::uint64_t seed) const {
    std::mt19937_64 rng(seed);
    std::vector<std::size_t> indeg = indegrees();

    std::vector<std::size_t> ready;
    for (std::size_t i = 0; i < nodes_.size(); ++i) {
      if (indeg[i] == 0) {
        ready.push_back(i);
      }
    }

    std::vector<std::size_t> out;
    out.reserve(nodes_.size());
    while (!ready.empty()) {
      std::uniform_int_distribution<std::size_t> pick(0, ready.size() - 1);
      const std::size_t k = pick(rng);
      const std::size_t v = ready[k];
      ready[k] = ready.back();
      ready.pop_back();
      out.push_back(v);

      for (const std::size_t s : nodes_[v].succs) {
        if (--indeg[s] == 0) {
          ready.push_back(s);
        }
      }
    }
    if (out.size() != nodes_.size()) {
      throw std::logic_error("dag contains a cycle");
    }
    return out;
  }

  /// ASAP schedule: layer i holds every gate whose longest path from a source
  /// has length i. Gates in one layer are pairwise independent by construction.
  [[nodiscard]] std::vector<std::vector<std::size_t>> layers() const {
    if (nodes_.empty()) {
      return {};
    }
    std::vector<std::size_t> level(nodes_.size(), 0);
    std::size_t deepest = 0;

    for (const std::size_t v : topological_order()) {
      std::size_t l = 0;
      for (const std::size_t p : nodes_[v].preds) {
        l = std::max(l, level[p] + 1);
      }
      level[v] = l;
      deepest = std::max(deepest, l);
    }

    std::vector<std::vector<std::size_t>> out(deepest + 1);
    for (std::size_t i = 0; i < nodes_.size(); ++i) {
      out[level[i]].push_back(i);
    }
    return out;
  }

  /// Longest path through the graph. Must equal Circuit::depth() -- the wire
  /// critical path and the graph critical path are the same number, computed
  /// two completely different ways. If they ever disagree, the DAG is wrong.
  [[nodiscard]] std::size_t depth() const { return layers().size(); }

  /// True iff `order` lists every node exactly once, with every node after all
  /// of its predecessors.
  [[nodiscard]] bool is_valid_schedule(
      const std::vector<std::size_t>& order) const {
    if (order.size() != nodes_.size()) {
      return false;
    }
    std::vector<std::size_t> position(nodes_.size(), kNone);
    for (std::size_t i = 0; i < order.size(); ++i) {
      if (order[i] >= nodes_.size() || position[order[i]] != kNone) {
        return false;  // out of range, or listed twice
      }
      position[order[i]] = i;
    }
    for (std::size_t v = 0; v < nodes_.size(); ++v) {
      for (const std::size_t p : nodes_[v].preds) {
        if (position[p] > position[v]) {
          return false;
        }
      }
    }
    return true;
  }

  /// Linearise back to a Circuit in the canonical topological order.
  [[nodiscard]] Circuit to_circuit() const {
    return to_circuit(topological_order());
  }

  /// Linearise back to a Circuit in a caller-supplied schedule.
  [[nodiscard]] Circuit to_circuit(
      const std::vector<std::size_t>& order) const {
    if (!is_valid_schedule(order)) {
      throw std::invalid_argument(
          "the given order is not a valid topological schedule of this dag");
    }
    Circuit qc(n_);
    for (const std::size_t v : order) {
      qc.add(nodes_[v].gate);
    }
    return qc;
  }

 private:
  [[nodiscard]] bool has_pred(std::size_t id, std::size_t p) const {
    const std::vector<std::size_t>& ps = nodes_[id].preds;
    return std::find(ps.begin(), ps.end(), p) != ps.end();
  }

  [[nodiscard]] std::vector<std::size_t> indegrees() const {
    std::vector<std::size_t> indeg(nodes_.size());
    for (std::size_t i = 0; i < nodes_.size(); ++i) {
      indeg[i] = nodes_[i].preds.size();
    }
    return indeg;
  }

  std::size_t n_;
  std::vector<Node> nodes_{};
};

}  // namespace qcdsl
