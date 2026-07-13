#pragma once

#include <algorithm>
#include <cstddef>
#include <queue>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "qcdsl/gate.hpp"

namespace qcdsl {

/// The connectivity of a physical device: which pairs of qubits can actually
/// host a two-qubit gate.
///
/// Everything above this line -- the IR, the passes, the simulator -- assumes a
/// machine where any qubit can talk to any other. No such machine exists. On
/// real hardware `cx q[0], q[17]` is not an instruction; it is a request that
/// the compiler must satisfy by physically walking those two qubits next to
/// each other with SWAPs. This class is the constraint that makes that
/// necessary, and the all-pairs distance matrix is what makes it tractable.
class CouplingMap {
 public:
  static constexpr std::size_t kUnreachable = static_cast<std::size_t>(-1);

  CouplingMap(std::size_t num_qubits,
              const std::vector<std::pair<Qubit, Qubit>>& edges)
      : n_(num_qubits), adj_(num_qubits) {
    if (num_qubits == 0) {
      throw std::invalid_argument("a device must have at least one qubit");
    }
    for (const auto& e : edges) {
      add_edge(e.first, e.second);
    }
    build_distances();
  }

  /// 0 - 1 - 2 - ... - (n-1). The worst realistic topology, and the one that
  /// hurts most: routing cost grows linearly with the distance between qubits.
  static CouplingMap line(std::size_t n) {
    std::vector<std::pair<Qubit, Qubit>> e;
    for (std::size_t i = 0; i + 1 < n; ++i) {
      e.emplace_back(i, i + 1);
    }
    return {n, e};
  }

  static CouplingMap ring(std::size_t n) {
    std::vector<std::pair<Qubit, Qubit>> e;
    for (std::size_t i = 0; i + 1 < n; ++i) {
      e.emplace_back(i, i + 1);
    }
    if (n > 2) {
      e.emplace_back(n - 1, 0);
    }
    return {n, e};
  }

  /// Nearest-neighbour 2-D lattice, row-major. Google's Sycamore and most
  /// neutral-atom arrays are grid-like.
  static CouplingMap grid(std::size_t rows, std::size_t cols) {
    std::vector<std::pair<Qubit, Qubit>> e;
    const auto id = [cols](std::size_t r, std::size_t c) {
      return r * cols + c;
    };
    for (std::size_t r = 0; r < rows; ++r) {
      for (std::size_t c = 0; c < cols; ++c) {
        if (c + 1 < cols) {
          e.emplace_back(id(r, c), id(r, c + 1));
        }
        if (r + 1 < rows) {
          e.emplace_back(id(r, c), id(r + 1, c));
        }
      }
    }
    return {rows * cols, e};
  }

  /// The fiction the rest of the library assumes. Useful as a control: routing
  /// against this must insert exactly zero SWAPs.
  static CouplingMap all_to_all(std::size_t n) {
    std::vector<std::pair<Qubit, Qubit>> e;
    for (std::size_t i = 0; i < n; ++i) {
      for (std::size_t j = i + 1; j < n; ++j) {
        e.emplace_back(i, j);
      }
    }
    return {n, e};
  }

  [[nodiscard]] std::size_t num_qubits() const noexcept { return n_; }
  [[nodiscard]] std::size_t num_edges() const noexcept { return edges_.size(); }
  [[nodiscard]] const std::vector<std::pair<Qubit, Qubit>>& edges()
      const noexcept {
    return edges_;
  }

  [[nodiscard]] const std::vector<Qubit>& neighbours(Qubit q) const {
    check(q);
    return adj_[q];
  }

  [[nodiscard]] std::size_t degree(Qubit q) const {
    return neighbours(q).size();
  }

  [[nodiscard]] bool are_connected(Qubit a, Qubit b) const {
    check(a);
    check(b);
    return std::find(adj_[a].begin(), adj_[a].end(), b) != adj_[a].end();
  }

  /// Shortest path in edges, or kUnreachable. This is the number the router's
  /// heuristic is built on: how far apart two qubits are is exactly how many
  /// SWAPs it costs to bring them together.
  [[nodiscard]] std::size_t distance(Qubit a, Qubit b) const {
    check(a);
    check(b);
    return dist_[a * n_ + b];
  }

  /// A disconnected device cannot route every circuit, and the router must say
  /// so rather than loop forever looking for a path that is not there.
  [[nodiscard]] bool is_connected() const {
    for (std::size_t i = 0; i < n_; ++i) {
      if (dist_[i] == kUnreachable) {
        return false;
      }
    }
    return true;
  }

 private:
  void check(Qubit q) const {
    if (q >= n_) {
      throw std::out_of_range("physical qubit " + std::to_string(q) +
                              " out of range for a " + std::to_string(n_) +
                              "-qubit device");
    }
  }

  void add_edge(Qubit a, Qubit b) {
    check(a);
    check(b);
    if (a == b) {
      throw std::invalid_argument("a qubit cannot be coupled to itself");
    }
    if (are_connected(a, b)) {
      return;  // idempotent; a device listing an edge twice is not an error
    }
    adj_[a].push_back(b);
    adj_[b].push_back(a);
    edges_.emplace_back(std::min(a, b), std::max(a, b));
  }

  /// BFS from every node. O(n * (n + e)) -- trivial at device scale, and it
  /// buys the router a constant-time distance query in its innermost loop.
  void build_distances() {
    dist_.assign(n_ * n_, kUnreachable);
    for (std::size_t src = 0; src < n_; ++src) {
      std::queue<Qubit> q;
      dist_[src * n_ + src] = 0;
      q.push(src);
      while (!q.empty()) {
        const Qubit v = q.front();
        q.pop();
        for (const Qubit w : adj_[v]) {
          if (dist_[src * n_ + w] == kUnreachable) {
            dist_[src * n_ + w] = dist_[src * n_ + v] + 1;
            q.push(w);
          }
        }
      }
    }
  }

  std::size_t n_;
  std::vector<std::pair<Qubit, Qubit>> edges_{};
  std::vector<std::vector<Qubit>> adj_;
  std::vector<std::size_t> dist_{};
};

}  // namespace qcdsl
