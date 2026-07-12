#pragma once

#include <algorithm>
#include <cstddef>
#include <stdexcept>
#include <string>
#include <vector>

#include "qcdsl/gate.hpp"

namespace qcdsl {

/// An ordered sequence of gates over a fixed qubit register.
///
/// This is the *front-end* representation: program order, exactly as written.
/// The DAG IR (built from a Circuit) is what the compilation passes consume;
/// it discards program order in favour of true data dependencies.
class Circuit {
 public:
  explicit Circuit(std::size_t num_qubits) : num_qubits_(num_qubits) {
    if (num_qubits == 0) {
      throw std::invalid_argument("circuit must have at least one qubit");
    }
  }

  Circuit& add(const Gate& g) {
    for (Qubit q : g.qubits) {
      if (q >= num_qubits_) {
        throw std::out_of_range("qubit index " + std::to_string(q) +
                                " out of range for " +
                                std::to_string(num_qubits_) + "-qubit circuit");
      }
    }
    gates_.push_back(g);
    return *this;
  }

  Circuit& add(GateKind k, std::initializer_list<Qubit> qs, double p = 0.0) {
    return add(Gate(k, qs, p));
  }

  [[nodiscard]] std::size_t num_qubits() const noexcept { return num_qubits_; }

  /// Number of gate instances (Qiskit calls this `size`).
  [[nodiscard]] std::size_t size() const noexcept { return gates_.size(); }

  [[nodiscard]] const std::vector<Gate>& gates() const noexcept {
    return gates_;
  }

  /// Circuit depth: the length of the longest chain of gates along any wire.
  ///
  /// Gates on disjoint qubits share a layer. A two-qubit gate synchronises both
  /// of its wires — it must wait for the later of the two, and both advance
  /// together afterwards. This is the same definition Qiskit's
  /// `QuantumCircuit.depth()` uses, and it is the critical-path length the
  /// scheduler will later care about.
  [[nodiscard]] std::size_t depth() const {
    std::vector<std::size_t> layer(num_qubits_, 0);
    std::size_t max_layer = 0;

    for (const Gate& g : gates_) {
      std::size_t ready = 0;
      for (Qubit q : g.qubits) {
        ready = std::max(ready, layer[q]);
      }
      const std::size_t placed = ready + 1;
      for (Qubit q : g.qubits) {
        layer[q] = placed;
      }
      max_layer = std::max(max_layer, placed);
    }
    return max_layer;
  }

  /// How many gates act on `q`.
  [[nodiscard]] std::size_t gate_count_on(Qubit q) const {
    if (q >= num_qubits_) {
      throw std::out_of_range("qubit index out of range");
    }
    std::size_t n = 0;
    for (const Gate& g : gates_) {
      if (std::find(g.qubits.begin(), g.qubits.end(), q) != g.qubits.end()) {
        ++n;
      }
    }
    return n;
  }

 private:
  std::size_t num_qubits_;
  std::vector<Gate> gates_{};
};

}  // namespace qcdsl
