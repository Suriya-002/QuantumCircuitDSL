#pragma once

#include <cstddef>
#include <cstdint>
#include <initializer_list>
#include <stdexcept>
#include <string>
#include <vector>

namespace qcdsl {

using Qubit = std::size_t;

/// Gate kinds recognised by the front end.
/// Kept flat (not a class hierarchy) so gates are trivially copyable and can
/// live contiguously in a vector — the DAG and the simulator both iterate them
/// hot, and virtual dispatch per gate would dominate the inner loop.
enum class GateKind : std::uint8_t {
  I,
  X,
  Y,
  Z,
  H,
  S,
  Sdg,
  T,
  Tdg,
  RX,
  RY,
  RZ,
  CX,
  CZ,
  SWAP,
  MEASURE,
};

/// Human-readable name, used by the QASM writer and by test failure messages.
constexpr const char* to_string(GateKind k) noexcept {
  switch (k) {
    case GateKind::I:
      return "id";
    case GateKind::X:
      return "x";
    case GateKind::Y:
      return "y";
    case GateKind::Z:
      return "z";
    case GateKind::H:
      return "h";
    case GateKind::S:
      return "s";
    case GateKind::Sdg:
      return "sdg";
    case GateKind::T:
      return "t";
    case GateKind::Tdg:
      return "tdg";
    case GateKind::RX:
      return "rx";
    case GateKind::RY:
      return "ry";
    case GateKind::RZ:
      return "rz";
    case GateKind::CX:
      return "cx";
    case GateKind::CZ:
      return "cz";
    case GateKind::SWAP:
      return "swap";
    case GateKind::MEASURE:
      return "measure";
  }
  return "unknown";
}

/// Number of qubits a gate kind acts on.
constexpr std::size_t arity(GateKind k) noexcept {
  switch (k) {
    case GateKind::CX:
    case GateKind::CZ:
    case GateKind::SWAP:
      return 2;
    default:
      return 1;
  }
}

/// True for gates that take a continuous rotation angle.
constexpr bool is_parametric(GateKind k) noexcept {
  return k == GateKind::RX || k == GateKind::RY || k == GateKind::RZ;
}

/// A single gate instance: what it is, which qubits it touches, its angle.
struct Gate {
  GateKind kind{GateKind::I};
  std::vector<Qubit> qubits{};
  double param{0.0};

  Gate() = default;

  Gate(GateKind k, std::initializer_list<Qubit> qs, double p = 0.0)
      : kind(k), qubits(qs), param(p) {
    validate();
  }

  Gate(GateKind k, std::vector<Qubit> qs, double p = 0.0)
      : kind(k), qubits(std::move(qs)), param(p) {
    validate();
  }

  [[nodiscard]] std::size_t width() const noexcept { return qubits.size(); }

 private:
  void validate() const {
    if (qubits.size() != arity(kind)) {
      throw std::invalid_argument(std::string("gate '") + to_string(kind) +
                                  "' expects " + std::to_string(arity(kind)) +
                                  " qubit(s), got " +
                                  std::to_string(qubits.size()));
    }
    if (qubits.size() == 2 && qubits[0] == qubits[1]) {
      throw std::invalid_argument(std::string("gate '") + to_string(kind) +
                                  "' cannot act twice on qubit " +
                                  std::to_string(qubits[0]));
    }
  }
};

}  // namespace qcdsl
