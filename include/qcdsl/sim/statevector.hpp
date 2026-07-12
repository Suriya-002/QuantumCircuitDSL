#pragma once

#include <array>
#include <cmath>
#include <complex>
#include <cstddef>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "qcdsl/circuit.hpp"
#include "qcdsl/gate.hpp"

namespace qcdsl {

/// A 2x2 single-qubit unitary, row-major: {u00, u01, u10, u11}.
template <typename Real>
using Matrix2 = std::array<std::complex<Real>, 4>;

/// The 2x2 matrix for a single-qubit gate kind. `theta` is ignored by
/// non-parametric kinds. Throws for two-qubit kinds and for MEASURE, which are
/// not expressible as a 2x2 unitary on the target alone.
///
/// Phase conventions match Qiskit exactly -- in particular RZ carries the
/// e^{-i*theta/2} global phase -- so a state vector produced here can be
/// compared amplitude-for-amplitude against qiskit.quantum_info.Statevector
/// with no phase fixup.
template <typename Real = double>
Matrix2<Real> matrix_of(GateKind kind, Real theta = Real(0)) {
  using C = std::complex<Real>;
  const C zero(0, 0);
  const C one(1, 0);
  const C imag(0, 1);
  const Real r2 = Real(1) / std::sqrt(Real(2));
  const Real h = theta / Real(2);

  switch (kind) {
    case GateKind::I:
      return {one, zero, zero, one};
    case GateKind::X:
      return {zero, one, one, zero};
    case GateKind::Y:
      return {zero, -imag, imag, zero};
    case GateKind::Z:
      return {one, zero, zero, -one};
    case GateKind::H:
      return {C(r2, 0), C(r2, 0), C(r2, 0), C(-r2, 0)};
    case GateKind::S:
      return {one, zero, zero, imag};
    case GateKind::Sdg:
      return {one, zero, zero, -imag};
    case GateKind::T:
      return {one, zero, zero, C(r2, r2)};
    case GateKind::Tdg:
      return {one, zero, zero, C(r2, -r2)};
    case GateKind::RX:
      return {C(std::cos(h), 0), C(0, -std::sin(h)), C(0, -std::sin(h)),
              C(std::cos(h), 0)};
    case GateKind::RY:
      return {C(std::cos(h), 0), C(-std::sin(h), 0), C(std::sin(h), 0),
              C(std::cos(h), 0)};
    case GateKind::RZ:
      return {C(std::cos(h), -std::sin(h)), zero, zero,
              C(std::cos(h), std::sin(h))};
    default:
      throw std::invalid_argument(std::string("no 2x2 matrix for gate '") +
                                  to_string(kind) + "'");
  }
}

/// Dense state-vector simulator.
///
/// QUBIT ORDERING: qubit q is bit q of the amplitude index, so qubit 0 is the
/// LEAST significant bit. This is Qiskit's convention. X on qubit 1 of a
/// two-qubit register moves the amplitude to index 2 (binary 10), not index 1.
///
/// The simulator is the correctness oracle for the compiler: a pass is valid
/// iff running the circuit before and after the pass yields the same vector.
template <typename Real = double>
class Statevector {
 public:
  using C = std::complex<Real>;

  explicit Statevector(std::size_t num_qubits) : n_(num_qubits) {
    if (num_qubits == 0) {
      throw std::invalid_argument("state vector needs at least one qubit");
    }
    if (num_qubits > 30) {
      throw std::invalid_argument(
          "refusing to allocate a state vector for more than 30 qubits");
    }
    amps_.assign(std::size_t(1) << num_qubits, C(0, 0));
    amps_[0] = C(1, 0);
  }

  [[nodiscard]] std::size_t num_qubits() const noexcept { return n_; }
  [[nodiscard]] std::size_t dim() const noexcept { return amps_.size(); }
  [[nodiscard]] const std::vector<C>& amplitudes() const noexcept {
    return amps_;
  }

  [[nodiscard]] C amplitude(std::size_t index) const {
    if (index >= amps_.size()) {
      throw std::out_of_range("amplitude index out of range");
    }
    return amps_[index];
  }

  /// Sum of |amplitude|^2. Must stay at 1 under unitary evolution -- the
  /// cheapest smoke test that a gate kernel is not corrupting the vector.
  [[nodiscard]] Real norm() const {
    Real acc = 0;
    for (const C& a : amps_) {
      acc += std::norm(a);
    }
    return acc;
  }

  [[nodiscard]] std::vector<Real> probabilities() const {
    std::vector<Real> p(amps_.size());
    for (std::size_t i = 0; i < amps_.size(); ++i) {
      p[i] = std::norm(amps_[i]);
    }
    return p;
  }

  /// Apply a 2x2 unitary to `target`.
  ///
  /// The amplitude pairs that a single-qubit gate couples are exactly those
  /// whose indices differ only in bit `target`. Enumerate them by counting k
  /// over half the vector and INSERTING a zero bit at position `target`: the
  /// bits of k above `target` shift up one place, the bits below stay put.
  /// That gives the "0" partner i; the "1" partner is i | bit. Every pair is
  /// visited exactly once, with no branch in the loop body -- which is what
  /// lets this vectorise later.
  void apply_1q(const Matrix2<Real>& u, Qubit target) {
    check_qubit(target);
    const std::size_t bit = std::size_t(1) << target;
    const std::size_t low = bit - 1;
    const std::size_t half = amps_.size() >> 1;

    for (std::size_t k = 0; k < half; ++k) {
      const std::size_t i = ((k >> target) << (target + 1)) | (k & low);
      const std::size_t j = i | bit;
      const C a = amps_[i];
      const C b = amps_[j];
      amps_[i] = u[0] * a + u[1] * b;
      amps_[j] = u[2] * a + u[3] * b;
    }
  }

  /// Apply a 2x2 unitary to `target`, but only on amplitudes where `control`
  /// is 1. Same pair enumeration; the control bit just gates the update.
  void apply_controlled_1q(const Matrix2<Real>& u, Qubit control,
                           Qubit target) {
    check_qubit(control);
    check_qubit(target);
    if (control == target) {
      throw std::invalid_argument("control and target must differ");
    }
    const std::size_t cbit = std::size_t(1) << control;
    const std::size_t tbit = std::size_t(1) << target;
    const std::size_t low = tbit - 1;
    const std::size_t half = amps_.size() >> 1;

    for (std::size_t k = 0; k < half; ++k) {
      const std::size_t i = ((k >> target) << (target + 1)) | (k & low);
      if ((i & cbit) == 0) {
        continue;
      }
      const std::size_t j = i | tbit;
      const C a = amps_[i];
      const C b = amps_[j];
      amps_[i] = u[0] * a + u[1] * b;
      amps_[j] = u[2] * a + u[3] * b;
    }
  }

  /// Exchange the two wires. Only the amplitudes where the two bits disagree
  /// move; visit each such pair once by fixing (a=1, b=0).
  void apply_swap(Qubit qa, Qubit qb) {
    check_qubit(qa);
    check_qubit(qb);
    if (qa == qb) {
      return;
    }
    const std::size_t abit = std::size_t(1) << qa;
    const std::size_t bbit = std::size_t(1) << qb;

    for (std::size_t i = 0; i < amps_.size(); ++i) {
      if ((i & abit) != 0 && (i & bbit) == 0) {
        const std::size_t j = (i & ~abit) | bbit;
        std::swap(amps_[i], amps_[j]);
      }
    }
  }

  /// Dispatch one gate onto the vector.
  ///
  /// The Gate constructor already enforces that a gate carries exactly
  /// arity(kind) wires, but that invariant lives in another translation unit
  /// and neither the compiler nor a reader can see it here. Re-check it, so the
  /// indexing below is provably in bounds rather than merely correct.
  void apply(const Gate& g) {
    if (g.qubits.size() != arity(g.kind)) {
      throw std::invalid_argument(
          std::string("gate '") + to_string(g.kind) + "' carries " +
          std::to_string(g.qubits.size()) + " wires but needs " +
          std::to_string(arity(g.kind)));
    }
    if (g.qubits.empty()) {
      throw std::invalid_argument("gate carries no wires");
    }
    const Qubit q0 = g.qubits[0];

    switch (g.kind) {
      case GateKind::CX:
      case GateKind::CZ:
      case GateKind::SWAP: {
        if (g.qubits.size() < 2) {
          throw std::invalid_argument("two-qubit gate is missing a wire");
        }
        const Qubit q1 = g.qubits[1];
        if (g.kind == GateKind::SWAP) {
          apply_swap(q0, q1);
        } else {
          const GateKind base =
              (g.kind == GateKind::CX) ? GateKind::X : GateKind::Z;
          apply_controlled_1q(matrix_of<Real>(base), q0, q1);
        }
        return;
      }
      case GateKind::MEASURE:
        throw std::invalid_argument(
            "the state-vector backend does not simulate measurement; strip "
            "measurements before simulating");
      default:
        apply_1q(matrix_of<Real>(g.kind, static_cast<Real>(g.param)), q0);
        return;
    }
  }

  /// Run a whole circuit in program order.
  void run(const Circuit& qc) {
    if (qc.num_qubits() != n_) {
      throw std::invalid_argument(
          "circuit has " + std::to_string(qc.num_qubits()) +
          " qubits but the state vector has " + std::to_string(n_));
    }
    for (const Gate& g : qc.gates()) {
      apply(g);
    }
  }

 private:
  void check_qubit(Qubit q) const {
    if (q >= n_) {
      throw std::out_of_range("qubit index " + std::to_string(q) +
                              " out of range for " + std::to_string(n_) +
                              "-qubit state vector");
    }
  }

  std::size_t n_;
  std::vector<C> amps_{};
};

/// Convenience: simulate a circuit from the all-zero state.
template <typename Real = double>
Statevector<Real> simulate(const Circuit& qc) {
  Statevector<Real> sv(qc.num_qubits());
  sv.run(qc);
  return sv;
}

}  // namespace qcdsl
