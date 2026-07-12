#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <stdexcept>
#include <string>
#include <vector>

#include "qcdsl/circuit.hpp"
#include "qcdsl/gate.hpp"
#include "qcdsl/ir/dag.hpp"
#include "qcdsl/pass/pass.hpp"

namespace qcdsl {

/// The inverse of a non-parametric gate, or MEASURE for the ones that have no
/// inverse in this gate set. Rotations invert by negating their angle, which is
/// MergeRotations' job, not this one's.
constexpr GateKind inverse_of(GateKind k) noexcept {
  switch (k) {
    case GateKind::I:
      return GateKind::I;
    case GateKind::X:
      return GateKind::X;
    case GateKind::Y:
      return GateKind::Y;
    case GateKind::Z:
      return GateKind::Z;
    case GateKind::H:
      return GateKind::H;
    case GateKind::S:
      return GateKind::Sdg;
    case GateKind::Sdg:
      return GateKind::S;
    case GateKind::T:
      return GateKind::Tdg;
    case GateKind::Tdg:
      return GateKind::T;
    case GateKind::CX:
      return GateKind::CX;
    case GateKind::CZ:
      return GateKind::CZ;
    case GateKind::SWAP:
      return GateKind::SWAP;
    default:
      return GateKind::MEASURE;  // no inverse: rotations, measurement
  }
}

/// CZ and SWAP act the same however their two wires are written down; CX does
/// not. Cancellation has to know the difference: cz(0,1) . cz(1,0) is a pair,
/// cx(0,1) . cx(1,0) is not.
constexpr bool is_symmetric(GateKind k) noexcept {
  return k == GateKind::CZ || k == GateKind::SWAP || arity(k) == 1;
}

namespace detail {

inline bool same_wires(const Gate& a, const Gate& b) {
  if (a.qubits.size() != b.qubits.size()) {
    return false;
  }
  if (is_symmetric(a.kind)) {
    std::vector<Qubit> x = a.qubits;
    std::vector<Qubit> y = b.qubits;
    std::sort(x.begin(), x.end());
    std::sort(y.begin(), y.end());
    return x == y;
  }
  return a.qubits == b.qubits;
}

/// Emit every node except the ones marked dead, keeping program order.
inline Circuit emit_except(const Dag& dag, const std::vector<bool>& dead) {
  Circuit out(dag.num_qubits());
  for (std::size_t i = 0; i < dag.size(); ++i) {
    if (!dead[i]) {
      out.add(dag.node(i).gate);
    }
  }
  return out;
}

}  // namespace detail

/// g . g^-1 -> nothing, when the two are adjacent on every wire they share.
///
/// Adjacency is the whole difficulty. h q0; x q0; h q0 has two H gates that are
/// each other's inverse and are NOT cancellable -- the X sits between them. The
/// DAG answers this directly: are_adjacent(a, b) is true only when b is the
/// immediate successor of a on every shared wire.
class CancelInversePairs final : public Pass {
 public:
  [[nodiscard]] std::string name() const override {
    return "cancel-inverse-pairs";
  }

  [[nodiscard]] Circuit run(const Dag& dag) const override {
    std::vector<bool> dead(dag.size(), false);

    for (std::size_t a = 0; a < dag.size(); ++a) {
      if (dead[a]) {
        continue;
      }
      const Gate& ga = dag.node(a).gate;
      const GateKind inv = inverse_of(ga.kind);
      if (inv == GateKind::MEASURE || ga.kind == GateKind::MEASURE) {
        continue;  // rotations and measurement are not this pass's problem
      }

      for (const std::size_t b : dag.node(a).succs) {
        if (dead[b]) {
          continue;
        }
        const Gate& gb = dag.node(b).gate;
        if (gb.kind == inv && detail::same_wires(ga, gb) &&
            dag.are_adjacent(a, b)) {
          dead[a] = true;
          dead[b] = true;
          break;
        }
      }
    }
    return detail::emit_except(dag, dead);
  }
};

/// RZ(a) . RZ(b) -> RZ(a+b), and likewise for RX and RY.
///
/// Exact, including global phase: rotations about a fixed axis form a group, so
/// RZ(a)RZ(b) IS RZ(a+b) as a matrix, not merely up to phase. The angles are
/// NOT reduced modulo 2*pi -- RZ(2*pi) is -I, not I, and folding it away would
/// silently change the state by a sign.
class MergeRotations final : public Pass {
 public:
  [[nodiscard]] std::string name() const override { return "merge-rotations"; }

  [[nodiscard]] Circuit run(const Dag& dag) const override {
    std::vector<bool> dead(dag.size(), false);
    std::vector<double> angle(dag.size(), 0.0);
    for (std::size_t i = 0; i < dag.size(); ++i) {
      angle[i] = dag.node(i).gate.param;
    }

    for (std::size_t a = 0; a < dag.size(); ++a) {
      if (dead[a] || !is_parametric(dag.node(a).gate.kind)) {
        continue;
      }
      // Walk forward absorbing every same-axis rotation that is still adjacent.
      std::size_t head = a;
      for (;;) {
        const std::size_t nxt =
            dag.next_on_wire(head, dag.node(a).gate.qubits[0]);
        if (nxt == Dag::kNone || dead[nxt]) {
          break;
        }
        if (dag.node(nxt).gate.kind != dag.node(a).gate.kind) {
          break;  // different axis: RZ then RX does not fold
        }
        angle[a] += angle[nxt];
        dead[nxt] = true;
        head = nxt;
      }
    }

    Circuit out(dag.num_qubits());
    for (std::size_t i = 0; i < dag.size(); ++i) {
      if (dead[i]) {
        continue;
      }
      const Gate& g = dag.node(i).gate;
      out.add(Gate(g.kind, g.qubits, angle[i]));
    }
    return out;
  }
};

/// Drop I gates and rotations by (numerically) zero.
///
/// Only |theta| < eps is dropped, never theta mod 2*pi. RZ(2*pi) = -I is a
/// global sign, and this library preserves global phase exactly.
class RemoveIdentities final : public Pass {
 public:
  explicit RemoveIdentities(double eps = 1e-12) : eps_(eps) {}

  [[nodiscard]] std::string name() const override {
    return "remove-identities";
  }

  [[nodiscard]] Circuit run(const Dag& dag) const override {
    std::vector<bool> dead(dag.size(), false);
    for (std::size_t i = 0; i < dag.size(); ++i) {
      const Gate& g = dag.node(i).gate;
      dead[i] = g.kind == GateKind::I ||
                (is_parametric(g.kind) && std::abs(g.param) < eps_);
    }
    return detail::emit_except(dag, dead);
  }

 private:
  double eps_;
};

/// Rewrite every two-qubit gate into CX plus single-qubit gates.
///
/// This is gate-set targeting: real hardware exposes one native entangling
/// gate, and everything else has to be expressed in terms of it. Both rewrites
/// are exact matrix identities, not approximations:
///
///   cz a,b   =  h b ; cx a,b ; h b          (H Z H = X)
///   swap a,b =  cx a,b ; cx b,a ; cx a,b
class DecomposeToCx final : public Pass {
 public:
  [[nodiscard]] std::string name() const override { return "decompose-to-cx"; }

  [[nodiscard]] Circuit run(const Dag& dag) const override {
    Circuit out(dag.num_qubits());
    // Node ids are in program order, and program order is always a valid
    // topological order. Keep it: a rewrite pass should not reorder
    // gratuitously.
    for (std::size_t id = 0; id < dag.size(); ++id) {
      const Gate& g = dag.node(id).gate;
      const bool two_qubit = g.kind == GateKind::CZ || g.kind == GateKind::SWAP;
      if (!two_qubit) {
        out.add(g);
        continue;
      }
      if (g.qubits.size() < 2) {
        throw std::invalid_argument("two-qubit gate is missing a wire");
      }
      const Qubit a = g.qubits[0];
      const Qubit b = g.qubits[1];

      if (g.kind == GateKind::CZ) {
        out.add(GateKind::H, {b});
        out.add(GateKind::CX, {a, b});
        out.add(GateKind::H, {b});
      } else {
        out.add(GateKind::CX, {a, b});
        out.add(GateKind::CX, {b, a});
        out.add(GateKind::CX, {a, b});
      }
    }
    return out;
  }
};

}  // namespace qcdsl
