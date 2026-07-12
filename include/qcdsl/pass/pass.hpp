#pragma once

#include <cstddef>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "qcdsl/circuit.hpp"
#include "qcdsl/ir/dag.hpp"

namespace qcdsl {

/// What one pass did to one circuit.
struct PassStats {
  std::string pass;
  std::size_t gates_before = 0;
  std::size_t gates_after = 0;
  std::size_t depth_before = 0;
  std::size_t depth_after = 0;

  [[nodiscard]] bool changed() const noexcept {
    return gates_before != gates_after || depth_before != depth_after;
  }
};

/// A compilation pass: reads the dependency graph, emits a rewritten circuit.
///
/// THE CONTRACT: a pass must not change what the circuit computes. The state
/// vector before and the state vector after must agree amplitude for amplitude,
/// global phase included. Every pass in this library is exact -- none of them
/// approximate an angle, drop a phase, or resynthesise a rotation. That is why
/// each one can be tested by simulating both sides and demanding equality; a
/// pass that only preserved the state "up to global phase" could not be.
///
/// Passes consume a Dag rather than a Circuit because the questions they need
/// answered -- are these two gates adjacent, do these two gates commute, what
/// else touches this wire -- are graph questions. A flat gate list cannot
/// answer them.
class Pass {
 public:
  Pass() = default;
  Pass(const Pass&) = default;
  Pass(Pass&&) = default;
  Pass& operator=(const Pass&) = default;
  Pass& operator=(Pass&&) = default;
  virtual ~Pass() = default;

  [[nodiscard]] virtual std::string name() const = 0;
  [[nodiscard]] virtual Circuit run(const Dag& dag) const = 0;

  /// Convenience: build the IR, run, hand back a circuit.
  [[nodiscard]] Circuit operator()(const Circuit& qc) const {
    return run(Dag(qc));
  }
};

/// An ordered pipeline of passes.
class PassManager {
 public:
  PassManager& add(std::shared_ptr<Pass> p) {
    passes_.push_back(std::move(p));
    return *this;
  }

  [[nodiscard]] std::size_t size() const noexcept { return passes_.size(); }

  /// One sweep through the pipeline.
  Circuit run(const Circuit& input) {
    stats_.clear();
    Circuit qc = input;
    for (const std::shared_ptr<Pass>& p : passes_) {
      qc = run_one(*p, qc);
    }
    return qc;
  }

  /// Sweep until nothing changes. Passes enable each other: cancelling an
  /// inverse pair can make two rotations adjacent, merging those rotations can
  /// produce a zero angle, and removing that identity can expose a new inverse
  /// pair. One sweep leaves work on the table.
  Circuit run_to_fixed_point(const Circuit& input,
                             std::size_t max_sweeps = 16) {
    stats_.clear();
    Circuit qc = input;
    sweeps_ = 0;

    for (std::size_t sweep = 0; sweep < max_sweeps; ++sweep) {
      const std::size_t before = qc.size();
      for (const std::shared_ptr<Pass>& p : passes_) {
        qc = run_one(*p, qc);
      }
      ++sweeps_;
      if (qc.size() == before) {
        break;  // a sweep that removes no gates will not remove any next time
      }
    }
    return qc;
  }

  [[nodiscard]] const std::vector<PassStats>& stats() const noexcept {
    return stats_;
  }
  [[nodiscard]] std::size_t sweeps() const noexcept { return sweeps_; }

 private:
  Circuit run_one(const Pass& p, const Circuit& qc) {
    const Dag dag(qc);
    PassStats s;
    s.pass = p.name();
    s.gates_before = qc.size();
    s.depth_before = dag.depth();

    Circuit out = p.run(dag);

    s.gates_after = out.size();
    s.depth_after = Dag(out).depth();
    stats_.push_back(s);
    return out;
  }

  std::vector<std::shared_ptr<Pass>> passes_{};
  std::vector<PassStats> stats_{};
  std::size_t sweeps_ = 0;
};

}  // namespace qcdsl
