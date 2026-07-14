# qcdsl

[![ci](https://github.com/Suriya-002/QuantumCircuitDSL/actions/workflows/ci.yml/badge.svg)](https://github.com/Suriya-002/QuantumCircuitDSL/actions/workflows/ci.yml)

A quantum circuit **compiler** in C++17: a DAG intermediate representation,
optimisation passes over that IR, layout and routing against a device coupling
map, a SIMD state-vector backend to check the passes preserve semantics, and
Python bindings so it can be driven from a notebook.

**On a 25-qubit grid it inserts 11% fewer SWAPs than Qiskit's transpiler**, at
matched search budget, with a bootstrap interval that clears zero by a wide
margin — because Qiskit's `SabreLayout` computes a good layout and then throws it
away. It also takes **2.3x to 5.1x longer** to do it. Both numbers are below, and
both are in the benchmarks.

Header-only core. No linear-algebra dependency — the state-vector kernels are
hand-written, because the point of the project is the kernels.

```cpp
#include "qcdsl/qcdsl.hpp"

qcdsl::Circuit qc(2);
qc.add(qcdsl::GateKind::H,  {0})
  .add(qcdsl::GateKind::CX, {0, 1});

qc.size();   // 2
qc.depth();  // 2
```

```python
from qcdsl import Circuit, GateKind

qc = Circuit(3)
for q in range(3):
    qc.add(GateKind.H, [q])

qc.size()   # 3
qc.depth()  # 1  -- disjoint wires share a layer
```

## Why

Most quantum "circuit libraries" are simulators with a builder API bolted on.
This one is the other way round: the simulator exists to validate the compiler.
The interesting surface is the IR and the passes that rewrite it — gate fusion,
decomposition into a hardware-constrained target gate set, and the scheduling
that falls out of the dependency graph.

## Status

Under active development. What is in `main` is what is implemented and tested —
nothing is listed here that does not have a test behind it.

| Component | State |
|---|---|
| Gate / `Circuit` front end, depth analysis | done |
| CMake + GoogleTest + pytest + CI + clang-tidy | done |
| State-vector simulator (scalar reference) | done |
| Cross-validation against Qiskit on random circuits | done |
| DAG IR (dependency graph, topological schedule) | done |
| Passes: cancellation, rotation fusion, CX gate-set targeting | done |
| OpenQASM 3.0 import / export (two-way Qiskit interop) | done |
| SIMD (AVX-512) + OpenMP gate kernels, benchmarks | done |
| Layout + routing (SABRE) against a device coupling map | done |
| Randomised layout restarts, parallelised (beats Qiskit on grids) | done |
| Cheaper layout search (95% of a compile is `find_layout`) | next |
| Full single-qubit fusion (U gate + global phase) | planned |
| Cache-blocked multi-gate kernel (kill the bandwidth wall) | planned |

143 C++ tests, 333 Python tests, 96% line coverage and 100% function coverage
(gated at 90% in CI). The C++ suite is a GoogleTest *typed* suite -- the whole
battery runs against both the `double` and the `float` instantiation, because a
templated backend with one tested instantiation is a half-tested backend.

## Correctness

The simulator is the compiler's oracle: a pass is valid iff the state vector is
unchanged across it. That only means something if the simulator itself is right,
so it is checked against an independent implementation -- `qiskit.quantum_info.Statevector`
-- on every gate, every wire, every ordered qubit pair, and 48 seeded random
circuits, compared amplitude for amplitude including global phase.

This is not ceremony. Transpose the `Y` gate (`[[0,i],[-i,0]]` instead of
`[[0,-i],[i,0]]`) and every C++ test still passes -- `Y * Y` is the identity
under either sign convention, so nothing in a self-written suite pins it down.
The Qiskit comparison fails 21 tests immediately.

The DAG is pinned the same way, by two tests that fail in opposite directions:

* `Dag::depth()` (longest path through the graph) must equal `Circuit::depth()`
  (a walk over the wires). A **spurious** edge lengthens the critical path and
  the two numbers diverge.
* Every topological order of the DAG is a legal schedule, so all of them must
  produce the same state vector. A **missing** edge lets some random schedule
  run two dependent gates out of order, and the state changes.

Serialise the DAG completely and the state tests still pass -- an
over-constrained IR computes the right answer, it just schedules badly. Only the
depth cross-check catches it. Drop the dependency on a two-qubit gate's second
wire and the state tests fail. Neither test alone is sufficient; together they
admit exactly one edge set.

## OpenQASM 3

`to_qasm3` / `from_qasm3`: a tokeniser, a recursive-descent parser and an angle
*expression* evaluator, because Qiskit writes `rz(pi/2) q[0];` as readily as it
writes a float. Comments, barriers, both measurement spellings and the OpenQASM 2
register syntax are all accepted; malformed input raises with a line number.

Round-tripping through one's own parser proves nothing -- an emitter and a parser
that share a private dialect will agree perfectly and still be unable to talk to
anything else. Conformance is checked in **both** directions against Qiskit:

```
export:  qcdsl emits  ->  qiskit.qasm3.loads  ->  same state vector
import:  qiskit.qasm3.dumps  ->  qcdsl parses  ->  same state vector
```

on randomly generated circuits, amplitude for amplitude.

## Optimisation

Four passes, all exact -- none of them approximates an angle or drops a global
phase, which is why each can be validated by simulating both sides and demanding
amplitude-for-amplitude equality:

| pass | rewrite |
|---|---|
| `CancelInversePairs` | `g . g^-1` -> nothing, when adjacent on every shared wire |
| `MergeRotations` | `RZ(a) . RZ(b)` -> `RZ(a+b)`, same axis only |
| `RemoveIdentities` | drops `I` and zero-angle rotations |
| `DecomposeToCx` | `CZ` -> `H CX H`, `SWAP` -> three `CX` |

`PassManager` runs a pipeline to a fixed point, because passes enable one
another: cancelling `H . H` can bring two rotations together, merging those can
produce a zero angle, and removing that identity can expose a fresh inverse pair.

`bench/optimise.cpp` produces the table below -- 300 circuits, 6 qubits, 200
gates each. Positive is a reduction.

| pipeline | gates | depth | CX-only |
|---|---|---|---|
| optimise only | **-40.3%** | **-45.1%** | no |
| lower first | -21.7% | -20.5% | yes |
| optimise -> lower -> optimise | **-22.6%** | **-22.0%** | yes |

Optimising before lowering beats lowering first, which is the expected direction:
once `CZ` has become `H CX H` the optimiser has to rediscover structure the
higher-level gate stated outright.

On *uniformly random* circuits the same pipeline **adds 30% more gates**. That is
not a regression. `DecomposeToCx` turns one `CZ` into three gates and one `SWAP`
into three; on a circuit with nothing to cancel there is nothing to pay for it.
Decomposition is gate-set targeting, not optimisation, and hardware compliance
has a price. Reporting only the flattering workload would hide that.

## Layout and routing

Everything above this line assumes a machine where any qubit can talk to any
other. No such machine exists. On real hardware `cx q[0], q[17]` is not an
instruction -- it is a request, and the compiler has to satisfy it by physically
walking those two qubits next to each other with SWAPs.

`CouplingMap` is the device graph (line, ring, grid, all-to-all, or arbitrary
edges) with an all-pairs BFS distance matrix. `SabreRouter` implements SABRE
(Li, Ding & Xie, ASPLOS 2019) -- the algorithm Qiskit ships:

- take the **front layer** of the DAG, the gates with nothing left blocking them;
- execute every gate that is already legal (a two-qubit gate is legal iff its
  qubits are coupled);
- if none are, score every SWAP on an edge touching a front-layer qubit and apply
  the best. The score is the mean distance the front layer still has to travel,
  plus a **lookahead** term over the next few gates so the router does not fix
  one gate by wrecking the next, scaled by a **decay** on recently-moved qubits
  so it does not shuffle the same qubit back and forth.

`find_layout` is SabreLayout: route forwards, take the mapping you ended on,
route the *reversed* circuit from it, and use *that* final mapping as the initial
layout. A good final mapping for a circuit is a good initial mapping for its
reverse, because the router has already worked out which qubits want to be near
which.

### Routing permutes the qubits, and forgetting that is the classic bug

The routed circuit does **not** produce the original state vector. It produces
the original state with its amplitude indices permuted by `final_layout` -- the
SWAPs moved the logical qubits, and logical qubit `l` now lives on physical qubit
`final_layout[l]`. Compare the two naively and a *correct* router looks broken;
skip the comparison and a *broken* router looks correct. `permute_index` closes
the gap, and one test deliberately checks that the unpermuted comparison FAILS,
so the check cannot pass vacuously.

### Against Qiskit's transpiler

The comparison that matters. Same circuits, same devices, **same search budget on
both sides** (8 layout restarts, 8 routing trials), 120 random circuits, paired
bootstrap 95% interval on the difference. `bench/layout_ablation.py`.

| device | qubits | qcdsl | qiskit | | 95% CI on paired diff |
|---|---:|---:|---:|---|---|
| grid(3x3) | 9 | **8.0** | 8.4 | **0.945x** | [-0.71, -0.23] |
| line(12) | 12 | **78.5** | 80.2 | **0.979x** | [-2.63, -0.78] |
| grid(4x4) | 16 | **40.9** | 45.6 | **0.895x** | [-5.36, -4.22] |
| grid(5x5) | 25 | **103.5** | 116.3 | **0.890x** | [-13.80, -11.75] |
| ring(12) | 12 | 61.4 | 61.4 | 1.001x | tie |
| ring(7) | 7 | 14.8 | 13.9 | 1.061x | we lose |

**11% fewer SWAPs than Qiskit on a 25-qubit grid, and the margin grows with the
device**: 0.945x at 9 qubits, 0.890x at 25.

**And it costs 2.3x to 5.1x the compile time.** Both columns are in
`bench/layout_speed.py`, side by side, permanently. A better circuit produced
five times more slowly is a trade, not a victory, and the reader is entitled to
see the price. `bench/layout_pareto.py` publishes the whole curve.

### Why we win: SabreLayout throws away its own best answer

The forward-reverse iteration is a **descent**, and it is **not monotone**. It
finds a good layout and then wanders off it:

| | iteration 2 | iteration 3 |
|---|---:|---:|
| grid(2x4) | **12.7** swaps | 13.4 swaps |
| ring(7) | **16.4** swaps | 16.7 swaps |

`find_layout` used to return the layout from the **last** iteration. So did
Qiskit's `SabreLayout` -- run `max_iterations`, take the final one. Keeping the
**best** iterate instead is free, and it is most of the win above.

The second half is that the descent can only improve the layout it is handed; it
cannot leave that layout's basin. `find_layout` always started from the identity,
so however many `trials` it ran, it explored exactly **one** basin. `trials`
re-rolls tie-breaks inside a basin. `layout_trials` restarts from a different
random permutation and reaches a different one. Qiskit does this; we did not,
and it was the entire original 5-27% gap.

### What each piece is actually worth

`bench/layout_ablation.py`. Four configurations, 300 circuits:

| | line(7) | grid(3x3) | grid(5x5) |
|---|---:|---:|---:|
| **A** identity layout, no search | 29.2 | 15.1 | 146.7 |
| **B** descent from identity | 21.0 | 9.8 | 115.8 |
| **C** descent + 8 random restarts | **20.0** | **8.0** | **103.5** |
| **D** 8 random layouts, *no* descent | 24.0 | 12.4 | 133.7 |

**D is the control**, and it is what makes this an experiment rather than a
leaderboard. It gets the same number of random layouts as C and the same routing
heuristic, but no SABRE refinement at all. C beats D everywhere, so the descent
is real work and not multi-start in a trench coat. Note also **B ≈ D**: *one*
descent is worth about *eight* random draws.

The restarts are worth **~5% on 1D and ~21% on 2D** -- four times more on a grid
than on a chain. On a chain the good layout is nearly forced, so restarting lands
you back where you began; a grid's basin structure is rich enough that
multi-start is what finds the good one.

### The gaps that were not real

Almost every deficit worth chasing here turned out to be a **seven-qubit
artifact**. Router only, identity layout, 250 circuits:

| n | line | ring |
|---:|---:|---:|
| 5 | 1.060x | 1.079x |
| 7 | 1.043x | 1.069x |
| 9 | 1.012x | 1.040x |
| **12** | **0.997x** | **1.009x** |

The deficit shrinks monotonically with device size and is **gone by n=12**. Rings
show it worse than lines because they have fewer edges and fewer swap candidates,
so tie-breaking dominates; as the device grows the score landscape gets richer
and the greedy choice stops being a coin flip. There is no ring pathology. An
afternoon went into looking for one.

Two things this repository now does as a matter of course, because both were
learned the hard way:

**Report an interval, not a mean.** Thirty random circuits has a standard error
of roughly ±5%, and the effects being argued about here are 2-6%. The *same code*
on the *same topology* scored 1.07x against Qiskit on one 30-circuit sample and
0.99x on another. Every benchmark here now runs 120-300 circuits and prints a
bootstrap 95% interval on the paired difference. If it straddles zero, no claim
is made.

**Your threads are part of the experiment.** OpenMP threads busy-wait by default.
When `layout_speed.py` interleaved the two routers in one loop, our eight threads
were still spinning while Qiskit's SABRE -- which also parallelises -- tried to
run. It measured Qiskit at 16.2 ms on a 4x4 grid against a true 5.3 ms, and
reported that we were **faster** than it. We are not. `OMP_WAIT_POLICY` is now
forced to `PASSIVE` and the two are timed in separate passes.

## Kernels

Three kernels compute the same thing at different speeds. The scalar one is the
reference the other two are validated against -- on every target qubit, on random
circuits, and at the sizes where each one is at its worst -- so "speedup" always
means speedup over code known to be correct.

**The AVX-512 kernel is compiled unconditionally and gated at run time.** Building
with `-mavx512f` would let the compiler emit AVX-512 anywhere it liked, and the
binary would die with SIGILL on a machine without it, including some CI runners.
Instead the vector path carries `__attribute__((target("avx512f")))` and
`__builtin_cpu_supports` decides at the first call whether it may run.

Amplitudes whose bit `t` is zero come in contiguous runs of `2^t`. For `t >= 2` a
run is at least one full 512-bit register, so both halves of every pair load as a
straight vector -- no gather, no shuffle. Targets 0 and 1 interleave inside a
register and fall back to scalar: two of `n` targets, stated rather than hidden.
There is no `float` vector kernel yet.

`bench/simulate.cpp`, 100 single-qubit gates, best of 3, 8 threads:

| qubits | amplitudes | scalar | AVX-512 | +OpenMP | SIMD | GB/s |
|---:|---:|---:|---:|---:|---:|---:|
| 14 | 16 K | 5.0 ms | 1.5 ms | 8.7 ms | **3.4x** | 35 |
| 16 | 66 K | 17.3 ms | 5.6 ms | 9.2 ms | **3.1x** | 38 |
| 18 | 262 K | 57.7 ms | 18.2 ms | 30.6 ms | **3.2x** | 46 |
| 20 | 1.0 M | 241 ms | 112 ms | **91.8 ms** | 2.2x | 37 |
| 22 | 4.2 M | 1054 ms | 643 ms | 749 ms | 1.6x | 21 |
| 24 | 16.8 M | 4084 ms | 2253 ms | 2782 ms | 1.8x | 24 |

### Is the baseline fair?

The obvious objection is that the scalar kernel was denied AVX-512, so of course
it lost. Give it everything -- `-O3 -march=native`, autovectoriser unleashed --
and at 22 qubits it improves by **nine percent**. The hand-written kernel still
wins 2.6x.

The reason is the whole point. The speedup is not from *using* AVX-512
instructions, it is from restructuring the loop so that AVX-512 *becomes usable*.
The scalar kernel computes each index as `((k >> t) << (t + 1)) | (k & low)`. GCC
sees bit arithmetic, cannot prove the resulting addresses are adjacent, and
declines to vectorise. Blocking the loop makes contiguity a property of the loop
bounds instead of a theorem about bit twiddling. No autovectoriser recovers that.

### OpenMP wins exactly once, and the GB/s column says why

Watch the achieved bandwidth. Each amplitude pair moves 64 bytes -- two 16-byte
loads and two 16-byte stores -- so this is computed, not guessed. It holds at
35-46 GB/s through 20 qubits and then **collapses to 21** at 22.

That is the last-level cache boundary. At 20 qubits the state vector is 16 MB and
still resident, so eight threads have real work to do and OpenMP finally wins
(2.6x over scalar, beating the serial vector kernel's 2.2x). At 22 qubits it is
67 MB, the kernel is waiting on DRAM, and eight threads only contend for one
memory controller. Below 20 qubits the fork-join costs more than the gate.

The window where threads help is one doubling wide, and it moves with the cache
size and the channel count of the machine. So **`Kernel::Auto` does not use
threads** -- a default that guesses wrong most of the time is worse than one that
does not guess. `Kernel::Parallel` is there for anyone whose machine has the
bandwidth headroom.

The real fix is not more threads, it is fewer sweeps: every gate currently streams
the whole state vector through the CPU once. Applying several gates to one
cache-resident tile before moving on divides the memory traffic by the number of
gates fused, turning a bandwidth-bound kernel back into a compute-bound one. That
is the next kernel, and the point at which threads become worth having again.

### Profile

`scripts/profile.sh` runs `perf`, falling back to Callgrind where perf has no
usable PMU -- WSL2 does not expose hardware counters to the guest, so Callgrind's
simulated cache is the instrument there. At 18 qubits:

| | |
|---|---|
| instructions in `apply_1q` | **99.86%** of 13.2 billion |
| branch mispredictions | 2,788 of 774 million -- **0.0004%** |
| last-level cache miss rate | 0.0% |

The hot loop is the entire program: the `Gate` objects, the `Circuit` iteration
and the kernel dispatch cost nothing measurable. The mispredict rate is the
branch-free pair enumeration paying off -- it is why the loop vectorises at all.
And the 0% LL miss rate at 18 qubits is the other end of the bandwidth story: at
that size nothing reaches DRAM, which is exactly why the table reads 46 GB/s
there and 21 GB/s at 22 qubits.

## Build

Requires CMake ≥ 3.20 and a C++17 compiler. GoogleTest and pybind11 are pulled
in by `FetchContent`; nothing needs to be installed by hand.

```sh
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release
cmake --build build --parallel
ctest --test-dir build --output-on-failure
PYTHONPATH=build/python python -m pytest python/tests -v
```

Options: `QCDSL_BUILD_TESTS`, `QCDSL_BUILD_PYTHON`, `QCDSL_BUILD_BENCH`,
`QCDSL_COVERAGE`.

## Design notes

**`GateKind` is a flat enum, not a class hierarchy.** Gates are iterated in the
inner loop of both the DAG builder and the simulator. A virtual call per gate
would cost more than the gate application itself for one- and two-qubit gates on
small registers. Flat enum + `switch` keeps `Gate` trivially copyable and keeps
the gate list contiguous.

**Depth follows Qiskit's definition.** The critical path along any wire; gates on
disjoint qubits share a layer; a two-qubit gate synchronises both of its wires.
That is the number a scheduler actually cares about, so it is the number the
front end reports.

## Licence

MIT.
