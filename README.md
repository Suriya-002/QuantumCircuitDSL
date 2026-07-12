# qcdsl

[![ci](https://github.com/Suriya-002/QuantumCircuitDSL/actions/workflows/ci.yml/badge.svg)](https://github.com/Suriya-002/QuantumCircuitDSL/actions/workflows/ci.yml)

A quantum circuit **compiler** in C++17: a DAG intermediate representation,
optimisation passes over that IR, a SIMD state-vector backend to check the
passes preserve semantics, and Python bindings so it can be driven from a
notebook.

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
| State-vector simulator (scalar reference) | in progress |
| DAG IR (dependency graph, topological schedule) | planned |
| Passes: gate fusion, Clifford+T decomposition | planned |
| OpenQASM 3.0 import / export | planned |
| SIMD (AVX-512) + OpenMP gate kernels, benchmarks | planned |

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
