# Circuit attribution

The `.qasm` files in this directory are from **QASMBench**, redistributed here
unmodified for reproducibility of the routing benchmark.

> Ang Li, Samuel Stein, Sriram Krishnamoorthy, James Ang.
> "QASMBench: A Low-Level Quantum Benchmark Suite for NISQ Evaluation and
> Simulation." ACM Transactions on Quantum Computing 4(2), 2023. arXiv:2005.13018.
> https://github.com/pnnl/QASMBench

QASMBench is licensed under the BSD 3-Clause License (Battelle / Pacific
Northwest National Laboratory). The included circuits are:

- `qft_n4.qasm`, `qft_n18.qasm`, `qft_n29.qasm` -- quantum Fourier transform
- `adder_n28.qasm` -- ripple-carry adder

Only these circuit files are covered by the QASMBench license; the benchmark
script `real_circuit_bench.py` is part of this repository and under its licence.
