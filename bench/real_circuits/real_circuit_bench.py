"""qcdsl vs Qiskit on REAL circuits, not random ones.

    PYTHONPATH=$HOME/build-qcdsl/python python3 real_circuit_bench.py

Every routing number in the qcdsl repo is measured on random CX circuits. That
is the repo's one real weakness: random circuits have no structure, and real
quantum programs do. This runs the same head-to-head on circuits from QASMBench
(Li, Stein, Krishnamoorthy, Ang; arXiv:2005.13018) -- QFT at several sizes plus an arithmetic adder -- which have the
dense or long-range interaction that actually stresses a router.

Most QASMBench circuits turn out to be already-local on a grid and need zero
SWAPs, which is itself why QFT is the standard routing benchmark. To give even
the near-local circuits something to route, each is run on BOTH a square grid and
a line; a line forces routing for almost any non-trivial interaction graph.

THE FAIRNESS CONTRACT
---------------------
A routing comparison is only fair if both compilers route the SAME thing on the
SAME device. This does not hand Qiskit the full circuit and qcdsl a reduction. It
extracts the two-qubit INTERACTION GRAPH from the benchmark -- the ordered list
of which qubit pairs interact -- and builds an identical circuit for each
compiler from that graph. One-qubit gates do not affect routing and are dropped
from both. Both then route on the same device, at the same search budget.

Per circuit and device, both compilers run at several seeds; the table reports
the mean and a bootstrap 95% interval on the paired difference. If the interval
straddles zero, no claim is made.
"""

from __future__ import annotations

import glob
import math
import os
import statistics

import numpy as np
from qiskit import QuantumCircuit
from qiskit.transpiler import CouplingMap as QkCouplingMap
from qiskit.transpiler import PassManager
from qiskit.transpiler.passes import SabreLayout

import qcdsl
from qcdsl import Circuit, CouplingMap, GateKind, SabreOptions, SabreRouter

SEEDS = 12
BUDGET = 8


def grid_edges(rows, cols):
    e = []
    for i in range(rows):
        for j in range(cols):
            q = i * cols + j
            if j + 1 < cols:
                e.append((q, q + 1))
            if i + 1 < rows:
                e.append((q, q + cols))
    return e


def interaction_pairs(qc):
    """The ordered list of two-qubit interactions -- the only thing that matters
    for routing. One-qubit gates are dropped, from both compilers equally."""
    index = {bit: i for i, bit in enumerate(qc.qubits)}
    pairs = []
    for inst in qc.data:
        if inst.operation.num_qubits == 2:
            a, b = (index[q] for q in inst.qubits)
            pairs.append((a, b))
    return pairs


def build_ours(n, pairs):
    c = Circuit(n)
    for a, b in pairs:
        c.add(GateKind.CX, [a, b])
    return c


def build_theirs(n, pairs):
    qc = QuantumCircuit(n)
    for a, b in pairs:
        qc.cx(a, b)
    return qc


def paired_ci(ours, theirs, reps=4000):
    d = np.asarray(ours, float) - np.asarray(theirs, float)
    rng = np.random.default_rng(0)
    means = [d[rng.integers(0, len(d), len(d))].mean() for _ in range(reps)]
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def main():
    files = sorted(glob.glob(os.path.join(os.path.dirname(__file__), "*.qasm")))
    if not files:
        print("no .qasm files next to this script")
        return

    print("qcdsl vs Qiskit on real circuits (QASMBench). Interaction graph only,")
    print("same device, budget %d, %d seeds. Paired bootstrap 95%% CI.\n"
          % (BUDGET, SEEDS))
    print("%-14s %3s %9s %7s %7s %7s  %s"
          % ("circuit", "n", "device", "qcdsl", "qiskit", "ratio",
             "95% CI (paired)"))

    for path in files:
        name = os.path.splitext(os.path.basename(path))[0]
        qc = QuantumCircuit.from_qasm_file(path)
        n = qc.num_qubits
        pairs = interaction_pairs(qc)
        if not pairs:
            continue

        ours_c = build_ours(n, pairs)
        theirs_c = build_theirs(n, pairs)
        side = math.ceil(math.sqrt(n))

        devices = [
            ("grid%dx%d" % (side, side), CouplingMap.grid(side, side),
             QkCouplingMap(grid_edges(side, side))),
            ("line%d" % n, CouplingMap.line(n),
             QkCouplingMap([(i, i + 1) for i in range(n - 1)])),
        ]

        for dev_name, ours_dev, qk_dev in devices:
            us, qk = [], []
            for seed in range(SEEDS):
                o = SabreOptions()
                o.trials = BUDGET
                o.layout_trials = BUDGET
                o.seed = seed
                us.append(SabreRouter(ours_dev, o).compile(ours_c).swaps_added)
                pm = PassManager([
                    SabreLayout(qk_dev, seed=seed,
                                layout_trials=BUDGET, swap_trials=BUDGET)
                ])
                qk.append(pm.run(theirs_c).count_ops().get("swap", 0))

            us_m = statistics.mean(us)
            qk_m = statistics.mean(qk)
            lo, hi = paired_ci(us, qk)
            verdict = "WIN" if hi < 0 else ("LOSE" if lo > 0 else "TIE")
            print("%-14s %3d %9s %7.1f %7.1f %6.3fx  [%+6.1f,%+6.1f] %s"
                  % (name, n, dev_name, us_m, qk_m,
                     us_m / max(qk_m, 1e-9), lo, hi, verdict))

    print("\nWIN means qcdsl inserts fewer SWAPs on a real, structured circuit.")
    print("Drop more QASMBench .qasm files next to this script to extend the table.")


if __name__ == "__main__":
    main()
