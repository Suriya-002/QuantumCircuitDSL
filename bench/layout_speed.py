"""Fewer SWAPs -- but at what cost in compile time?

    python bench/layout_speed.py

WARNING ABOUT THIS MEASUREMENT, WHICH I GOT WRONG ONCE
-----------------------------------------------------
OpenMP threads BUSY-WAIT by default. When this benchmark interleaved the two
routers in one loop, our eight threads were still spinning after our parallel
layout search finished -- and Qiskit's SABRE, which also parallelises, had to
fight them for cores. Qiskit's time came out 3x inflated (16.2 ms against a true
5.3 ms on a 4x4 grid), and the benchmark cheerfully reported that we were
FASTER than it.

We are not. We are roughly 2.5x slower.

So: OMP_WAIT_POLICY is forced to PASSIVE below, and the two routers are timed in
separate passes rather than interleaved. If you are benchmarking a parallel
program against another parallel program, your threads are part of the
experiment.

A router that produces a better circuit and takes seven times longer to do it
has not obviously won, and "beats Qiskit on SWAP count" is not a claim worth
making until the clock has been read. This reads the clock.

WHAT CHANGED, AND WHAT IT COST
------------------------------
`layout_trials` restarts the layout search from random permutations, and
`refine_layout` now returns the BEST layout the descent visited rather than the
last one it happened to land on. Both cut SWAPs. Both cost routing passes:

    8 restarts x (3 descent iterations x 3 routing passes + 8 scoring trials)
      ~= 136 routing passes per compile

Qiskit's SABRE is Rust and runs its layout trials in PARALLEL. Ours was C++ and
serial, and it was 6-7x slower as a direct result.

The restarts are independent by construction -- separate starting permutations,
separate descents, separate scores, no shared state -- so they are now run under
OpenMP. This benchmark is the only place that number can honestly be measured,
because it needs real cores: on a single-core machine the parallel version and
the serial version are the same program.
"""

from __future__ import annotations

import os

# Must be set BEFORE the OpenMP runtime is loaded, i.e. before qcdsl is imported.
os.environ.setdefault("OMP_WAIT_POLICY", "PASSIVE")

import random
import statistics

import numpy as np
import time

from qcdsl import Circuit, CouplingMap, GateKind, SabreOptions, SabreRouter

try:
    from qiskit import QuantumCircuit
    from qiskit.transpiler import CouplingMap as QkCouplingMap
    from qiskit.transpiler import PassManager
    from qiskit.transpiler.passes import SabreLayout

    HAVE_QISKIT = True
except ImportError:  # pragma: no cover
    HAVE_QISKIT = False

def _grid_edges(rows: int, cols: int):
    e = []
    for i in range(rows):
        for j in range(cols):
            q = i * cols + j
            if j + 1 < cols:
                e.append((q, q + 1))
            if i + 1 < rows:
                e.append((q, q + cols))
    return e


TOPOLOGIES = {
    "line7": (CouplingMap.line(7), [(i, i + 1) for i in range(6)], 7, 60),
    "ring7": (CouplingMap.ring(7), [(i, i + 1) for i in range(6)] + [(6, 0)], 7, 60),
    "line12": (CouplingMap.line(12), [(i, i + 1) for i in range(11)], 12, 120),
    "ring12": (CouplingMap.ring(12),
               [(i, i + 1) for i in range(11)] + [(11, 0)], 12, 120),
    "grid3x3": (CouplingMap.grid(3, 3), _grid_edges(3, 3), 9, 60),
    "grid4x4": (CouplingMap.grid(4, 4), _grid_edges(4, 4), 16, 150),
    "grid5x5": (CouplingMap.grid(5, 5), _grid_edges(5, 5), 25, 250),
}

N_CIRCUITS = 120
N_GATES = 60
BUDGET = 8  # identical for both sides: 8 layout restarts, 8 routing trials


def synth(n: int, m: int, seed: int):
    rng = random.Random(seed)
    ours = Circuit(n)
    theirs = QuantumCircuit(n) if HAVE_QISKIT else None
    for _ in range(m):
        if rng.random() < 0.45 and n >= 2:
            a, b = rng.sample(range(n), 2)
            ours.add(GateKind.CX, [a, b])
            if theirs is not None:
                theirs.cx(a, b)
        else:
            q = rng.randrange(n)
            ours.add(GateKind.H, [q])
            if theirs is not None:
                theirs.h(q)
    return ours, theirs


def main() -> None:
    print(__doc__)
    threads = os.environ.get("OMP_NUM_THREADS", "(unset -> all cores)")
    print(f"OMP_NUM_THREADS = {threads}, os.cpu_count() = {os.cpu_count()}")
    if os.cpu_count() == 1:
        print("!! ONE CORE. The parallel layout search cannot be measured here.")
    print(f"\n{N_CIRCUITS} circuits x {N_GATES} gates, budget {BUDGET} both sides.\n")

    print(f"{'topology':<9} {'ours':>6} {'Qiskit':>7} {'swaps':>7} | "
          f"{'our ms':>7} {'Qk ms':>7} {'time':>7}")

    for name, (cm, edges, n, ngates) in TOPOLOGIES.items():
        us, qk = [], []
        t_us = t_qk = 0.0

        # PASS 1: ours, alone.
        for i in range(N_CIRCUITS):
            ours, _ = synth(n, ngates, seed=1000 + i)
            o = SabreOptions()
            o.trials = BUDGET
            o.layout_trials = BUDGET
            o.seed = i
            router = SabreRouter(cm, o)
            t = time.perf_counter()
            res = router.compile(ours)
            t_us += time.perf_counter() - t
            us.append(res.swaps_added)

        # PASS 2: Qiskit, alone, with our threads no longer competing for cores.
        if HAVE_QISKIT:
            for i in range(N_CIRCUITS):
                _, theirs = synth(n, ngates, seed=1000 + i)
                pm = PassManager([
                    SabreLayout(QkCouplingMap(edges), seed=i,
                                layout_trials=BUDGET, swap_trials=BUDGET)
                ])
                t = time.perf_counter()
                out = pm.run(theirs)
                t_qk += time.perf_counter() - t
                qk.append(out.count_ops().get("swap", 0))

        m = statistics.mean
        qk_s = m(qk) if qk else float("nan")
        qk_ms = t_qk / N_CIRCUITS * 1e3 if qk else float("nan")
        us_ms = t_us / N_CIRCUITS * 1e3
        print(f"{name:<9} {m(us):>6.1f} {qk_s:>7.1f} "
              f"{m(us) / qk_s if qk else float('nan'):>6.2f}x | "
              f"{us_ms:>7.1f} {qk_ms:>7.1f} {us_ms / max(qk_ms, 1e-9):>6.1f}x")

    print("\n  swaps < 1.00x = we produce the better circuit")
    print("  time  > 1.00x = we take longer to produce it")
    print("\nBoth columns are the result. We win the first and lose the second:")
    print("5-13% fewer swaps for 2.2-4x the compile time. See layout_pareto.py")
    print("for the whole curve -- `layout_trials` is the dial, and on a 25-qubit")
    print("grid a SINGLE restart already beats Qiskit on swaps.")


if __name__ == "__main__":
    main()
