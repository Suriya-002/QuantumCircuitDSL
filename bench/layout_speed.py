"""Fewer SWAPs -- but at what cost in compile time?

    python bench/layout_speed.py

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

TOPOLOGIES = {
    "line7": (CouplingMap.line(7), [(i, i + 1) for i in range(6)], 7),
    "ring7": (CouplingMap.ring(7), [(i, i + 1) for i in range(6)] + [(6, 0)], 7),
    "grid2x4": (
        CouplingMap.grid(2, 4),
        [(0, 1), (0, 4), (1, 2), (1, 5), (2, 3), (2, 6), (3, 7),
         (4, 5), (5, 6), (6, 7)],
        8,
    ),
    "grid3x3": (
        CouplingMap.grid(3, 3),
        [(0, 1), (1, 2), (3, 4), (4, 5), (6, 7), (7, 8),
         (0, 3), (3, 6), (1, 4), (4, 7), (2, 5), (5, 8)],
        9,
    ),
}

N_CIRCUITS = 300
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

    for name, (cm, edges, n) in TOPOLOGIES.items():
        us, qk = [], []
        t_us = t_qk = 0.0
        for i in range(N_CIRCUITS):
            ours, theirs = synth(n, N_GATES, seed=1000 + i)

            o = SabreOptions()
            o.trials = BUDGET
            o.layout_trials = BUDGET
            o.seed = i
            router = SabreRouter(cm, o)

            t = time.perf_counter()
            res = router.compile(ours)
            t_us += time.perf_counter() - t
            us.append(res.swaps_added)

            if HAVE_QISKIT:
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
    print("\nBoth columns are the result. Reporting only the first one would be")
    print("a way of losing an argument you had not noticed you were having.")


if __name__ == "__main__":
    main()
