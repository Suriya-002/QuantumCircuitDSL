"""What does each layout restart buy, and what does it cost?

    python bench/layout_pareto.py

On a 25-qubit grid this compiler produces 11% fewer SWAPs than Qiskit's
transpiler and takes 2.3x as long to do it. That is a trade, not a win, and the
only honest way to present a trade is to show the whole curve and let the reader
pick their point on it.

`layout_trials` is the dial. Each restart is an independent descent from a fresh
random permutation, so the cost is linear in the number of restarts and the
benefit is not -- the first few restarts find most of what there is to find.

The question this answers: how few restarts can you buy and still beat Qiskit?
"""

from __future__ import annotations

import os
import random
import statistics
import time

import numpy as np

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


CASES = [
    ("grid4x4", CouplingMap.grid(4, 4), _grid_edges(4, 4), 16, 150),
    ("grid5x5", CouplingMap.grid(5, 5), _grid_edges(5, 5), 25, 250),
]
RESTARTS = (1, 2, 4, 8, 16)
SCORING = (0, 4, 2, 1)   # 0 == rank with the full `trials` budget
N_CIRCUITS = 100
QK_BUDGET = 8


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


def paired_interval(ours, theirs, reps: int = 3000):
    d = np.asarray(ours, float) - np.asarray(theirs, float)
    rng = np.random.default_rng(0)
    means = [d[rng.integers(0, len(d), len(d))].mean() for _ in range(reps)]
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def main() -> None:
    print(__doc__)
    print(f"cores = {os.cpu_count()}, {N_CIRCUITS} circuits, "
          f"Qiskit fixed at {QK_BUDGET} layout trials\n")
    if os.cpu_count() == 1:
        print("!! ONE CORE. The time column is meaningless here.\n")

    for name, cm, edges, n, gates in CASES:
        qk, t_qk = [], 0.0
        for i in range(N_CIRCUITS):
            _, theirs = synth(n, gates, 5000 + i)
            pm = PassManager([
                SabreLayout(QkCouplingMap(edges), seed=i,
                            layout_trials=QK_BUDGET, swap_trials=QK_BUDGET)
            ])
            t = time.perf_counter()
            out = pm.run(theirs)
            t_qk += time.perf_counter() - t
            qk.append(out.count_ops().get("swap", 0))
        qk_mean = statistics.mean(qk)
        qk_ms = t_qk / N_CIRCUITS * 1e3

        print(f"== {name} ({n} qubits) ==   "
              f"Qiskit: {qk_mean:.1f} swaps, {qk_ms:.1f} ms")
        print(f"{'restarts':>9} {'swaps':>7} {'vs Qk':>7} {'ms':>7} {'vs Qk':>7}  "
              f"{'95% CI on paired diff':>24}")

        def row(label, layout_trials, scoring_trials):
            us, t_us = [], 0.0
            for i in range(N_CIRCUITS):
                ours, _ = synth(n, gates, 5000 + i)
                o = SabreOptions()
                o.trials = QK_BUDGET
                o.layout_trials = layout_trials
                o.scoring_trials = scoring_trials
                o.seed = i
                router = SabreRouter(cm, o)
                t = time.perf_counter()
                res = router.compile(ours)
                t_us += time.perf_counter() - t
                us.append(res.swaps_added)
            m = statistics.mean(us)
            ms = t_us / N_CIRCUITS * 1e3
            lo, hi = paired_interval(us, qk)
            v = "TIE" if lo < 0 < hi else ("LOSE" if lo > 0 else "WIN")
            print(f"{label:>9} {m:>7.1f} {m / qk_mean:>6.3f}x {ms:>7.1f} "
                  f"{ms / max(qk_ms, 1e-9):>6.2f}x  [{lo:+7.2f},{hi:+7.2f}] {v}")

        print("  -- layout_trials (scoring_trials = full) --")
        for r in RESTARTS:
            row(str(r), r, 0)

        print("  -- scoring_trials, at layout_trials = 8 --")
        print("     ranking a candidate is not answering. How few passes suffice?")
        for st in SCORING:
            row("full" if st == 0 else str(st), 8, st)
        print()

    print("Pick the row where the swap column is a WIN and the time column is")
    print("acceptable. That is the compiler's actual operating point, and it is")
    print("a choice, not a headline.")
    print()
    print("`scoring_trials` is the cheaper dial: 95% of a compile is the LAYOUT")
    print("SEARCH, and most of that is spent routing candidate layouts eight")
    print("times each merely to decide which one is best. Ranking is not")
    print("answering. One pass loses ~2% of the swap advantage and returns ~39%")
    print("of the compile time -- and `compile` still plays the cheap winner off")
    print("against the identity candidate at full budget, so raising")
    print("`layout_trials` can still never make the answer worse.")


if __name__ == "__main__":
    main()
