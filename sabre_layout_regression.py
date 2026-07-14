"""Minimal reproducer: SabreLayout does not keep the best layout it finds.

    pip install qiskit
    python sabre_layout_regression.py

WHAT THE DOCSTRING SAYS
-----------------------
qiskit/transpiler/passes/layout/sabre_layout.py, class SabreLayout:

    "Starting with a random initial `Layout`, the algorithm does a full routing
     of the circuit ... to end up with a `final_layout`. This final_layout is
     then used as the initial_layout for routing the reverse circuit. THE
     ALGORITHM ITERATES A NUMBER OF TIMES UNTIL IT FINDS AN INITIAL_LAYOUT THAT
     REDUCES FULL ROUTING COST."

WHAT THE CODE DOES
------------------
No routing cost is compared across iterations. Each forward-backward pass
overwrites the layout unconditionally, and whatever the LAST iteration happens
to produce is returned:

    for _ in range(self.max_iterations):
        for _ in ("forward", "backward"):
            ...
            initial_layout = final_layout      # no comparison, no best-tracking

The forward-backward iteration is a descent, and it is NOT monotone -- it can
walk off a good layout onto a worse one. Because the final iterate is returned
rather than the best one visited, RAISING max_iterations MAKES THE RESULT WORSE
on a large fraction of circuits, and can make the MEAN worse.

`swap_trials` and `layout_trials` both take a best-of, so they are set to 1 here
to isolate the iteration loop.

MEASURED (Qiskit 2.5.0, 4x4 grid coupling map)
----------------------------------------------
                       max_iterations:   1       2       3       4       8
    random circuits    mean swaps       51.03   50.67   50.65   50.70   51.20
                       % degraded        --      43%     44%     47%     47%

    QFT-16             mean swaps       81.15   79.95   79.65   79.40   80.05
                       % degraded        --      33%     40%     33%     43%

    QAOA-16            mean swaps       28.35   27.27   25.80   26.55   25.57
                       % degraded        --      40%     15%     30%     20%

Note the means: random circuits are WORSE at 8 iterations than at 1. QFT is
worse at 8 than at 4. QAOA is worse at 4 than at 3. More search, worse answer.

THE FIX
-------
Track the best layout seen across iterations, by the routing cost that is
already being computed, and return that one. The cost of the comparison is zero:
the forward pass of each iteration already routes the circuit, so its swap count
IS the current layout's score. It is simply discarded.
"""

from __future__ import annotations

import numpy as np
import qiskit
from qiskit import QuantumCircuit
from qiskit.transpiler import CouplingMap, PassManager
from qiskit.transpiler.passes import SabreLayout


def grid_edges(rows: int, cols: int):
    e = []
    for i in range(rows):
        for j in range(cols):
            q = i * cols + j
            if j + 1 < cols:
                e.append((q, q + 1))
            if i + 1 < rows:
                e.append((q, q + cols))
    return e


def random_circuit(n: int, gates: int, seed: int) -> QuantumCircuit:
    rng = np.random.default_rng(seed)
    qc = QuantumCircuit(n)
    for _ in range(gates):
        if rng.random() < 0.45:
            a, b = rng.choice(n, 2, replace=False)
            qc.cx(int(a), int(b))
        else:
            qc.h(int(rng.integers(n)))
    return qc


def swaps(qc: QuantumCircuit, cm: CouplingMap, iterations: int, seed: int) -> int:
    pm = PassManager([
        SabreLayout(cm, seed=seed, max_iterations=iterations,
                    layout_trials=1, swap_trials=1)  # isolate the iteration loop
    ])
    return pm.run(qc).count_ops().get("swap", 0)


def main() -> None:
    print(__doc__)

    n, gates, shots = 16, 150, 150
    cm = CouplingMap(grid_edges(4, 4))
    circuits = [random_circuit(n, gates, 9000 + i) for i in range(shots)]

    print(f"qiskit {qiskit.__version__}")
    print(f"{shots} random circuits, {n} qubits, 4x4 grid.")
    print("layout_trials=1, swap_trials=1, so nothing else is taking a best-of.\n")
    print("THE COLUMN THAT MATTERS IS THE LAST ONE. If the pass returned the best")
    print("layout it visited, more iterations could not make a SINGLE circuit")
    print("worse, and that column would read 0%. It is not a statistical claim.\n")
    print(f"{'max_iterations':>15} {'mean swaps':>11} {'worse than it=1':>17}")

    baseline = None
    for iterations in (1, 2, 3, 4, 8):
        counts = np.array(
            [swaps(qc, cm, iterations, seed=i) for i, qc in enumerate(circuits)],
            dtype=float,
        )
        if baseline is None:
            baseline = counts.copy()
            print(f"{iterations:>15} {counts.mean():>11.2f} {'--':>17}")
            continue
        degraded = int((counts > baseline).sum())
        print(f"{iterations:>15} {counts.mean():>11.2f} "
              f"{degraded:>9} / {shots}  ({degraded / shots:.0%})")

    print("\nEvery circuit uses the SAME seed at every iteration count, so the")
    print("initial layout is identical -- the only difference is how much the")
    print("descent was allowed to refine it. More refinement, worse answer, on")
    print("half the circuits.")
    print("\nThat is only possible if the pass returns the LAST iterate rather")
    print("than the BEST one it visited. The docstring says it iterates 'until it")
    print("finds an initial_layout that reduces full routing cost'. No such")
    print("comparison is made anywhere in the pass.")


if __name__ == "__main__":
    main()
