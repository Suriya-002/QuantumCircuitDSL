"""Cross-validate the router against Qiskit.

Two questions, and they are different:

  1. Is the routed circuit CORRECT? Routing permutes the qubits, so the routed
     state vector is the original one with its amplitude indices moved around by
     `final_layout`. The check has to undo that permutation -- and it is checked
     against QISKIT's state for the original circuit, not our own, so a bug that
     happened to be shared by our simulator and our router could not hide.

  2. Is the routed circuit any GOOD? Correct routing is easy: swap everything to
     qubit 0 and back. The measure is how many SWAPs it costs, and the honest
     way to know is to run Qiskit's transpiler on the same circuit and the same
     device and compare. We lose, and by how much is recorded below.
"""

import random

import numpy as np
import pytest

import qcdsl
from qcdsl import Circuit, CouplingMap, GateKind, SabreOptions, SabreRouter

qiskit = pytest.importorskip("qiskit", reason="qiskit not installed")

from qiskit import QuantumCircuit  # noqa: E402
from qiskit.quantum_info import Statevector as QiskitStatevector  # noqa: E402
from qiskit.transpiler import CouplingMap as QiskitCouplingMap  # noqa: E402
from qiskit.transpiler import PassManager  # noqa: E402
from qiskit.transpiler.passes import SabreLayout  # noqa: E402

TOPOLOGIES = {
    "line7": (CouplingMap.line(7), [(i, i + 1) for i in range(6)], 7),
    "ring7": (CouplingMap.ring(7), [(i, i + 1) for i in range(6)] + [(6, 0)], 7),
    "grid2x4": (
        CouplingMap.grid(2, 4),
        [(0, 1), (0, 4), (1, 2), (1, 5), (2, 3), (2, 6), (3, 7), (4, 5), (5, 6), (6, 7)],
        8,
    ),
}


def twin_circuits(num_qubits: int, num_gates: int, seed: int):
    """The same circuit, built for both toolchains."""
    rng = random.Random(seed)
    ours = Circuit(num_qubits)
    theirs = QuantumCircuit(num_qubits)
    for _ in range(num_gates):
        if rng.random() < 0.45:
            a, b = rng.sample(range(num_qubits), 2)
            ours.add(GateKind.CX, [a, b])
            theirs.cx(a, b)
        else:
            roll = rng.random()
            q = rng.randrange(num_qubits)
            if roll < 0.4:
                ours.add(GateKind.H, [q])
                theirs.h(q)
            elif roll < 0.7:
                ours.add(GateKind.T, [q])
                theirs.t(q)
            else:
                angle = rng.uniform(-3, 3)
                ours.add(GateKind.RZ, [q], angle)
                theirs.rz(angle, q)
    return ours, theirs


def widened(qc: Circuit, num_qubits: int) -> Circuit:
    out = Circuit(num_qubits)
    for g in qc.gates():
        out.add(g.kind, list(g.qubits), g.param)
    return out


def qiskit_amps(qc: QuantumCircuit) -> np.ndarray:
    return np.asarray(QiskitStatevector(qc).data, dtype=complex)


# --------------------------------------------------------------------------
# 1. Correctness
# --------------------------------------------------------------------------


@pytest.mark.parametrize("topology", list(TOPOLOGIES))
@pytest.mark.parametrize("seed", range(6))
def test_the_routed_circuit_computes_the_original_state(topology, seed):
    device, _edges, n = TOPOLOGIES[topology]
    router = SabreRouter(device)

    ours, theirs = twin_circuits(n, 40, 500 + seed)
    result = router.compile(ours)

    # Qiskit's state for the ORIGINAL circuit, padded to the device width.
    padded = QuantumCircuit(device.num_qubits)
    padded.compose(theirs, qubits=range(n), inplace=True)
    want = qiskit_amps(padded)

    got = np.array(qcdsl.simulate(result.circuit).amplitudes(), dtype=complex)

    # Routing moved the qubits. Undo it.
    for i in range(len(want)):
        j = qcdsl.permute_index(i, result.final_layout)
        assert abs(want[i] - got[j]) < 1e-9, f"amplitude {i} -> {j}"


def test_the_permutation_is_actually_needed():
    """If the unpermuted comparison also passed, the test above proves nothing."""
    device = CouplingMap.line(5)
    router = SabreRouter(device)

    qc = Circuit(5)
    qc.add(GateKind.H, [0])
    qc.add(GateKind.CX, [0, 4])  # distance 4 on a line: swaps are unavoidable

    result = router.route(qc)  # trivial layout, so it MUST swap
    assert result.swaps_added > 0

    want = np.array(qcdsl.simulate(widened(qc, 5)).amplitudes(), dtype=complex)
    got = np.array(qcdsl.simulate(result.circuit).amplitudes(), dtype=complex)
    assert not np.allclose(want, got, atol=1e-9), "nothing was permuted"

    reordered = np.array(
        [got[qcdsl.permute_index(i, result.final_layout)] for i in range(len(want))]
    )
    assert np.allclose(want, reordered, atol=1e-9)


# --------------------------------------------------------------------------
# 2. Legality -- judged by Qiskit's own coupling map, not ours
# --------------------------------------------------------------------------


@pytest.mark.parametrize("topology", list(TOPOLOGIES))
def test_every_two_qubit_gate_lands_on_a_coupled_pair(topology):
    device, edges, n = TOPOLOGIES[topology]
    router = SabreRouter(device)
    legal = set()
    for a, b in edges:
        legal.add((a, b))
        legal.add((b, a))

    for seed in range(8):
        ours, _ = twin_circuits(n, 50, 700 + seed)
        result = router.compile(ours)
        for g in result.circuit.gates():
            if len(g.qubits) == 2:
                assert tuple(g.qubits) in legal, f"{g.kind} on {tuple(g.qubits)}"


# --------------------------------------------------------------------------
# 3. Quality -- how far behind Qiskit's transpiler are we?
# --------------------------------------------------------------------------


def swap_counts(topology: str, trials: int, circuits: int = 20):
    device, edges, n = TOPOLOGIES[topology]
    opt = SabreOptions()
    opt.trials = trials
    opt.seed = 0
    router = SabreRouter(device, opt)
    qcm = QiskitCouplingMap([list(e) for e in edges])

    ours_total = theirs_total = 0
    for seed in range(circuits):
        ours, theirs = twin_circuits(n, 60, 1000 + seed)
        ours_total += router.compile(ours).swaps_added
        routed = PassManager(
            [SabreLayout(qcm, seed=0, swap_trials=trials, layout_trials=trials)]
        ).run(theirs)
        theirs_total += routed.count_ops().get("swap", 0)
    return ours_total, theirs_total


@pytest.mark.parametrize("topology", list(TOPOLOGIES))
def test_swap_count_stays_within_reach_of_qiskit(topology):
    """A regression bound, not a victory lap.

    Qiskit's SABRE is tuned Rust with years behind it; ours is a header-only
    reimplementation of the paper. Being within a modest factor is the goal.
    Measured at 1.05x-1.27x; the bound is set at 1.6x so that a real regression
    in the heuristic trips it while ordinary noise does not.
    """
    ours, theirs = swap_counts(topology, trials=1)
    assert theirs > 0
    assert ours <= 1.6 * theirs, f"{topology}: qcdsl {ours} vs qiskit {theirs}"


def test_routing_a_device_with_no_constraints_inserts_nothing():
    router = SabreRouter(CouplingMap.all_to_all(6))
    for seed in range(6):
        ours, _ = twin_circuits(6, 40, seed)
        assert router.compile(ours).swaps_added == 0


def test_layout_search_beats_the_identity_layout():
    device, _edges, n = TOPOLOGIES["line7"]
    router = SabreRouter(device)
    trivial = sabre = 0
    for seed in range(20):
        ours, _ = twin_circuits(n, 60, 2000 + seed)
        trivial += router.route(ours).swaps_added
        sabre += router.compile(ours).swaps_added
    assert sabre < trivial, f"trivial {trivial}, sabre {sabre}"
