"""Cross-validate the qcdsl state-vector backend against Qiskit.

The C++ test suite proves the simulator is self-consistent. It cannot prove the
simulator is *correct* -- the same person wrote the code and the expectations.
This module closes that gap: it drives randomly generated circuits through both
qcdsl and qiskit.quantum_info.Statevector and demands agreement amplitude for
amplitude, including global phase.

Once this passes, the simulator is an oracle. Every compilation pass added later
is validated the same way: simulate before the pass, simulate after, assert the
vectors are identical. A pass that changes the state is a broken pass.
"""

import math
import random

import numpy as np
import pytest

import qcdsl
from qcdsl import Circuit, GateKind

qiskit = pytest.importorskip("qiskit", reason="qiskit not installed")
from qiskit import QuantumCircuit  # noqa: E402
from qiskit.quantum_info import Statevector as QiskitStatevector  # noqa: E402

# Every gate qcdsl can simulate, paired with the qiskit method that means the
# same thing. MEASURE is excluded -- the state-vector backend rejects it.
ONE_QUBIT = [
    GateKind.I,
    GateKind.X,
    GateKind.Y,
    GateKind.Z,
    GateKind.H,
    GateKind.S,
    GateKind.Sdg,
    GateKind.T,
    GateKind.Tdg,
]
ROTATIONS = [GateKind.RX, GateKind.RY, GateKind.RZ]
TWO_QUBIT = [GateKind.CX, GateKind.CZ, GateKind.SWAP]

QISKIT_METHOD = {
    GateKind.I: "id",
    GateKind.X: "x",
    GateKind.Y: "y",
    GateKind.Z: "z",
    GateKind.H: "h",
    GateKind.S: "s",
    GateKind.Sdg: "sdg",
    GateKind.T: "t",
    GateKind.Tdg: "tdg",
    GateKind.RX: "rx",
    GateKind.RY: "ry",
    GateKind.RZ: "rz",
    GateKind.CX: "cx",
    GateKind.CZ: "cz",
    GateKind.SWAP: "swap",
}


def to_qiskit(qc: Circuit) -> QuantumCircuit:
    """Transcribe a qcdsl circuit into the equivalent qiskit circuit."""
    out = QuantumCircuit(qc.num_qubits)
    for g in qc.gates():
        method = getattr(out, QISKIT_METHOD[g.kind])
        if qcdsl.is_parametric(g.kind):
            method(g.param, *g.qubits)
        else:
            method(*g.qubits)
    return out


def amplitudes(qc: Circuit) -> np.ndarray:
    return np.array(qcdsl.simulate(qc).amplitudes(), dtype=complex)


def qiskit_amplitudes(qc: Circuit) -> np.ndarray:
    return np.asarray(QiskitStatevector(to_qiskit(qc)).data, dtype=complex)


def assert_agrees(qc: Circuit, atol: float = 1e-10) -> None:
    mine = amplitudes(qc)
    theirs = qiskit_amplitudes(qc)
    assert mine.shape == theirs.shape
    if not np.allclose(mine, theirs, atol=atol):
        worst = int(np.argmax(np.abs(mine - theirs)))
        raise AssertionError(
            f"disagreement at index {worst} (binary {worst:0{qc.num_qubits}b}): "
            f"qcdsl={mine[worst]!r} qiskit={theirs[worst]!r}\n"
            f"circuit: {[(qcdsl.gate_name(g.kind), list(g.qubits), g.param) for g in qc.gates()]}"
        )


def random_circuit(num_qubits: int, num_gates: int, rng: random.Random) -> Circuit:
    qc = Circuit(num_qubits)
    for _ in range(num_gates):
        if num_qubits >= 2 and rng.random() < 0.35:
            kind = rng.choice(TWO_QUBIT)
            a, b = rng.sample(range(num_qubits), 2)
            qc.add(kind, [a, b])
        elif rng.random() < 0.4:
            kind = rng.choice(ROTATIONS)
            qc.add(kind, [rng.randrange(num_qubits)], rng.uniform(-2 * math.pi, 2 * math.pi))
        else:
            kind = rng.choice(ONE_QUBIT)
            qc.add(kind, [rng.randrange(num_qubits)])
    return qc


# --------------------------------------------------------------------------
# Known states, checked against qiskit rather than against hand-typed numbers.
# --------------------------------------------------------------------------


def test_bell_pair_matches_qiskit():
    qc = Circuit(2)
    qc.add(GateKind.H, [0])
    qc.add(GateKind.CX, [0, 1])
    assert_agrees(qc)


def test_ghz_matches_qiskit():
    qc = Circuit(4)
    qc.add(GateKind.H, [0])
    for q in range(3):
        qc.add(GateKind.CX, [q, q + 1])
    assert_agrees(qc)


# --------------------------------------------------------------------------
# Every gate, in isolation, on every wire of a 3-qubit register -- run on a
# non-trivial input state so a wrong phase or a transposed matrix cannot hide.
# --------------------------------------------------------------------------


@pytest.mark.parametrize("kind", ONE_QUBIT + ROTATIONS)
@pytest.mark.parametrize("target", [0, 1, 2])
def test_single_qubit_gate_matches_qiskit(kind, target):
    qc = Circuit(3)
    qc.add(GateKind.H, [0])
    qc.add(GateKind.T, [1])
    qc.add(GateKind.RY, [2], 0.9)
    qc.add(kind, [target], 0.6180339887)
    assert_agrees(qc)


@pytest.mark.parametrize("kind", TWO_QUBIT)
@pytest.mark.parametrize("pair", [(0, 1), (1, 0), (0, 2), (2, 0), (1, 2)])
def test_two_qubit_gate_matches_qiskit(kind, pair):
    qc = Circuit(3)
    qc.add(GateKind.H, [0])
    qc.add(GateKind.RX, [1], 1.3)
    qc.add(GateKind.T, [2])
    qc.add(kind, list(pair))
    assert_agrees(qc)


# --------------------------------------------------------------------------
# The real test: random circuits. Seeded, so a failure is reproducible.
# --------------------------------------------------------------------------


@pytest.mark.parametrize("seed", range(12))
@pytest.mark.parametrize("num_qubits", [1, 2, 3, 5])
def test_random_circuits_match_qiskit(seed, num_qubits):
    rng = random.Random(1000 * num_qubits + seed)
    qc = random_circuit(num_qubits, num_gates=30, rng=rng)
    assert_agrees(qc)


def test_a_deep_wide_random_circuit_matches_qiskit():
    rng = random.Random(20260712)
    qc = random_circuit(7, num_gates=250, rng=rng)
    assert qc.depth() > 30
    assert_agrees(qc, atol=1e-9)


def test_norm_survives_a_long_circuit():
    rng = random.Random(7)
    qc = random_circuit(6, num_gates=400, rng=rng)
    assert qcdsl.simulate(qc).norm() == pytest.approx(1.0, abs=1e-9)
