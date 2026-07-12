"""Cross-validate the compilation passes against Qiskit.

The C++ tests already assert that each pass leaves our own simulator's state
vector unchanged. That is a closed loop: our optimiser checked by our simulator.
Here the witness is external -- optimise the circuit with qcdsl, then compare the
result against the state QISKIT computes for the ORIGINAL circuit.

If a pass were unsound, and our simulator happened to be wrong in the matching
way, the C++ tests would still pass. This test would not.
"""

import random

import numpy as np
import pytest

import qcdsl
from qcdsl import (
    CancelInversePairs,
    Circuit,
    DecomposeToCx,
    GateKind,
    MergeRotations,
    PassManager,
    RemoveIdentities,
)

qiskit = pytest.importorskip("qiskit", reason="qiskit not installed")

from test_vs_qiskit import qiskit_amplitudes, random_circuit  # noqa: E402

ALL_PASSES = [
    CancelInversePairs,
    MergeRotations,
    RemoveIdentities,
    DecomposeToCx,
]


def amps(qc: Circuit) -> np.ndarray:
    return np.array(qcdsl.simulate(qc).amplitudes(), dtype=complex)


def redundant_circuit(num_qubits: int, num_gates: int, rng: random.Random) -> Circuit:
    """A circuit with repeated gates -- what unoptimised output actually looks
    like, and the only kind on which an optimiser has anything to do."""
    base = random_circuit(num_qubits, num_gates // 2, rng)
    qc = Circuit(num_qubits)
    for g in base.gates():
        qc.add(g.kind, list(g.qubits), g.param)
        if rng.random() < 0.4:  # emit it twice
            qc.add(g.kind, list(g.qubits), g.param)
    return qc


def full_pipeline() -> PassManager:
    pm = PassManager()
    pm.add(MergeRotations())
    pm.add(RemoveIdentities())
    pm.add(CancelInversePairs())
    pm.add(DecomposeToCx())
    pm.add(MergeRotations())
    pm.add(RemoveIdentities())
    pm.add(CancelInversePairs())
    return pm


@pytest.mark.parametrize("pass_type", ALL_PASSES)
@pytest.mark.parametrize("seed", range(8))
def test_each_pass_preserves_the_qiskit_state(pass_type, seed):
    rng = random.Random(4000 + seed)
    qc = redundant_circuit(4, 40, rng)
    optimised = pass_type()(qc)
    assert np.allclose(amps(optimised), qiskit_amplitudes(qc), atol=1e-10)


@pytest.mark.parametrize("seed", range(12))
def test_the_full_pipeline_preserves_the_qiskit_state(seed):
    rng = random.Random(6000 + seed)
    qc = redundant_circuit(5, 60, rng)
    optimised = full_pipeline().run_to_fixed_point(qc)
    assert np.allclose(amps(optimised), qiskit_amplitudes(qc), atol=1e-9)


def test_the_pipeline_lowers_to_a_cx_only_gate_set():
    rng = random.Random(77)
    for _ in range(20):
        qc = redundant_circuit(5, 60, rng)
        out = full_pipeline().run_to_fixed_point(qc)
        for g in out.gates():
            assert g.kind not in (GateKind.CZ, GateKind.SWAP)


def test_the_pipeline_actually_removes_gates():
    rng = random.Random(2026)
    before = after = 0
    for _ in range(30):
        qc = redundant_circuit(6, 120, rng)
        before += qc.size()
        after += full_pipeline().run_to_fixed_point(qc).size()
    assert after < before, "the optimiser removed nothing"


def test_stats_are_reported_per_pass():
    rng = random.Random(5)
    qc = redundant_circuit(4, 40, rng)
    pm = full_pipeline()
    pm.run(qc)
    stats = pm.stats()
    assert len(stats) == 7
    assert stats[0].pass_name == "merge-rotations"
    assert stats[0].gates_before == qc.size()
    assert any(s.changed() for s in stats)


def test_a_hand_built_cancellation_collapses_completely():
    qc = Circuit(2)
    for kind, qubits in [
        (GateKind.H, [0]),
        (GateKind.H, [0]),
        (GateKind.CX, [0, 1]),
        (GateKind.CX, [0, 1]),
        (GateKind.T, [1]),
        (GateKind.Tdg, [1]),
    ]:
        qc.add(kind, qubits)
    out = full_pipeline().run_to_fixed_point(qc)
    assert out.size() == 0
    assert np.allclose(amps(out), qiskit_amplitudes(qc), atol=1e-12)
