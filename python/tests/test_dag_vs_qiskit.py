"""Cross-validate the DAG IR against Qiskit.

Two independent claims are checked here, both against an outside implementation
rather than against our own expectations:

1. `Dag.depth()` -- the longest path through the dependency graph -- equals
   `QuantumCircuit.depth()` in Qiskit. Three algorithms now agree on the same
   number (our wire walk, our graph walk, Qiskit's), computed three different
   ways.

2. Any topological order of the DAG is a legal schedule, so rescheduling a
   circuit through the DAG must not change the state Qiskit computes for it.
   This is the property every later pass depends on: if a reordering is safe,
   the DAG must say so, and if it is unsafe, the DAG must forbid it.
"""

import random

import numpy as np
import pytest

import qcdsl
from qcdsl import Circuit, Dag, GateKind

qiskit = pytest.importorskip("qiskit", reason="qiskit not installed")

from test_vs_qiskit import (  # noqa: E402
    qiskit_amplitudes,
    random_circuit,
    to_qiskit,
)


def test_dag_of_an_empty_circuit_is_empty():
    d = Dag(Circuit(3))
    assert d.size() == 0
    assert d.num_edges() == 0
    assert d.depth() == 0


def test_independent_gates_are_not_ordered():
    qc = Circuit(2)
    qc.add(GateKind.H, [0])
    qc.add(GateKind.T, [1])
    d = Dag(qc)
    assert d.num_edges() == 0
    assert d.depth() == 1
    assert sorted(d.frontier()) == [0, 1]


@pytest.mark.parametrize("seed", range(15))
@pytest.mark.parametrize("num_qubits", [1, 2, 4, 6])
def test_dag_depth_matches_qiskit(seed, num_qubits):
    rng = random.Random(500 * num_qubits + seed)
    qc = random_circuit(num_qubits, num_gates=40, rng=rng)
    assert Dag(qc).depth() == to_qiskit(qc).depth()


@pytest.mark.parametrize("seed", range(10))
def test_rescheduling_through_the_dag_preserves_the_qiskit_state(seed):
    rng = random.Random(9000 + seed)
    qc = random_circuit(4, num_gates=35, rng=rng)
    reference = qiskit_amplitudes(qc)
    d = Dag(qc)

    reordered = 0
    canonical = d.topological_order()
    for shuffle in range(6):
        order = d.random_topological_order(100 * seed + shuffle)
        assert d.is_valid_schedule(order)
        if order != canonical:
            reordered += 1
        got = np.array(qcdsl.simulate(d.to_circuit(order)).amplitudes(), dtype=complex)
        assert np.allclose(got, reference, atol=1e-10)

    # If the schedules were all identical the assertion above would be vacuous.
    assert reordered > 0


def test_layers_hold_only_mutually_independent_gates():
    rng = random.Random(31337)
    qc = random_circuit(6, num_gates=80, rng=rng)
    d = Dag(qc)
    layers = d.layers()
    assert len(layers) == d.depth()
    assert sum(len(layer) for layer in layers) == d.size()

    for layer in layers:
        seen = set()
        for node_id in layer:
            wires = set(d.node(node_id).gate.qubits)
            assert not (wires & seen), "two gates in one layer share a wire"
            seen |= wires


def test_a_non_topological_order_is_rejected():
    qc = Circuit(1)
    qc.add(GateKind.H, [0])
    qc.add(GateKind.T, [0])
    d = Dag(qc)
    assert not d.is_valid_schedule([1, 0])
    with pytest.raises(ValueError):
        d.to_circuit([1, 0])
