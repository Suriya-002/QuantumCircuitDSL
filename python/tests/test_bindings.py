"""Python-side tests. These exercise the pybind11 boundary, not the C++ logic
(which GoogleTest already covers) -- the point is that types, exceptions and
overloads cross the boundary correctly."""

import pytest

import qcdsl
from qcdsl import Circuit, Gate, GateKind


def test_version_is_exposed():
    assert qcdsl.__version__ == "0.1.0"


def test_arity_and_parametric_cross_the_boundary():
    assert qcdsl.arity(GateKind.CX) == 2
    assert qcdsl.arity(GateKind.H) == 1
    assert qcdsl.is_parametric(GateKind.RZ)
    assert not qcdsl.is_parametric(GateKind.T)


def test_bell_pair_size_and_depth():
    qc = Circuit(2)
    qc.add(GateKind.H, [0])
    qc.add(GateKind.CX, [0, 1])
    assert qc.num_qubits == 2
    assert qc.size() == 2
    assert len(qc) == 2
    assert qc.depth() == 2


def test_disjoint_gates_share_a_layer():
    qc = Circuit(3)
    for q in range(3):
        qc.add(GateKind.H, [q])
    assert qc.size() == 3
    assert qc.depth() == 1


def test_gate_object_accepted():
    qc = Circuit(2)
    qc.add(Gate(GateKind.RX, [0], 0.5))
    (g,) = qc.gates()
    assert g.kind == GateKind.RX
    assert g.qubits == [0]
    assert g.param == pytest.approx(0.5)


def test_cpp_exceptions_become_python_exceptions():
    with pytest.raises(ValueError):
        Circuit(0)
    with pytest.raises(ValueError):
        Gate(GateKind.CX, [0])  # wrong arity
    with pytest.raises(IndexError):
        Circuit(2).add(GateKind.X, [7])  # out of range
