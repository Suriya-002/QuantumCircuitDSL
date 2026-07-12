"""Two-way OpenQASM 3 interop with Qiskit.

Round-tripping through our own parser proves nothing: an emitter and a parser
that agree on the same private dialect will round-trip perfectly and still be
unable to talk to anything else. Conformance means an OUTSIDE implementation
agrees, in both directions:

  export:  qcdsl emits  ->  Qiskit parses   ->  same state vector
  import:  Qiskit emits ->  qcdsl parses    ->  same state vector

Both are checked here on randomly generated circuits, amplitude for amplitude.
"""

import random

import numpy as np
import pytest

import qcdsl
from qcdsl import Circuit, GateKind

qiskit = pytest.importorskip("qiskit", reason="qiskit not installed")
pytest.importorskip("qiskit_qasm3_import", reason="qiskit_qasm3_import not installed")

from qiskit import qasm3  # noqa: E402
from qiskit.quantum_info import Statevector as QiskitStatevector  # noqa: E402

from test_vs_qiskit import qiskit_amplitudes, random_circuit, to_qiskit  # noqa: E402


def amps(qc: Circuit) -> np.ndarray:
    return np.array(qcdsl.simulate(qc).amplitudes(), dtype=complex)


# --------------------------------------------------------------------------
# export:  qcdsl -> QASM3 -> Qiskit
# --------------------------------------------------------------------------


@pytest.mark.parametrize("seed", range(12))
@pytest.mark.parametrize("num_qubits", [1, 2, 4])
def test_qiskit_can_parse_what_we_emit(seed, num_qubits):
    rng = random.Random(100 * num_qubits + seed)
    qc = random_circuit(num_qubits, num_gates=30, rng=rng)

    parsed_by_qiskit = qasm3.loads(qcdsl.to_qasm3(qc))
    theirs = np.asarray(QiskitStatevector(parsed_by_qiskit).data, dtype=complex)

    assert np.allclose(amps(qc), theirs, atol=1e-10)


# --------------------------------------------------------------------------
# import:  Qiskit -> QASM3 -> qcdsl
# --------------------------------------------------------------------------


@pytest.mark.parametrize("seed", range(12))
@pytest.mark.parametrize("num_qubits", [1, 2, 4])
def test_we_can_parse_what_qiskit_emits(seed, num_qubits):
    rng = random.Random(7000 + 100 * num_qubits + seed)
    original = random_circuit(num_qubits, num_gates=30, rng=rng)

    qasm_from_qiskit = qasm3.dumps(to_qiskit(original))
    reparsed = qcdsl.from_qasm3(qasm_from_qiskit)

    assert reparsed.num_qubits == num_qubits
    assert reparsed.size() == original.size()
    assert np.allclose(amps(reparsed), qiskit_amplitudes(original), atol=1e-10)


def test_we_parse_qiskits_pi_expressions():
    """Qiskit does not always emit a bare float."""
    qc = qcdsl.from_qasm3(
        """
        OPENQASM 3.0;
        include "stdgates.inc";
        qubit[1] q;
        rz(pi/2) q[0];
        rx(-pi/4) q[0];
        ry(2*pi) q[0];
        """
    )
    assert qc.size() == 3
    assert qc.gates()[0].param == pytest.approx(np.pi / 2)
    assert qc.gates()[1].param == pytest.approx(-np.pi / 4)
    assert qc.gates()[2].param == pytest.approx(2 * np.pi)


# --------------------------------------------------------------------------
# The full loop: compile, export, and hand the result to Qiskit.
# --------------------------------------------------------------------------


def test_an_optimised_circuit_still_round_trips_through_qiskit():
    from qcdsl import (
        CancelInversePairs,
        DecomposeToCx,
        MergeRotations,
        PassManager,
        RemoveIdentities,
    )

    rng = random.Random(20260712)
    for _ in range(10):
        qc = random_circuit(4, num_gates=40, rng=rng)

        pm = PassManager()
        for p in (MergeRotations(), RemoveIdentities(), CancelInversePairs(),
                  DecomposeToCx(), MergeRotations(), RemoveIdentities(),
                  CancelInversePairs()):
            pm.add(p)
        optimised = pm.run_to_fixed_point(qc)

        # Qiskit parses the compiler's output, and it still computes the state
        # of the circuit we started from.
        theirs = np.asarray(
            QiskitStatevector(qasm3.loads(qcdsl.to_qasm3(optimised))).data,
            dtype=complex,
        )
        assert np.allclose(theirs, qiskit_amplitudes(qc), atol=1e-9)


def test_measurement_survives_the_round_trip():
    qc = Circuit(2)
    qc.add(GateKind.H, [0])
    qc.add(GateKind.CX, [0, 1])
    qc.add(GateKind.MEASURE, [0])
    qc.add(GateKind.MEASURE, [1])

    src = qcdsl.to_qasm3(qc)
    assert "bit[2] c;" in src

    parsed = qasm3.loads(src)  # Qiskit accepts it
    assert parsed.num_clbits == 2

    back = qcdsl.from_qasm3(src)  # and so do we
    assert back.size() == 4
    assert back.gates()[2].kind == GateKind.MEASURE
