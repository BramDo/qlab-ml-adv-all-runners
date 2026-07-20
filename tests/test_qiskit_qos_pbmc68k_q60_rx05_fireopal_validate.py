from __future__ import annotations

import inspect

import numpy as np
from qiskit.quantum_info import SparsePauliOp, Statevector

import qiskit_qos_pbmc68k_q60_rx05_fireopal_validate as runner
import qiskit_qos_pbmc68k_q60_reuploading_tune as reupload
import qiskit_qos_toy_model as toy


def test_rx05_circuit_matches_reupload_causal_cone_expectations() -> None:
    linear = np.asarray([0.2, -0.4, 0.7, -0.1, 0.3, -0.6])
    pair = linear[:-1] * linear[1:]
    circuit = runner.rx05_circuit(
        linear,
        pair,
        single_scale=0.75,
        phase_scale=0.25,
        pair_scale=0.95,
    )
    state = Statevector.from_instruction(circuit)
    for mapping in ({2: "X"}, {2: "Y"}, {2: "X", 3: "X"}):
        expected = float(
            np.real(state.expectation_value(toy.operator(len(linear), mapping)))
        )
        actual = reupload.exact_local_reupload_expectation(
            linear,
            pair,
            mapping,
            single_scale=0.75,
            phase_scale=0.25,
            pair_scale=0.95,
            post_axis="rx",
            post_scale=0.5,
        )
        assert np.isclose(actual, expected, atol=1e-12, rtol=0.0)


def test_rx05_circuit_is_depth_six_before_measurement() -> None:
    linear = np.linspace(-0.6, 0.7, 60)
    pair = linear[:-1] * linear[1:]
    circuit = runner.rx05_circuit(
        linear,
        pair,
        single_scale=0.75,
        phase_scale=0.25,
        pair_scale=0.95,
    )
    assert circuit.num_qubits == 60
    assert circuit.depth() == 6
    assert circuit.count_ops()["rx"] == 60


def test_source_has_validate_only_provider_boundary() -> None:
    source = inspect.getsource(runner)
    assert "fireopal.execute" not in source
    assert "validate_fireopal_batch" in source
    assert '"execution_attempted": False' in source


def test_reuploading_structural_gate_uses_category_counts() -> None:
    source = inspect.getsource(runner.prepare_batch)
    assert 'selected_counts = seed_meta["selection_audit"]["selected_counts"]' in source
    assert 'selected_pair_sensitive_count"] ==' not in source


def test_probability_operator_import_is_numeric() -> None:
    operator = SparsePauliOp("X")
    assert np.isfinite(float(np.real(operator.coeffs[0])))
