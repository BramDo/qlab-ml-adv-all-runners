from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from qiskit.quantum_info import SparsePauliOp, Statevector

import qiskit_qos_pbmc68k_q60_shallow_fireopal_validate as runner
import qiskit_qos_toy_model as toy


def _full_expectation(
    values: np.ndarray, mapping: dict[int, str]
) -> float:
    circuit = toy.query_circuit(
        values,
        single_scale=runner.SINGLE_SCALE,
        phase_scale=runner.PHASE_SCALE,
        pair_scale=runner.PAIR_SCALE,
    )
    label = ["I"] * len(values)
    for qubit, pauli in mapping.items():
        label[len(values) - 1 - qubit] = pauli
    state = Statevector.from_instruction(circuit)
    value = state.expectation_value(SparsePauliOp("".join(label)))
    return float(np.real_if_close(value).real)


def test_even_odd_schedule_is_unitary_equivalent_and_shallow() -> None:
    result = runner.verify_even_odd_equivalence()
    values = np.linspace(-0.8, 0.8, 60)
    linear, pair = runner.query_parameters(values)
    circuit = runner.shallow_circuit(linear, pair)

    assert result["passed"] is True
    assert result["statevector_fidelity"] == pytest.approx(1.0, abs=1e-12)
    assert circuit.depth() == 5
    assert circuit.count_ops()["rzz"] == 59


@pytest.mark.parametrize(
    "mapping",
    [
        {0: "X"},
        {2: "Y"},
        {5: "Z"},
        {1: "X", 2: "X"},
        {3: "Z", 4: "Z"},
    ],
)
def test_local_causal_cone_matches_full_six_qubit_statevector(
    mapping: dict[int, str],
) -> None:
    values = np.asarray([-0.7, 0.3, 0.8, -0.2, 0.5, -0.4])
    linear, pair = runner.query_parameters(values)

    local = runner.exact_local_expectation(linear, pair, mapping)
    full = _full_expectation(values, mapping)

    assert local == pytest.approx(full, abs=1e-12)


def test_seed_batch_uses_three_global_bases_and_train_only_selection() -> None:
    encoded_train = np.asarray(
        [[0.2, -0.1, 0.3, 0.4], [-0.3, 0.5, 0.1, -0.2]],
        dtype=np.float64,
    )
    encoded_test = np.asarray(
        [[0.1, 0.2, -0.4, 0.3], [0.4, -0.2, 0.2, -0.1]],
        dtype=np.float64,
    )
    qasms, manifest, metadata = runner.build_seed_circuits(
        encoded_train=encoded_train,
        encoded_test=encoded_test,
        y_train=np.asarray([-1.0, 1.0]),
        y_test=np.asarray([1.0, -1.0]),
        train_indices=[10, 11],
        test_indices=[20, 21],
        seed=11,
        hash_seed=11,
        selected_feature_count=4,
        shot_intent=1024,
    )

    assert len(qasms) == 15
    assert metadata["logical_base_circuit_count"] == 5
    assert metadata["candidate_observable_count"] == 18
    assert metadata["selected_observable_count"] == 4
    assert metadata["selection_protocol"]["uses_training_labels_only"] is True
    assert metadata["selection_protocol"]["uses_test_labels"] is False
    assert [row["measurement_basis"] for row in manifest[:3]] == ["X", "Y", "Z"]
    assert manifest[3]["role"] == "query"
    assert manifest[3]["split"] == "train"
    assert manifest[3]["source_row_index"] == 10
    assert [row["circuit_index"] for row in manifest] == list(range(15))


def test_60q_global_basis_qasm_is_numeric_virtual_and_shallow() -> None:
    values = np.linspace(-0.8, 0.8, 60)
    linear, pair = runner.query_parameters(values)
    measured = runner.measurement_circuit_for_basis(
        runner.shallow_circuit(linear, pair), "Y"
    )
    qasm, metadata = runner.q40_validate.export_numeric_qasm2(measured)

    assert qasm.startswith("OPENQASM 2.0;")
    assert "pi" not in qasm
    assert metadata["metrics"]["num_qubits"] == 60
    assert metadata["metrics"]["num_clbits"] == 60
    assert metadata["metrics"]["depth"] <= runner.DEFAULT_MAX_PAYLOAD_DEPTH
    assert metadata["round_trip_validated"] is True
    assert metadata["virtual_qubits_only"] is True


def test_source_has_no_fire_opal_execution_call() -> None:
    source = Path(runner.__file__).read_text(encoding="utf-8")
    assert "fireopal.execute" not in source
    assert "validate_fireopal_batch" in source
