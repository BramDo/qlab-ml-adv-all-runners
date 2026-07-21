from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

import qiskit_qos_pbmc68k_q40_fireopal_validate as runner
import qiskit_qos_toy_model as toy


def test_expectation_accepts_integer_counts_and_float_probabilities() -> None:
    integer_counts = {"00": 40, "01": 30, "10": 20, "11": 10}
    probabilities = {"00": 0.4, "01": 0.3, "10": 0.2, "11": 0.1}

    integer_value = toy.expectation_from_counts(
        integer_counts, mapping={0: "Z"}, num_qubits=2
    )
    probability_value = toy.expectation_from_counts(
        probabilities, mapping={0: "Z"}, num_qubits=2
    )

    assert integer_value == pytest.approx(0.2)
    assert probability_value == pytest.approx(integer_value)


def test_expectation_rejects_invalid_probability_weights() -> None:
    with pytest.raises(ValueError, match="finite and non-negative"):
        toy.expectation_from_counts(
            {"0": 1.1, "1": -0.1}, mapping={0: "Z"}, num_qubits=1
        )


def test_seed_batch_preserves_base_then_mapping_order() -> None:
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
        feature_mapping_limit=2,
    )

    assert len(qasms) == 10
    assert metadata["logical_base_circuit_count"] == 5
    assert metadata["measured_circuit_count"] == 10
    assert manifest[0]["role"] == "weighted_training_sketch"
    assert manifest[0]["pauli_mapping"] == [{"qubit": 0, "pauli": "X"}]
    assert manifest[1]["pauli_mapping"] == [{"qubit": 0, "pauli": "Y"}]
    assert manifest[2]["role"] == "query"
    assert manifest[2]["split"] == "train"
    assert manifest[2]["source_row_index"] == 10
    assert [row["circuit_index"] for row in manifest] == list(range(10))


def test_numeric_qasm_round_trip_keeps_40q_measurement_mapping() -> None:
    circuit = toy.query_circuit(
        np.linspace(-0.8, 0.8, 40),
        single_scale=1.35,
        phase_scale=0.75,
        pair_scale=0.95,
    )
    measured = toy.measurement_circuit_for_mapping(circuit, {0: "X"})
    qasm, metadata = runner.export_numeric_qasm2(measured)

    assert qasm.startswith("OPENQASM 2.0;")
    assert "pi" not in qasm
    assert metadata["metrics"]["num_qubits"] == 40
    assert metadata["metrics"]["num_clbits"] == 40
    assert metadata["round_trip_validated"] is True
    assert metadata["measurement_mapping"] == [
        {"qubit": index, "clbit": index} for index in range(40)
    ]


class _FakeValidationJob:
    action_id = "validate-123"

    def result(self) -> dict[str, object]:
        return {"results": [], "warnings": ["test warning"]}


class _FakeFireOpal:
    def __init__(self) -> None:
        self.calls: list[str] = []

    def show_supported_devices(self, *, credentials: object) -> dict[str, object]:
        self.calls.append("show_supported_devices")
        return {"supported_devices": ["ibm_fez"]}

    def validate(
        self,
        *,
        circuits: list[str],
        credentials: object,
        backend_name: str,
    ) -> _FakeValidationJob:
        self.calls.append("validate")
        assert circuits == ["qasm-a", "qasm-b"]
        assert backend_name == "ibm_fez"
        return _FakeValidationJob()


def test_provider_route_calls_discovery_and_validate_only() -> None:
    fake = _FakeFireOpal()
    with patch.object(
        runner,
        "_fire_opal_credentials_from_source",
        return_value=(fake, object(), {"token_source": "test"}, "test"),
    ):
        result = runner.validate_fireopal_batch(
            ["qasm-a", "qasm-b"],
            backend="ibm_fez",
            qiskit_account=None,
            qctrl_notebook=None,
            instance=None,
        )

    assert fake.calls == ["show_supported_devices", "validate"]
    assert result["passed"] is True
    assert result["execution_attempted"] is False
    assert result["quantum_seconds_used"] == 0
    assert result["validation_action_id"] == "validate-123"
    assert result["warnings"] == ["test warning"]
    source = Path(runner.__file__).read_text(encoding="utf-8")
    assert "fireopal.execute" not in source
