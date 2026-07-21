from __future__ import annotations

import ast
import inspect
from pathlib import Path

import numpy as np
from qiskit.quantum_info import Statevector

import qiskit_official_qos_flat_fireopal_pilot as pilot
import qiskit_official_qos_sampling_port as port


def test_interference_readout_matches_jax_and_exact_controls() -> None:
    specs = pilot._instance_specs()[:2]

    for spec in specs:
        circuit, _ = port.build_flat_interference_circuit_from_samples(
            spec["sampled_indices"], spec["sampled_values"], pilot.DIMENSION
        )
        qiskit_state = np.asarray(Statevector.from_instruction(circuit).data)
        jax_state = port.flat_interference_state_from_jax(
            spec["sampled_indices"], spec["sampled_values"], pilot.DIMENSION
        )
        target = int(spec["expected_target"], 2)

        assert np.max(np.abs(qiskit_state - jax_state)) < 1e-12
        assert abs(qiskit_state[target]) ** 2 > 1.0 - 1e-12
        assert np.isclose(np.sum(np.abs(qiskit_state) ** 2), 1.0)


def test_frozen_instance_batch_is_reproducible_and_exactly_66() -> None:
    first = pilot._instance_specs()
    second = pilot._instance_specs()

    assert len(first) == pilot.CIRCUITS == 66
    assert [row["role"] for row in first[:2]] == [
        "control_identity",
        "control_linear_phase",
    ]
    assert sum(row["role"] == "random_flat_kernel" for row in first) == 64
    for left, right in zip(first, second, strict=True):
        assert left["instance_seed"] == right["instance_seed"]
        assert np.array_equal(left["vector"], right["vector"])
        assert np.array_equal(left["sampled_indices"], right["sampled_indices"])
        assert np.array_equal(left["sampled_values"], right["sampled_values"])


def test_prepare_writes_roundtrippable_verified_bundle(tmp_path: Path) -> None:
    bundle = tmp_path / "bundle.json.gz"
    report_path = tmp_path / "prepare.json"
    args = pilot.build_parser().parse_args(
        ["prepare", "--bundle", str(bundle), "--report", str(report_path)]
    )

    report = pilot.prepare_pilot(args)
    qasms, manifest, info = pilot.load_bundle(bundle)

    assert report["local_validation"]["passed"] is True
    assert report["execution_attempted"] is False
    assert report["provider_calls"] == []
    assert len(qasms) == len(manifest) == 66
    assert info["aggregate_qasm_sha256"] == report["bundle"]["aggregate_qasm_sha256"]
    assert report["local_validation"]["minimum_control_target_probability"] > 1.0 - 1e-12


def test_hardware_result_validation_preserves_probability_semantics() -> None:
    manifest = []
    distributions = []
    for index in range(pilot.CIRCUITS):
        if index == 0:
            role = "control_identity"
            target = "0000"
            ideal = {target: 1.0}
        elif index == 1:
            role = "control_linear_phase"
            target = "1011"
            ideal = {target: 1.0}
        else:
            role = "random_flat_kernel"
            target = None
            ideal = {format(value, "04b"): 1.0 / 16.0 for value in range(16)}
        manifest.append(
            {
                "circuit_index": index,
                "role": role,
                "expected_target": target,
                "ideal_probabilities": ideal,
                "qasm_sha256": f"hash-{index}",
            }
        )
        distributions.append(ideal)

    result = pilot.validate_hardware_result({"results": distributions}, manifest)

    assert result["distribution_validation"]["passed"] is True
    assert result["distribution_validation"]["semantics"] == ["probability"]
    assert result["random_kernel_summary"]["mean_hellinger_fidelity"] == 1.0
    assert result["random_kernel_summary"]["mean_total_variation_distance"] == 0.0
    assert all(row["hardware_target_probability"] == 1.0 for row in result["controls"])


def test_provider_boundary_is_hard_and_secrets_are_not_embedded() -> None:
    source = inspect.getsource(pilot)
    tree = ast.parse(source)
    string_values = {
        node.value for node in ast.walk(tree) if isinstance(node, ast.Constant) and isinstance(node.value, str)
    }

    assert pilot.SUBMIT_CONFIRMATION in string_values
    assert pilot.AUTHORIZATION_CONFIRMATION in string_values
    assert "api_key=" not in source
    assert "IBM_CLOUD_API_KEY" not in source
    assert "QCTRL_API_KEY" not in source
    assert "fireopal.execute" not in inspect.getsource(pilot.prepare_pilot)
    assert "fireopal.execute" not in inspect.getsource(pilot.validate_pilot)
    assert "confirm_submit" in inspect.getsource(pilot.submit_pilot)
    assert "intent.exists() or args.receipt.exists()" in inspect.getsource(
        pilot.submit_pilot
    )


def test_cli_defaults_match_authorized_protocol() -> None:
    prepare = pilot.build_parser().parse_args(["prepare"])
    submit = pilot.build_parser().parse_args(["submit"])

    assert prepare.bundle == pilot.DEFAULT_BUNDLE
    assert submit.confirm_submit == ""
    assert submit.qiskit_account == "default-ibm-cloud"
    assert pilot.BACKEND == "ibm_fez"
    assert pilot.SHOTS == 4096
    assert pilot.CIRCUITS == 66
