from __future__ import annotations

import ast
import inspect
import math

import numpy as np
import pytest

import qiskit_qos_pbmc_q40_sqrtq_b4_fireopal_validate as validate


def _blocks(rows: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    values = rng.normal(size=(rows, validate.BLOCK_COUNT, validate.QUBITS))
    return values / np.linalg.norm(values, axis=2, keepdims=True)


def test_frozen_screen_advances_sqrtq_only_as_structural_candidate() -> None:
    screen, source = validate.load_frozen_specifications(
        validate.DEFAULT_SCREEN_REPORT, validate.DEFAULT_SOURCE_REPORT
    )

    assert screen["preflight_funnel"]["survivors"] == ["sqrt_q"]
    assert screen["confirmation_gate"]["structural_candidate_for_next_gate"] == "sqrt_q"
    assert screen["confirmation_gate"]["selected_scale_law"] is None
    assert screen["confirmation_gate"]["passed"] is False
    assert len(source["split"]["train_indices"]) == 32
    assert len(source["split"]["test_indices"]) == 32


def test_small_batch_has_three_global_bases_and_numeric_round_trips() -> None:
    train = _blocks(2, 11)
    test = _blocks(2, 12)
    qasms, manifest, metadata = validate.build_seed_circuits(
        train,
        test,
        np.asarray([-1.0, 1.0]),
        np.asarray([1.0, -1.0]),
        [10, 11],
        [20, 21],
        seed_transpiler=1729,
    )

    assert len(qasms) == 12
    assert metadata["logical_base_circuit_count"] == 4
    assert metadata["measurement_bases"] == ["X", "Y", "Z"]
    assert metadata["pair_multiplier"] == pytest.approx(math.sqrt(40.0))
    assert metadata["selection_protocol"]["labels_in_provider_payload"] is False
    assert [row["measurement_basis"] for row in manifest[:3]] == ["X", "Y", "Z"]
    assert manifest[0]["split"] == "train"
    assert manifest[0]["source_row_index"] == 10
    assert manifest[-1]["split"] == "test"
    assert manifest[-1]["source_row_index"] == 21
    assert [row["circuit_index"] for row in manifest] == list(range(12))
    assert all(row["round_trip_validated"] for row in manifest)
    assert all(row["all_parameters_numeric"] for row in manifest)
    assert all(row["metrics"]["num_qubits"] == 40 for row in manifest)
    assert {
        row["logical_metrics_before_measurement"]["depth"] for row in manifest
    } == {20}
    assert {
        row["logical_metrics_before_measurement"]["two_qubit_gates"]
        for row in manifest
    } == {87}
    assert all("OPENQASM 2.0;" in qasm and "pi" not in qasm for qasm in qasms)


def test_runner_has_no_hardware_execution_or_retrieval_path() -> None:
    source = inspect.getsource(validate)
    tree = ast.parse(source)
    imported_modules = {
        alias.name
        for node in ast.walk(tree)
        if isinstance(node, (ast.Import, ast.ImportFrom))
        for alias in node.names
    }

    assert "fireopal" not in imported_modules
    assert ".execute(" not in source
    assert ".get_result(" not in source
    assert '"execution_attempted": False' in source
    assert '"quantum_seconds_used": 0' in source

    args = validate.build_parser().parse_args([])
    assert args.backend == "ibm_fez"
    assert args.shots == 128
    assert args.validate is False
    assert not hasattr(args, "confirm_submit")


def test_claim_boundary_keeps_resource_screen_conservative() -> None:
    source = inspect.getsource(validate.main)

    assert "input compatibility only" in source
    assert "not proof of classical hardness" in source
    assert "quantum advantage" in source
