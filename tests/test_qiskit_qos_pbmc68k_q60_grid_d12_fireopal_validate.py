from __future__ import annotations

import ast
import inspect

import numpy as np

import qiskit_qos_pbmc68k_q60_grid_d12_fireopal_validate as validate
from qiskit_qos_pbmc68k_q60_scrambled_mixer_screen import SeedData


def _small_seed_data() -> SeedData:
    rng = np.random.default_rng(11)
    return SeedData(
        encoded_train=rng.normal(size=(2, 60)),
        encoded_test=rng.normal(size=(2, 60)),
        y_train=np.asarray([-1.0, 1.0]),
        y_test=np.asarray([-1.0, 1.0]),
        train_indices=np.asarray([10, 11]),
        test_indices=np.asarray([12, 13]),
        metadata={"test_fixture": True},
    )


def test_frozen_screen_preserves_exploratory_signal_and_selected_pairs() -> None:
    report, mappings, selected = validate._load_frozen_screen(
        validate.DEFAULT_SCREEN_REPORT
    )
    signal = validate._advantage_signal(report)

    assert len(selected) == 24
    assert all(len(mappings[index]) in {1, 2} for index in selected)
    assert signal["classification"] == "exploratory_not_confirmed"
    assert signal["positive_training_cv_delta"] > 0.0
    assert signal["fixed_test_delta"] < 0.0
    assert signal[
        "hardware_is_scientifically_worthwhile_despite_unconfirmed_advantage"
    ] is True


def test_small_batch_has_three_global_bases_and_numeric_round_trips() -> None:
    _, mappings, selected = validate._load_frozen_screen(
        validate.DEFAULT_SCREEN_REPORT
    )
    qasms, manifest, metadata = validate.build_seed_circuits(
        _small_seed_data(),
        mappings,
        selected,
        seed_transpiler=1729,
    )

    assert len(qasms) == (1 + 2 + 2) * 3
    assert metadata["logical_base_circuit_count"] == 5
    assert metadata["measurement_bases"] == ["X", "Y", "Z"]
    assert metadata["selection_protocol"][
        "uses_test_metrics_for_hardware_configuration"
    ] is False
    assert {row["measurement_basis"] for row in manifest} == {"X", "Y", "Z"}
    assert all(row["round_trip_validated"] for row in manifest)
    assert all(row["all_parameters_numeric"] for row in manifest)
    assert all(row["metrics"]["num_qubits"] == 60 for row in manifest)


def test_validator_has_no_hardware_execution_or_retrieval_path() -> None:
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


def test_paper_minimum_does_not_hide_accuracy_boundary() -> None:
    source = inspect.getsource(validate.main)

    assert '"accuracy_superiority_is_not_a_submission_prerequisite": True' in source
    assert '"honest_classical_and_noiseless_comparisons_required": True' in source
    assert "quantum advantage" in source
