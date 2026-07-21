from __future__ import annotations

import ast
import inspect

import numpy as np

import qiskit_qos_pbmc68k_q60_grid_d12_large_blind_screen as screen
import qiskit_qos_pbmc68k_q60_shallow_fireopal_validate as q60


def test_balanced_split_is_deterministic_and_historically_disjoint() -> None:
    labels = np.concatenate([-np.ones(400), np.ones(400)])
    excluded = {0, 1, 2, 400, 401, 402}

    train, test = screen.balanced_disjoint_split(
        labels,
        excluded_indices=excluded,
        train_samples=256,
        test_samples=256,
        seed=1234,
    )
    repeated_train, repeated_test = screen.balanced_disjoint_split(
        labels,
        excluded_indices=excluded,
        train_samples=256,
        test_samples=256,
        seed=1234,
    )

    assert np.array_equal(train, repeated_train)
    assert np.array_equal(test, repeated_test)
    assert len(train) == len(set(train)) == 256
    assert len(test) == len(set(test)) == 256
    assert not set(train) & set(test)
    assert not (set(train) | set(test)) & excluded
    assert np.sum(labels[train] < 0) == np.sum(labels[train] > 0) == 128
    assert np.sum(labels[test] < 0) == np.sum(labels[test] > 0) == 128


def test_index_collection_only_reads_split_index_fields() -> None:
    report = {
        "train_indices": [1, 2],
        "nested": {
            "test_indices": [3],
            "selected_master_indices": [999],
            "other": [4, 5],
        },
    }

    assert screen._index_values(report) == {1, 2, 3}


def test_paired_statistics_detect_quantum_only_correct_predictions() -> None:
    labels = np.asarray([-1.0, -1.0, 1.0, 1.0])
    quantum_scores = labels.copy()
    classical_scores = -labels

    result = screen.paired_test_statistics(
        labels,
        quantum_scores,
        classical_scores,
        replicates=100,
        seed=7,
    )

    discordance = result["paired_discordance"]
    bootstrap = result["stratified_paired_bootstrap"]
    assert discordance["quantum_only_correct"] == 4
    assert discordance["classical_only_correct"] == 0
    assert discordance["mcnemar_exact_two_sided_p"] == 0.125
    assert bootstrap["mean_quantum_minus_classical"] == 1.0
    assert bootstrap["ci95_percentile"] == [1.0, 1.0]


def test_shot_noise_probe_returns_bounded_summary() -> None:
    features = np.asarray(
        [
            [0.2, -0.2],
            [0.7, -0.4],
            [-0.7, 0.4],
            [0.6, -0.3],
            [-0.6, 0.3],
        ]
    )
    labels = np.asarray([1.0, -1.0])

    result = screen.frozen_shot_noise_probe(
        features,
        labels,
        labels,
        shots=128,
        replicates=20,
        seed=11,
    )

    assert result["replicates"] == 20
    assert 0.0 <= result["mean_test_balanced_accuracy"] <= 1.0
    assert 0.0 <= result["ci95_percentile"][0] <= 1.0
    assert 0.0 <= result["ci95_percentile"][1] <= 1.0


def test_runner_is_local_only_and_test_is_not_used_for_fitting() -> None:
    source = inspect.getsource(screen)
    tree = ast.parse(source)
    imported_modules = {
        alias.name
        for node in ast.walk(tree)
        if isinstance(node, (ast.Import, ast.ImportFrom))
        for alias in node.names
    }

    assert "fireopal" not in imported_modules
    assert ".execute(" not in source
    assert ".validate(" not in source
    assert ".get_result(" not in source
    assert '"provider_calls": []' in source
    assert '"execution_attempted": False' in source
    assert '"test_labels_used_for_fitting_or_selection": False' in source


def test_defaults_and_frozen_observables_match_planned_hardware_scope() -> None:
    args = screen.build_parser().parse_args([])
    _, mappings, selected = screen.grid_validate._load_frozen_screen(
        args.screen_report
    )
    bases = {
        q60.measurement_basis_for_mapping(mappings[index]) for index in selected
    }

    assert args.train_samples == 256
    assert args.test_samples == 256
    assert args.shots == 128
    assert len(selected) == 24
    assert bases == {"Y", "Z"}
    assert not hasattr(args, "backend")
    assert not hasattr(args, "api_key")
