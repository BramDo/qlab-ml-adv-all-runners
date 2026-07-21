from __future__ import annotations

import ast
import inspect

import numpy as np

import qiskit_qos_pbmc68k_q60_grid_d12_quantum_teacher_screen as screen
import qiskit_qos_pbmc68k_q60_shallow_fireopal_validate as q60


def test_teacher_panel_is_broad_and_needs_only_yz_measurements() -> None:
    mappings, indices = screen.broad_yz_observables()

    assert len(indices) == 94
    assert {len(mappings[index]) for index in indices} == {4, 8}
    assert {
        q60.measurement_basis_for_mapping(mappings[index]) for index in indices
    } == {"Y", "Z"}


def test_teacher_selection_uses_largest_raw_variance_deterministically() -> None:
    base = np.linspace(-1.0, 1.0, 20)
    features = np.column_stack([0.1 * base, 3.0 * base, -2.0 * base, 0.2 * base])

    selected, center, weights = screen.select_teacher_observables(
        features, [100, 101, 102, 103], count=2
    )
    repeated = screen.select_teacher_observables(
        features, [100, 101, 102, 103], count=2
    )

    assert set(selected) == {1, 2}
    assert np.array_equal(selected, repeated[0])
    assert np.allclose(center, repeated[1])
    assert np.allclose(weights, repeated[2])
    assert np.isclose(np.linalg.norm(weights), 1.0)
    assert weights[np.argmax(np.abs(weights))] > 0.0


def test_margin_task_is_balanced_disjoint_and_reproducible() -> None:
    scores = np.linspace(-2.0, 2.0, 1024)
    source_indices = np.arange(10000, 11024)

    train, test, labels, threshold, audit = screen.construct_balanced_margin_task(
        scores,
        source_indices,
        train_samples=256,
        test_samples=256,
        seed=17,
    )
    repeated = screen.construct_balanced_margin_task(
        scores,
        source_indices,
        train_samples=256,
        test_samples=256,
        seed=17,
    )

    assert np.array_equal(train, repeated[0])
    assert np.array_equal(test, repeated[1])
    assert threshold == repeated[3]
    assert len(train) == len(set(train)) == 256
    assert len(test) == len(set(test)) == 256
    assert not set(train) & set(test)
    assert np.sum(labels[train] < 0.0) == np.sum(labels[train] > 0.0) == 128
    assert np.sum(labels[test] < 0.0) == np.sum(labels[test] > 0.0) == 128
    assert audit["discarded_middle_candidates"] == 512
    assert audit["selected_margin_minimum"] > 0.0
    assert audit["train_test_overlap"] == 0


def test_teacher_shot_noise_probe_is_bounded() -> None:
    train = np.asarray([[0.7, -0.2], [-0.7, 0.2], [0.6, -0.1], [-0.6, 0.1]])
    test = np.asarray([[0.65, -0.15], [-0.65, 0.15]])
    y_train = np.asarray([1.0, -1.0, 1.0, -1.0])
    y_test = np.asarray([1.0, -1.0])

    result = screen.shot_noise_probe(
        train,
        test,
        y_train,
        y_test,
        shots=128,
        replicates=20,
        seed=23,
    )

    assert result["replicates"] == 20
    assert 0.0 <= result["mean_test_balanced_accuracy"] <= 1.0
    assert 0.0 <= result["ci95_percentile"][0] <= 1.0
    assert 0.0 <= result["ci95_percentile"][1] <= 1.0


def test_runner_is_local_only_and_claim_boundary_is_explicit() -> None:
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
    assert "semi-synthetic quantum-teacher task" in source
    assert "not natural-label biology" in source


def test_cli_defaults_define_large_provider_free_screen() -> None:
    args = screen.build_parser().parse_args([])

    assert args.construction_samples == 256
    assert args.candidate_samples == 1024
    assert args.train_samples == 256
    assert args.test_samples == 256
    assert args.teacher_observables == 16
    assert args.shots == 128
    assert not hasattr(args, "backend")
    assert not hasattr(args, "api_key")
