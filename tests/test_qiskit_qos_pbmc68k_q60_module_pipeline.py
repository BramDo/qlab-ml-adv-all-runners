from __future__ import annotations

import ast
import inspect

import numpy as np
from scipy import sparse

import qiskit_qos_pbmc68k_q60_module_pipeline as pipeline


def test_study_allocation_is_deterministic_balanced_and_fully_disjoint() -> None:
    labels = np.concatenate((-np.ones(2500, dtype=int), np.ones(2500, dtype=int)))
    sentinel_train = np.concatenate((np.arange(16), np.arange(2500, 2516)))
    sentinel_test = np.concatenate((np.arange(16, 32), np.arange(2516, 2532)))

    first = pipeline.allocate_study_indices(
        labels,
        sentinel_train=sentinel_train,
        sentinel_test=sentinel_test,
    )
    second = pipeline.allocate_study_indices(
        labels,
        sentinel_train=sentinel_train,
        sentinel_test=sentinel_test,
    )

    assert first == second
    assert first["all_sections_pairwise_disjoint"] is True
    assert first["allocated_unique_rows"] == 64 + 512 + 6 * 512
    assert first["module_learning_pool"]["selection_used_labels"] is False
    sections: list[list[int]] = [
        first["sentinel"]["train_indices"],
        first["sentinel"]["test_indices"],
        first["module_learning_pool"]["indices"],
    ]
    for split in [*first["development"], first["final"]]:
        sections.extend((split["train_indices"], split["test_indices"]))
        assert np.sum(labels[split["train_indices"]] < 0) == 128
        assert np.sum(labels[split["train_indices"]] > 0) == 128
        assert np.sum(labels[split["test_indices"]] < 0) == 128
        assert np.sum(labels[split["test_indices"]] > 0) == 128
    flattened = [index for section in sections for index in section]
    assert len(flattened) == len(set(flattened))


def test_label_free_modules_and_four_statistics_are_deterministic() -> None:
    rng = np.random.default_rng(9)
    dense = rng.poisson(0.4, size=(80, 200)).astype(float)
    dense[:, :20] += rng.poisson(1.0, size=(80, 20))
    matrix = sparse.csr_matrix(dense)

    first = pipeline.learn_coexpression_modules(
        matrix,
        selected_genes=120,
        module_count=6,
        detection_min=0.01,
        detection_max=1.0,
        random_state=6110,
        n_init=20,
    )
    second = pipeline.learn_coexpression_modules(
        matrix,
        selected_genes=120,
        module_count=6,
        detection_min=0.01,
        detection_max=1.0,
        random_state=6110,
        n_init=20,
    )

    assert first["selection"]["selected_gene_indices"] == second["selection"][
        "selected_gene_indices"
    ]
    assert first["modules"] == second["modules"]
    assert sum(row["gene_count"] for row in first["modules"]) == 120
    assert min(row["gene_count"] for row in first["modules"]) > 0
    statistics = pipeline.module_statistics(matrix[:10], first["modules"])
    assert statistics.shape == (10, 4, 6)
    assert np.all(np.isfinite(statistics))
    assert np.all(statistics >= 0.0)


def test_robust_scaler_is_train_only_and_blocks_are_bounded() -> None:
    rng = np.random.default_rng(11)
    training = rng.normal(size=(16, 4, 60))
    test = rng.normal(size=(8, 4, 60))
    scaler = pipeline.fit_robust_block_scaler(training)
    repeated = pipeline.fit_robust_block_scaler(training.copy())
    extreme_test = test.copy()
    extreme_test[0] = 1e12

    assert np.array_equal(scaler["median"], repeated["median"])
    assert scaler["fit_scope"] == "training rows only"
    normal = pipeline.transform_robust_blocks(test, scaler)
    extreme = pipeline.transform_robust_blocks(extreme_test, scaler)
    assert np.array_equal(normal[1:], extreme[1:])
    assert np.max(np.abs(extreme)) <= 1.0
    norms = np.linalg.norm(extreme, axis=2)
    assert np.all((np.isclose(norms, 1.0)) | (np.isclose(norms, 0.0)))


def test_q60_circuit_and_readout_panel_match_frozen_counts() -> None:
    rng = np.random.default_rng(12)
    blocks = rng.normal(size=(1, 4, 60))
    blocks /= np.linalg.norm(blocks, axis=2, keepdims=True)
    circuit = pipeline.build_unmeasured_circuits(blocks)[0]
    metrics = pipeline.coherent.shallow.q40_validate.circuit_metrics(circuit)

    assert circuit.num_qubits == 60
    assert metrics["depth"] == 20
    assert metrics["two_qubit_gates"] == 134
    assert len(pipeline.observable_mappings()) == 627
    assert pipeline.PAIR_MULTIPLIER == np.sqrt(60.0)


def test_local_pipeline_has_no_provider_surface() -> None:
    source = inspect.getsource(pipeline)
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
    assert '"quantum_seconds_used": 0' in source
