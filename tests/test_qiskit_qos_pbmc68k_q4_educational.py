from __future__ import annotations

import numpy as np
from scipy import sparse

import qiskit_qos_pbmc68k_q4_educational as tutorial


def test_balanced_split_is_deterministic_balanced_and_disjoint() -> None:
    labels = np.concatenate((-np.ones(30, dtype=int), np.ones(30, dtype=int)))
    first = tutorial.balanced_split_indices(
        labels, train_size=16, test_size=16, seed=11
    )
    second = tutorial.balanced_split_indices(
        labels, train_size=16, test_size=16, seed=11
    )

    assert np.array_equal(first[0], second[0])
    assert np.array_equal(first[1], second[1])
    assert np.intersect1d(first[0], first[1]).size == 0
    assert np.sum(labels[first[0]] < 0) == 8
    assert np.sum(labels[first[0]] > 0) == 8
    assert np.sum(labels[first[1]] < 0) == 8
    assert np.sum(labels[first[1]] > 0) == 8


def test_label_free_gene_selection_is_deterministic() -> None:
    rng = np.random.default_rng(4)
    counts = sparse.csr_matrix(rng.poisson(0.5, size=(20, 30)))

    first = tutorial.select_label_free_genes(
        counts, detection_min=0.01, detection_max=1.0
    )
    second = tutorial.select_label_free_genes(
        counts.copy(), detection_min=0.01, detection_max=1.0
    )

    assert np.array_equal(first[0], second[0])
    assert len(first[0]) == tutorial.QUBITS
    assert np.all(first[1] >= 0.0)


def test_scaling_is_fit_on_training_rows_only() -> None:
    train = np.arange(32, dtype=float).reshape(8, 4)
    test = np.ones((3, 4), dtype=float)
    extreme_test = test.copy()
    extreme_test[0] = 1e12

    train_a, test_a, scaler_a = tutorial.scale_gene_inputs(train, test)
    train_b, test_b, scaler_b = tutorial.scale_gene_inputs(train, extreme_test)

    assert np.array_equal(train_a, train_b)
    assert np.array_equal(scaler_a["mean"], scaler_b["mean"])
    assert np.array_equal(scaler_a["std"], scaler_b["std"])
    assert np.array_equal(test_a[1:], test_b[1:])
    assert np.max(np.abs(test_b)) <= np.pi


def test_counts_to_z_and_zz_features() -> None:
    all_zero = tutorial.features_from_counts({"0000": 100})
    all_one = tutorial.features_from_counts({"1111": 100})
    balanced = tutorial.features_from_counts({"0000": 50, "1111": 50})

    assert np.array_equal(all_zero, np.ones(8))
    assert np.array_equal(all_one[:4], -np.ones(4))
    assert np.array_equal(all_one[4:], np.ones(4))
    assert np.array_equal(balanced[:4], np.zeros(4))
    assert np.array_equal(balanced[4:], np.ones(4))


def test_circuit_is_four_qubit_and_shallow() -> None:
    circuit = tutorial.build_feature_circuit(np.zeros(4))

    assert circuit.num_qubits == 4
    assert circuit.num_clbits == 4
    assert circuit.count_ops()["ry"] == 4
    assert circuit.count_ops()["cx"] == 4
    assert circuit.depth() <= 7
