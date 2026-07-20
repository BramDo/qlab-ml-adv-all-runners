from __future__ import annotations

import inspect

import numpy as np
from qiskit.quantum_info import Statevector

import qiskit_official_qos_realdata_gate as gate
import qiskit_official_qos_sampling_port as port


def test_exact_flat_probabilities_match_qiskit_and_official_jax() -> None:
    vector = np.asarray([1, -1, 1, 1, -1, -1, 1, -1] * 2, dtype=np.float64)
    rng = np.random.default_rng(1234)
    sampled_indices, sampled_values = port.sample_from_vector(vector, 64, rng)

    analytic = port.flat_interference_probabilities_from_samples(
        sampled_indices, sampled_values, 16
    )
    circuit, _ = port.build_flat_interference_circuit_from_samples(
        sampled_indices, sampled_values, 16
    )
    qiskit = np.abs(np.asarray(Statevector.from_instruction(circuit).data)) ** 2
    jax = np.abs(
        port.flat_interference_state_from_jax(sampled_indices, sampled_values, 16)
    ) ** 2

    assert np.max(np.abs(analytic - qiskit)) < 1e-12
    assert np.max(np.abs(analytic - jax)) < 1e-12
    assert np.isclose(np.sum(analytic), 1.0)


def test_training_median_mapping_has_no_test_leakage() -> None:
    train = np.asarray([[0.0, 2.0], [1.0, 0.0], [2.0, 1.0]])
    test = np.asarray([[100.0, -100.0]])
    train_flat, test_flat, thresholds = gate.training_median_flat_vectors(train, test)

    assert np.array_equal(thresholds, np.asarray([1.0, 1.0]))
    assert np.array_equal(train_flat, np.asarray([[-1, 1], [1, -1], [1, 1]]))
    assert np.array_equal(test_flat, np.asarray([[1, -1]]))


def test_hellinger_fidelity_kernel_is_psd_with_unit_diagonal() -> None:
    probabilities = np.asarray(
        [[1.0, 0.0, 0.0], [0.25, 0.75, 0.0], [0.2, 0.3, 0.5]],
        dtype=np.float64,
    )
    kernel = gate.hellinger_fidelity_kernel(probabilities)

    assert np.max(np.abs(kernel - kernel.T)) < 1e-12
    assert np.max(np.abs(np.diag(kernel) - 1.0)) < 1e-12
    assert np.min(np.linalg.eigvalsh(kernel)) > -1e-12


def test_gate_requires_four_strict_wins_and_positive_mean() -> None:
    passing = gate.evaluate_gate_summary(
        [{"balanced_accuracy_delta": value} for value in (0.05, 0.04, 0.03, 0.02, -0.01)],
        bootstrap_replicates=1000,
    )
    failing = gate.evaluate_gate_summary(
        [{"balanced_accuracy_delta": value} for value in (0.05, 0.04, 0.03, 0.0, 0.0)],
        bootstrap_replicates=1000,
    )

    assert passing["passed"] is True
    assert passing["strict_wins"] == 4
    assert failing["passed"] is False
    assert failing["strict_wins"] == 3


def test_local_gate_has_no_provider_boundary() -> None:
    source = inspect.getsource(gate)

    assert "import fireopal" not in source
    assert "fireopal.execute" not in source
    assert "authenticate_qctrl_account" not in source
    assert gate.GATE_MIN_WINS == 4
    assert gate.DEFAULT_DIMENSIONS == (16, 64, 256, 1024)
    assert gate.DEFAULT_SAMPLES_PER_DIMENSION == 4
