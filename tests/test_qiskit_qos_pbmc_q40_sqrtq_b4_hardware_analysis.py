from __future__ import annotations

import inspect

import numpy as np
from scipy import sparse

import qiskit_qos_pbmc_q40_sqrtq_b4_hardware_analysis as analysis


def test_training_only_selector_has_no_test_argument_and_is_deterministic() -> None:
    parameters = inspect.signature(analysis.select_model_training_only).parameters
    assert not any("test" in name for name in parameters)
    rng = np.random.default_rng(9)
    matrix = rng.normal(size=(32, 6))
    labels = np.asarray([-1, 1] * 16, dtype=np.int64)
    representations = {"hardware_pauli_405": matrix}
    candidates = [
        {
            "candidate_id": "c000",
            "representation": "hardware_pauli_405",
            "family": "ridge",
            "alpha": 1.0,
        },
        {
            "candidate_id": "c001",
            "representation": "hardware_pauli_405",
            "family": "linear_svc",
            "C": 0.1,
        },
    ]
    first = analysis.select_model_training_only(representations, labels, candidates)
    second = analysis.select_model_training_only(representations, labels, candidates)
    assert first == second


def test_library_log1p_is_finite_and_row_local() -> None:
    matrix = sparse.csr_matrix([[1.0, 3.0, 0.0], [0.0, 2.0, 2.0]])
    baseline = analysis._library_log1p(matrix).toarray()
    changed = matrix.copy().tolil()
    changed[1, 0] = 100.0
    changed = analysis._library_log1p(changed.tocsr()).toarray()
    assert np.all(np.isfinite(baseline))
    np.testing.assert_allclose(baseline[0], changed[0])


def test_paired_statistics_detect_direction() -> None:
    labels = np.asarray([-1] * 16 + [1] * 16)
    hardware = labels.copy()
    classical = labels.copy()
    classical[:8] *= -1
    report = analysis.paired_test_statistics(
        labels, hardware, classical, replicates=1000, seed=3
    )
    assert report["paired_discordance"]["hardware_only_correct"] == 8
    assert report["paired_discordance"]["classical_only_correct"] == 0
    assert report["paired_discordance"]["mcnemar_exact_two_sided_p"] < 0.05
    assert report["stratified_paired_bootstrap"]["mean_hardware_minus_classical"] > 0


def test_frozen_hardware_artifacts_load_when_present() -> None:
    if not analysis.DEFAULT_RESULT.is_file() or not analysis.DEFAULT_PLAN.is_file():
        return
    x_train, x_test, y_train, y_test, metadata = analysis.load_frozen_hardware(
        analysis.DEFAULT_RESULT, analysis.DEFAULT_PLAN
    )
    assert x_train.shape == (32, 405)
    assert x_test.shape == (32, 405)
    assert sorted(y_train.tolist()) == [-1] * 16 + [1] * 16
    assert sorted(y_test.tolist()) == [-1] * 16 + [1] * 16
    assert metadata["action_id"] == "2334162"
    assert metadata["total_shots"] == 24_576


def test_module_contains_no_provider_action_calls() -> None:
    source = inspect.getsource(analysis)
    assert "fireopal.execute" not in source
    assert "fireopal.get_result" not in source
    assert "authenticate_qctrl_account" not in source
