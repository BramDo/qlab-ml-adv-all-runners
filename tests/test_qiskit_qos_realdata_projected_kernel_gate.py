from __future__ import annotations

import inspect

import numpy as np
import pytest

import qiskit_qos_realdata_projected_kernel_gate as gate


def test_multiscale_panel_is_homogeneous_and_bounded_support() -> None:
    for width in (4, 6, 8, 10, 40, 60):
        mappings = gate.homogeneous_multiscale_mappings(width)
        summary = gate.mapping_panel_summary(mappings, num_qubits=width)

        assert summary["all_homogeneous_xyz"] is True
        assert summary["global_measurement_bases"] == ["X", "Y", "Z"]
        assert summary["measurement_circuits_per_sample"] == 3
        assert summary["largest_observable_support"] <= 4
        assert set(summary["support_size_counts"]) == {"1", "2", "3", "4"}
        assert len(mappings) > len(gate.hardware_local_mappings(width))
        assert len({gate._mapping_key(mapping) for mapping in mappings}) == len(mappings)


def test_observable_panel_selector_preserves_local_default() -> None:
    for width in (4, 40, 60):
        assert gate.observable_mappings(
            width, gate.DEFAULT_OBSERVABLE_PANEL
        ) == gate.hardware_local_mappings(width)
        assert gate.observable_mappings(
            width, "multiscale_support4"
        ) == gate.homogeneous_multiscale_mappings(width)


def test_default_hardware_panel_has_five_q_minus_two_observables() -> None:
    for width in (4, 10, 40, 60):
        mappings = gate.hardware_local_mappings(width)
        summary = gate.mapping_panel_summary(mappings, num_qubits=width)

        assert len(mappings) == 5 * width - 2
        assert summary["measurement_circuits_per_sample"] == 3
        assert summary["largest_observable_support"] == 2


def test_projected_features_match_mps_at_small_width() -> None:
    rows = gate.small_width_validation(
        (4,), bond_dimension=64, architecture_name="grid_mixer_d12"
    )

    assert rows[0]["passed"] is True
    assert rows[0]["max_abs_mps_minus_statevector"] < 1e-9
    assert rows[0]["kernel_diagonal_max_abs_error"] < 1e-12
    assert rows[0]["kernel_minimum_eigenvalue"] > -1e-10


def test_rbf_kernel_is_psd_with_unit_diagonal() -> None:
    features = np.asarray(
        [[0.1, -0.2, 0.3], [0.3, 0.4, -0.1], [-0.5, 0.2, 0.7]],
        dtype=np.float64,
    )
    gamma = gate.median_gamma(features)
    kernel = gate.projected_rbf_kernel(features, features, gamma=gamma)

    assert np.max(np.abs(kernel - kernel.T)) < 1e-12
    assert np.max(np.abs(np.diag(kernel) - 1.0)) < 1e-12
    assert np.min(np.linalg.eigvalsh(kernel)) > -1e-12


def test_projected_kernel_selection_and_test_evaluation() -> None:
    rng = np.random.default_rng(71)
    labels_train = np.asarray([-1, 1] * 20, dtype=np.int64)
    labels_test = np.asarray([-1, 1] * 8, dtype=np.int64)
    train = rng.normal(scale=0.2, size=(40, 12))
    test = rng.normal(scale=0.2, size=(16, 12))
    train[:, 0] += labels_train
    test[:, 0] += labels_test

    result = gate.fit_projected_kernel(
        train,
        test,
        labels_train,
        labels_test,
        feature_counts=(2, 4, 8),
        c_values=(0.1, 1.0),
        gamma_multipliers=(0.5, 1.0),
        cv_splits=4,
        cv_seed=19,
        shot_intent=128,
    )

    assert result["balanced_accuracy"] >= 0.9
    assert result["inner_cv"]["all_selection_training_only"] is True
    assert result["kernel_verification"]["passed"] is True


def test_bond_dimension_preflight_selects_lower_converged_candidate(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    reference = np.zeros((4, 3), dtype=np.float64)

    def fake_simulate(circuits, mappings, *, bond_dimension, threshold, progress_label):
        del circuits, mappings, threshold, progress_label
        offsets = {32: 1e-2, 64: 2e-3, 128: 2e-4, 256: 0.0}
        return reference + offsets[bond_dimension], 0.01

    monkeypatch.setattr(gate.architecture, "simulate_feature_rows", fake_simulate)
    selected, report = gate.select_mps_bond_dimension(
        np.zeros((4, 4), dtype=np.float64),
        gate.hardware_local_mappings(4),
        bond_dimensions=(32, 64, 128, 256),
        threshold=1e-10,
        tolerance=1e-3,
        label="test",
        architecture_name="grid_mixer_d12",
    )

    assert selected == 128
    assert report["reference_bond_dimension"] == 256
    assert report["passed"] is True


def test_local_runner_has_no_provider_boundary_or_exponential_allocation() -> None:
    source = inspect.getsource(gate)

    assert "import fireopal" not in source
    assert "fireopal.execute" not in source
    assert "authenticate_qctrl_account" not in source
    assert "np.zeros(2 **" not in source
    assert gate.DEFAULT_WIDTHS == (40, 60)
    assert gate.DEFAULT_BOND_DIMENSION_CANDIDATES == (32, 64, 128, 256)
    assert gate.GLOBAL_MEASUREMENT_BASES == ("X", "Y", "Z")


def test_grid_mixer_d8_has_frozen_intermediate_depth() -> None:
    vector = np.linspace(-0.5, 0.5, 60)
    path = gate.feature_map_circuit(vector, "path_rzz_d4")
    middle = gate.feature_map_circuit(vector, "grid_mixer_d8")
    grid = gate.feature_map_circuit(vector, "grid_mixer_d12")

    assert path.num_qubits == middle.num_qubits == grid.num_qubits == 60
    assert path.depth() < middle.depth() < grid.depth()
    assert middle.depth() == 8
    assert middle.count_ops()["rzz"] == 60


def test_grid_mixer_d8_matches_exact_statevector_at_small_width() -> None:
    rows = gate.small_width_validation(
        (4,), bond_dimension=64, architecture_name="grid_mixer_d8"
    )

    assert rows[0]["passed"] is True
    assert rows[0]["representative_circuit_metrics"]["depth"] == 8
    assert rows[0]["max_abs_mps_minus_statevector"] < 1e-9


def test_multiscale_grid_mixer_d8_matches_exact_statevector_at_small_width() -> None:
    rows = gate.small_width_validation(
        (4,),
        bond_dimension=64,
        architecture_name="grid_mixer_d8",
        observable_panel_name="multiscale_support4",
    )

    assert rows[0]["passed"] is True
    assert rows[0]["observables"] == len(gate.homogeneous_multiscale_mappings(4))
    assert rows[0]["max_abs_mps_minus_statevector"] < 1e-9
