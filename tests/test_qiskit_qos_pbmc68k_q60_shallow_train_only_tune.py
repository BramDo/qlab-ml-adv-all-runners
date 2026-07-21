from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

import qiskit_qos_pbmc68k_q60_shallow_fireopal_validate as q60
import qiskit_qos_pbmc68k_q60_shallow_train_only_tune as tuner
import qiskit_qos_toy_model as toy


def _synthetic_data() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(42)
    encoded_train = rng.uniform(-0.8, 0.8, size=(12, 4))
    encoded_test = rng.uniform(-0.8, 0.8, size=(6, 4))
    y_train = np.asarray([-1.0, 1.0] * 6)
    y_test = np.asarray([-1.0, 1.0] * 3)
    return encoded_train, y_train, encoded_test, y_test


def test_scaled_feature_matrix_matches_default_q60_expectations() -> None:
    values = np.asarray([-0.7, 0.2, 0.5, -0.3])
    parameters = [q60.query_parameters(values)]
    mappings = toy.pauli_feature_mappings(4, family="local")[:8]

    actual = tuner.scaled_feature_matrix(
        parameters,
        mappings,
        single_scale=q60.SINGLE_SCALE,
        phase_scale=q60.PHASE_SCALE,
        pair_scale=q60.PAIR_SCALE,
    )
    expected = q60.exact_local_feature_matrix(parameters, mappings)

    assert actual == pytest.approx(expected, abs=1e-12)


def test_grid_search_has_no_test_input_and_balanced_folds() -> None:
    encoded_train, y_train, _, _ = _synthetic_data()
    mappings = toy.pauli_feature_mappings(4, family="local")

    result = tuner.train_only_grid_search(
        encoded_train,
        y_train,
        mappings=mappings,
        single_scales=(1.0,),
        phase_scales=(0.5,),
        pair_scales=(2.0,),
        selected_counts=(4,),
        cv_folds=3,
        seed=11,
        shot_intent=1024,
    )

    assert result["selection_scope"] == "training_split_only"
    assert result["test_inputs_seen"] is False
    assert result["test_labels_seen"] is False
    assert result["candidate_configurations"] == 1
    assert len(result["chosen"]["folds"]) == 3
    assert all(row["validation_positive"] == 2 for row in result["chosen"]["folds"])
    assert all(row["validation_negative"] == 2 for row in result["chosen"]["folds"])


def test_test_labels_cannot_change_selected_observables() -> None:
    encoded_train, y_train, encoded_test, y_test = _synthetic_data()
    mappings = toy.pauli_feature_mappings(4, family="local")
    configuration = {
        "single_scale": 1.0,
        "phase_scale": 0.5,
        "pair_scale": 2.0,
        "selected_feature_count": 4,
    }

    normal = tuner.evaluate_fixed_configuration(
        encoded_train,
        y_train,
        encoded_test,
        y_test,
        mappings=mappings,
        configuration=configuration,
        shot_intent=1024,
        bootstrap_seed=100,
    )
    flipped = tuner.evaluate_fixed_configuration(
        encoded_train,
        y_train,
        encoded_test,
        -y_test,
        mappings=mappings,
        configuration=configuration,
        shot_intent=1024,
        bootstrap_seed=101,
    )

    assert normal["selected_observables"] == flipped["selected_observables"]
    assert normal["selection_uses_training_only"] is True


def test_tie_break_prefers_smaller_feature_set() -> None:
    base = {
        "cv_mean_balanced_accuracy": 0.75,
        "cv_worst_balanced_accuracy": 0.5,
        "cv_std_balanced_accuracy": 0.1,
        "single_scale": 1.0,
        "phase_scale": 0.5,
        "pair_scale": 2.0,
    }
    rows = [
        {**base, "selected_feature_count": 16},
        {**base, "selected_feature_count": 8},
    ]

    assert sorted(rows, key=tuner._config_key)[0]["selected_feature_count"] == 8


def test_source_has_no_provider_or_hardware_path() -> None:
    source = Path(tuner.__file__).read_text(encoding="utf-8")
    assert "validate_fireopal_batch" not in source
    assert "fireopal.execute" not in source
    assert '"--validate"' not in source
    assert "QiskitRuntimeService" not in source
    assert "provider_calls\": []" in source
