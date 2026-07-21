from __future__ import annotations

import inspect
from pathlib import Path

import numpy as np
import pytest

import qiskit_qos_pbmc68k_q60_balanced_representation as balanced


def _synthetic_selection_inputs() -> tuple[
    np.ndarray, np.ndarray, np.ndarray, list[dict[int, str]]
]:
    mappings = [
        {0: "Z"},
        {0: "X"},
        {0: "Y"},
        {0: "X", 1: "X"},
        {0: "Z", 1: "Z"},
    ]
    labels = np.asarray([-1.0, -1.0, 1.0, 1.0])
    pair_zero = np.asarray(
        [
            [-0.7, -0.4, -0.3, -0.2, -0.1],
            [-0.5, -0.2, -0.1, -0.1, -0.2],
            [0.5, 0.2, 0.1, 0.1, 0.2],
            [0.7, 0.4, 0.3, 0.2, 0.1],
        ]
    )
    original = pair_zero.copy()
    original[:, 1] += np.asarray([0.2, -0.2, 0.2, -0.2])
    original[:, 2] += np.asarray([0.3, -0.3, 0.3, -0.3])
    original[:, 3] += np.asarray([0.4, -0.4, 0.4, -0.4])
    return original, pair_zero, labels, mappings


def test_balanced_selector_enforces_all_three_categories() -> None:
    original, pair_zero, labels, mappings = _synthetic_selection_inputs()

    selected, _, audit = balanced.select_balanced_train_only_features(
        original,
        pair_zero,
        labels,
        mappings=mappings,
        z_quota=1,
        transverse_quota=1,
        multiqubit_quota=1,
        shot_intent=1024,
        sensitivity_threshold=1e-10,
    )

    assert len(selected) == 3
    assert selected[0] == 0
    assert selected[1] in (1, 2)
    assert selected[2] == 3
    assert audit["selected_counts"] == {
        "single_z": 1,
        "pair_sensitive_local_xy": 1,
        "pair_sensitive_multiqubit": 1,
    }
    assert audit["selected_pair_sensitive_count"] == 2
    assert audit["test_inputs_seen"] is False


def test_pair_insensitive_multiqubit_observable_cannot_fill_quota() -> None:
    original, pair_zero, labels, mappings = _synthetic_selection_inputs()
    original[:, 3] = pair_zero[:, 3]

    with pytest.raises(balanced.RunnerError, match="multiqubit"):
        balanced.select_balanced_train_only_features(
            original,
            pair_zero,
            labels,
            mappings=mappings,
            z_quota=1,
            transverse_quota=1,
            multiqubit_quota=1,
            shot_intent=1024,
            sensitivity_threshold=1e-10,
        )


def test_selection_api_has_no_test_inputs_or_labels() -> None:
    parameters = inspect.signature(
        balanced.select_balanced_train_only_features
    ).parameters
    assert "query_test" not in parameters
    assert "y_test" not in parameters
    assert "encoded_test" not in parameters


def test_aggregate_gate_requires_structure_and_training_cv() -> None:
    fold = {
        "selection_audit": {"selected_pair_sensitive_count": 2},
    }
    rows = [
        {"balanced_cv_mean": 0.6, "legacy_cv_mean": 0.55, "folds": [fold]},
        {"balanced_cv_mean": 0.5, "legacy_cv_mean": 0.55, "folds": [fold]},
    ]

    result = balanced._aggregate_cv(
        rows,
        cv_mean_gate=0.55,
        cv_worst_seed_gate=0.5,
        expected_pair_sensitive=2,
    )

    assert result["passes_confirmation_gate"] is True
    broken = [
        rows[0],
        {
            **rows[1],
            "folds": [
                {"selection_audit": {"selected_pair_sensitive_count": 1}}
            ],
        },
    ]
    assert (
        balanced._aggregate_cv(
            broken,
            cv_mean_gate=0.55,
            cv_worst_seed_gate=0.5,
            expected_pair_sensitive=2,
        )["passes_confirmation_gate"]
        is False
    )


def test_source_has_no_provider_or_hardware_path() -> None:
    source = Path(balanced.__file__).read_text(encoding="utf-8")
    assert "validate_fireopal_batch" not in source
    assert "fireopal.execute" not in source
    assert '"--validate"' not in source
    assert "QiskitRuntimeService" not in source
    assert '"provider_calls": []' in source
