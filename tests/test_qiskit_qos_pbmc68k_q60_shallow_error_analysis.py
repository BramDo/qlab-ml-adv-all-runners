from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

import qiskit_qos_pbmc68k_q60_shallow_error_analysis as analysis


def test_exact_mcnemar_known_values() -> None:
    assert analysis.exact_mcnemar_p_value(0, 0) == pytest.approx(1.0)
    assert analysis.exact_mcnemar_p_value(1, 0) == pytest.approx(1.0)
    assert analysis.exact_mcnemar_p_value(10, 0) == pytest.approx(0.001953125)
    assert analysis.exact_mcnemar_p_value(7, 3) == pytest.approx(0.34375)


def test_paired_outcomes_partition_every_sample() -> None:
    labels = np.asarray([-1.0, -1.0, 1.0, 1.0])
    q60_scores = np.asarray([-2.0, 1.0, 2.0, -1.0])
    classical_scores = np.asarray([-2.0, -1.0, -2.0, -1.0])

    counts = analysis.paired_outcome_counts(labels, q60_scores, classical_scores)

    assert counts == {
        "both_correct": 1,
        "q60_only_correct": 1,
        "classical_only_correct": 1,
        "both_wrong": 1,
    }


def test_paired_bootstrap_is_deterministic_and_paired() -> None:
    labels = np.asarray([-1.0, -1.0, 1.0, 1.0])
    q60_scores = np.asarray([-2.0, -1.0, 2.0, -1.0])
    classical_scores = np.asarray([2.0, 1.0, -2.0, 1.0])

    first = analysis.paired_stratified_bootstrap(
        labels, q60_scores, classical_scores, seed=77, replicates=200
    )
    second = analysis.paired_stratified_bootstrap(
        labels, q60_scores, classical_scores, seed=77, replicates=200
    )

    assert first == second
    assert first["point_difference"] == pytest.approx(0.5)
    assert first["probability_q60_strictly_better"] == pytest.approx(0.75)


def test_markdown_preserves_post_hoc_claim_boundary() -> None:
    report = {
        "provenance": {"actual_seed": 24},
        "paired_performance": {
            "q60_balanced_accuracy": 0.6,
            "classical_balanced_accuracy": 0.7,
            "q60_minus_classical_balanced_accuracy": -0.1,
            "exact_mcnemar_two_sided_p": 0.5,
            "paired_stratified_bootstrap": {"lower_95": -0.3, "upper_95": 0.1},
            "outcomes": {
                "both_correct": 20,
                "q60_only_correct": 5,
                "classical_only_correct": 10,
                "both_wrong": 13,
            },
        },
        "representation_audit": {
            "selected_measurement_basis_counts": {"Z": 23, "Y": 1},
            "selected_multiqubit_observables": 0,
            "selected_features_numerically_sensitive_to_pair_scale": 1,
            "selected_observables": 24,
            "pair_scale_zero_ablation": {
                "test_balanced_accuracy": 0.6,
                "delta_from_frozen_q60": 0.0,
            },
        },
        "claim_boundary": "Post-hoc diagnostic only.",
    }

    markdown = analysis.render_markdown(report)

    assert "Post-hoc diagnostic only" in markdown
    assert "No hardware step is justified" in markdown


def test_source_has_no_provider_or_hardware_path() -> None:
    source = Path(analysis.__file__).read_text(encoding="utf-8")
    assert "validate_fireopal_batch" not in source
    assert "fireopal.execute" not in source
    assert '"--validate"' not in source
    assert "QiskitRuntimeService" not in source
    assert '"provider_calls": []' in source
