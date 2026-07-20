from __future__ import annotations

import inspect

import numpy as np

import qiskit_qos_pbmc68k_q60_reuploading_tune as reupload
import qiskit_qos_pbmc68k_q60_shallow_fireopal_validate as q60


def _candidate(name: str, score: float, zero: float) -> dict:
    axis, scale = reupload.architecture_parameters(name)
    return {
        "architecture": name,
        "post_axis": axis,
        "post_scale": scale,
        "logical_circuit_depth": 5 if name == "none" else 6,
        "cv_mean_balanced_accuracy": score,
        "cv_worst_balanced_accuracy": score - 0.05,
        "cv_std_balanced_accuracy": 0.02,
        "same_observables_pair_zero_cv_mean": zero,
        "entangler_gain_cv_mean": score - zero,
    }


def _seed(seed: int, candidates: list[dict]) -> dict:
    return {
        "seed": seed,
        "candidates": candidates,
        "folds": [
            {
                "architectures": [
                    {
                        "architecture": row["architecture"],
                        "selection_audit": {"selected_pair_sensitive_count": 2},
                    }
                    for row in candidates
                ]
            }
        ],
    }


def test_none_limit_matches_existing_exact_expectation() -> None:
    linear = np.asarray([0.2, -0.4, 0.7, -0.1])
    pair = linear[:-1] * linear[1:]
    mappings = [{0: "X"}, {2: "Y"}, {1: "Z", 2: "Z"}]
    for mapping in mappings:
        expected = q60.exact_local_expectation(
            linear,
            pair,
            mapping,
            single_scale=0.75,
            phase_scale=0.25,
            pair_scale=0.95,
        )
        actual = reupload.exact_local_reupload_expectation(
            linear,
            pair,
            mapping,
            single_scale=0.75,
            phase_scale=0.25,
            pair_scale=0.95,
            post_axis="none",
            post_scale=0.0,
        )
        assert np.isclose(actual, expected, atol=1e-12, rtol=0.0)


def test_pair_zero_limit_removes_pair_value_dependence() -> None:
    linear = np.asarray([0.3, -0.5, 0.8])
    pair = np.asarray([0.9, -0.7])
    for axis, scale in (("rx", 0.5), ("ry", 1.0)):
        with_pair = reupload.exact_local_reupload_expectation(
            linear,
            pair,
            {1: "X"},
            single_scale=0.75,
            phase_scale=0.25,
            pair_scale=0.0,
            post_axis=axis,
            post_scale=scale,
        )
        without_pair = reupload.exact_local_reupload_expectation(
            linear,
            np.zeros_like(pair),
            {1: "X"},
            single_scale=0.75,
            phase_scale=0.25,
            pair_scale=0.0,
            post_axis=axis,
            post_scale=scale,
        )
        assert np.isclose(with_pair, without_pair, atol=1e-12, rtol=0.0)


def test_gate_requires_positive_entangler_gain_on_every_seed() -> None:
    rows = [
        _seed(11, [_candidate("none", 0.55, 0.56), _candidate("ry_0p5", 0.64, 0.58)]),
        _seed(13, [_candidate("none", 0.56, 0.57), _candidate("ry_0p5", 0.63, 0.59)]),
        _seed(17, [_candidate("none", 0.57, 0.58), _candidate("ry_0p5", 0.62, 0.63)]),
    ]
    result = reupload.aggregate_reuploading_cv(
        rows,
        cv_mean_gate=0.55,
        cv_worst_seed_gate=0.45,
        expected_pair_sensitive=2,
    )
    assert result["chosen"]["architecture"] == "ry_0p5"
    assert result["chosen_positive_entangler_gain_every_seed"] is False
    assert result["passes_fresh_confirmation_gate"] is False


def test_gate_opens_for_stable_nontrivial_reuploading_gain() -> None:
    rows = [
        _seed(11, [_candidate("none", 0.55, 0.56), _candidate("rx_1p0", 0.64, 0.58)]),
        _seed(13, [_candidate("none", 0.56, 0.57), _candidate("rx_1p0", 0.63, 0.59)]),
        _seed(17, [_candidate("none", 0.57, 0.58), _candidate("rx_1p0", 0.62, 0.60)]),
    ]
    result = reupload.aggregate_reuploading_cv(
        rows,
        cv_mean_gate=0.55,
        cv_worst_seed_gate=0.45,
        expected_pair_sensitive=2,
    )
    assert result["chosen"]["architecture"] == "rx_1p0"
    assert result["chosen_positive_entangler_gain_every_seed"] is True
    assert result["chosen_strictly_beats_strongest_pair_zero"] is True
    assert result["passes_fresh_confirmation_gate"] is True


def test_roundoff_is_not_a_positive_all_seed_gain() -> None:
    rows = [
        _seed(11, [_candidate("none", 0.55, 0.56), _candidate("ry_1p0", 0.62, 0.60)]),
        _seed(13, [_candidate("none", 0.56, 0.57), _candidate("ry_1p0", 0.61, 0.59)]),
        _seed(17, [_candidate("none", 0.57, 0.58), _candidate("ry_1p0", 0.6000000000001, 0.60)]),
    ]
    result = reupload.aggregate_reuploading_cv(
        rows,
        cv_mean_gate=0.55,
        cv_worst_seed_gate=0.45,
        expected_pair_sensitive=2,
    )
    assert result["chosen_positive_entangler_gain_every_seed"] is False
    assert result["passes_fresh_confirmation_gate"] is False


def test_training_api_and_source_have_no_provider_path() -> None:
    signature = inspect.signature(reupload.cross_validate_reuploading)
    assert "encoded_test" not in signature.parameters
    assert "y_test" not in signature.parameters
    source = inspect.getsource(reupload)
    assert "QiskitRuntimeService" not in source
    assert "authenticate_qctrl_account" not in source
    assert "execute_with_qctrl" not in source
