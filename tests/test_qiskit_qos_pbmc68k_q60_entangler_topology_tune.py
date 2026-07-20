from __future__ import annotations

import inspect

import numpy as np

import qiskit_qos_pbmc68k_q60_entangler_topology_tune as topology


def _seed_result(seed: int, candidates: list[dict[str, float | str]]) -> dict:
    fold_topologies = []
    for candidate in candidates:
        fold_topologies.append(
            {
                "topology": candidate["topology"],
                "selection_audit": {"selected_pair_sensitive_count": 2},
            }
        )
    return {
        "seed": seed,
        "candidates": candidates,
        "folds": [{"topologies": fold_topologies}],
    }


def _candidate(name: str, score: float, zero: float) -> dict[str, float | str]:
    return {
        "topology": name,
        "cv_mean_balanced_accuracy": score,
        "cv_worst_balanced_accuracy": score - 0.05,
        "cv_std_balanced_accuracy": 0.02,
        "same_observables_pair_zero_cv_mean": zero,
        "topology_minus_pair_zero_cv_mean": score - zero,
        "active_edge_count_mean": 2.0,
        "logical_entangler_depth": 1 if name != "full_path" else 2,
    }


def test_fixed_masks_have_expected_depth_structure() -> None:
    full = topology.fixed_topology_mask("full_path", 6)
    even = topology.fixed_topology_mask("even_matching", 6)
    odd = topology.fixed_topology_mask("odd_matching", 6)
    assert full.tolist() == [1.0] * 5
    assert even.tolist() == [1.0, 0.0, 1.0, 0.0, 1.0]
    assert odd.tolist() == [0.0, 1.0, 0.0, 1.0, 0.0]
    topology._validate_matching_mask(even)
    topology._validate_matching_mask(odd)


def test_maximum_weight_matching_is_exact_and_disjoint() -> None:
    encoded = np.asarray(
        [
            [1.0, 1.0, 1.0, 1.0],
            [1.0, 1.0, -1.0, -1.0],
            [1.0, -1.0, -1.0, 1.0],
            [1.0, -1.0, 1.0, -1.0],
        ]
    )
    labels = np.asarray([1.0, 1.0, -1.0, -1.0])
    mask, _ = topology.maximum_weight_path_matching(encoded, labels)
    topology._validate_matching_mask(mask)
    active = np.flatnonzero(mask > 0.5)
    assert not np.any(np.diff(active) == 1)


def test_aggregate_requires_winner_to_beat_all_zero_ablations() -> None:
    candidates_a = [
        _candidate("full_path", 0.61, 0.62),
        _candidate("supervised_matching", 0.60, 0.55),
    ]
    candidates_b = [
        _candidate("full_path", 0.63, 0.64),
        _candidate("supervised_matching", 0.62, 0.57),
    ]
    result = topology.aggregate_topology_cv(
        [_seed_result(1, candidates_a), _seed_result(2, candidates_b)],
        cv_mean_gate=0.55,
        cv_worst_seed_gate=0.45,
        expected_pair_sensitive=2,
    )
    assert result["chosen"]["topology"] == "full_path"
    assert result["chosen_strictly_beats_own_pair_zero"] is False
    assert result["passes_fresh_confirmation_gate"] is False


def test_aggregate_opens_gate_for_strict_global_entangler_gain() -> None:
    candidates_a = [
        _candidate("full_path", 0.57, 0.56),
        _candidate("supervised_matching", 0.64, 0.58),
    ]
    candidates_b = [
        _candidate("full_path", 0.58, 0.57),
        _candidate("supervised_matching", 0.62, 0.57),
    ]
    result = topology.aggregate_topology_cv(
        [_seed_result(1, candidates_a), _seed_result(2, candidates_b)],
        cv_mean_gate=0.55,
        cv_worst_seed_gate=0.45,
        expected_pair_sensitive=2,
    )
    assert result["chosen"]["topology"] == "supervised_matching"
    assert result["chosen_strictly_beats_own_pair_zero"] is True
    assert result["chosen_strictly_beats_strongest_pair_zero"] is True
    assert result["passes_fresh_confirmation_gate"] is True


def test_roundoff_does_not_count_as_entangler_gain() -> None:
    candidates = [
        _candidate("full_path", 0.6000000000001, 0.6),
        _candidate("supervised_matching", 0.59, 0.58),
    ]
    result = topology.aggregate_topology_cv(
        [_seed_result(1, candidates), _seed_result(2, candidates)],
        cv_mean_gate=0.55,
        cv_worst_seed_gate=0.45,
        expected_pair_sensitive=2,
    )
    assert result["chosen_strictly_beats_own_pair_zero"] is False
    assert result["passes_fresh_confirmation_gate"] is False


def test_training_api_and_source_have_no_provider_path() -> None:
    signature = inspect.signature(topology.cross_validate_topologies)
    assert "encoded_test" not in signature.parameters
    assert "y_test" not in signature.parameters
    source = inspect.getsource(topology)
    assert "QiskitRuntimeService" not in source
    assert "authenticate_qctrl_account" not in source
    assert "execute_with_qctrl" not in source
