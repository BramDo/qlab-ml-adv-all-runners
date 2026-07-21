from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import pytest

import qiskit_qos_pbmc68k_q60_shallow_cross_seed_freeze as freeze


def _candidate(
    *,
    single: float,
    count: int,
    scores: tuple[float, ...],
    pair: float = 2.0,
) -> dict[str, object]:
    mean = sum(scores) / len(scores)
    variance = sum((value - mean) ** 2 for value in scores) / len(scores)
    return {
        "single_scale": single,
        "phase_scale": 0.5,
        "pair_scale": pair,
        "selected_feature_count": count,
        "cv_mean_balanced_accuracy": mean,
        "cv_worst_balanced_accuracy": min(scores),
        "cv_std_balanced_accuracy": variance**0.5,
        "folds": [
            {"fold_index": index, "balanced_accuracy": score}
            for index, score in enumerate(scores)
        ],
    }


def _report(seed: int, leaderboard: list[dict[str, object]]) -> dict[str, object]:
    return {
        "config": {"seed": seed},
        "split": {"encoded_train_sha256": f"train-{seed}"},
        "train_only_tuning": {
            "selection_scope": "training_split_only",
            "test_inputs_seen": False,
            "test_labels_seen": False,
            "training_samples": 24,
            "cv_folds": 2,
            "candidate_configurations": len(leaderboard),
            "leaderboard": leaderboard,
        },
        "final_evaluation": {
            "tuned": {"test_balanced_accuracy": 0.0},
        },
    }


def test_cross_seed_selection_uses_equal_seed_mean_and_robust_tie_break() -> None:
    small = _candidate(single=1.0, count=8, scores=(0.7, 0.7))
    large = _candidate(single=1.0, count=16, scores=(0.7, 0.7))
    reports = [
        _report(11, [small, large]),
        _report(13, [deepcopy(small), deepcopy(large)]),
    ]
    runs = [
        freeze.extract_training_only_run(
            report, source_path=f"seed-{report['config']['seed']}", source_sha256="x"
        )
        for report in reports
    ]

    selected = freeze.select_cross_seed_configuration(runs)

    assert selected["source_test_inputs_used"] is False
    assert selected["confirmation_inputs_seen"] is False
    assert selected["chosen"]["selected_feature_count"] == 8
    assert selected["chosen"]["aggregate_cv_mean_balanced_accuracy"] == pytest.approx(
        0.7
    )


def test_numerical_roundoff_cannot_override_parameter_tie_break() -> None:
    simple = _candidate(single=0.75, pair=0.95, count=24, scores=(0.7, 0.7))
    roundoff_higher = _candidate(
        single=0.75,
        pair=3.0,
        count=24,
        scores=(0.7000000000000001, 0.7000000000000001),
    )
    reports = [
        _report(11, [simple, roundoff_higher]),
        _report(13, [deepcopy(simple), deepcopy(roundoff_higher)]),
    ]
    runs = [
        freeze.extract_training_only_run(
            report, source_path=str(index), source_sha256="x"
        )
        for index, report in enumerate(reports)
    ]

    selected = freeze.select_cross_seed_configuration(runs)

    assert selected["chosen"]["pair_scale"] == pytest.approx(0.95)


def test_source_test_results_cannot_change_frozen_configuration() -> None:
    candidates = [
        _candidate(single=0.5, count=8, scores=(0.75, 0.75)),
        _candidate(single=1.5, count=8, scores=(0.5, 0.5)),
    ]
    reports = [_report(11, candidates), _report(13, deepcopy(candidates))]
    poisoned = deepcopy(reports)
    poisoned[0]["final_evaluation"]["tuned"]["test_balanced_accuracy"] = 1.0
    poisoned[1]["final_evaluation"]["tuned"]["test_balanced_accuracy"] = -99.0

    def choose(source: list[dict[str, object]]) -> tuple[float, float, float, int]:
        runs = [
            freeze.extract_training_only_run(
                report, source_path=str(index), source_sha256="x"
            )
            for index, report in enumerate(source)
        ]
        return freeze._configuration_key(
            freeze.select_cross_seed_configuration(runs)["chosen"]
        )

    assert choose(reports) == choose(poisoned)


def test_mismatched_candidate_grids_are_rejected() -> None:
    report_a = _report(11, [_candidate(single=0.5, count=8, scores=(0.7, 0.6))])
    report_b = _report(13, [_candidate(single=1.5, count=8, scores=(0.7, 0.6))])
    runs = [
        freeze.extract_training_only_run(
            report, source_path=str(index), source_sha256="x"
        )
        for index, report in enumerate((report_a, report_b))
    ]

    with pytest.raises(freeze.RunnerError, match="same candidate grid"):
        freeze.select_cross_seed_configuration(runs)


def test_inconsistent_stored_fold_summary_is_rejected() -> None:
    row = _candidate(single=1.0, count=8, scores=(0.5, 0.75))
    row["cv_mean_balanced_accuracy"] = 0.99
    report = _report(11, [row])

    with pytest.raises(freeze.RunnerError, match="inconsistent with fold scores"):
        freeze.extract_training_only_run(
            report, source_path="bad", source_sha256="x"
        )


def test_source_has_no_provider_or_hardware_path() -> None:
    source = Path(freeze.__file__).read_text(encoding="utf-8")
    assert "validate_fireopal_batch" not in source
    assert "fireopal.execute" not in source
    assert '"--validate"' not in source
    assert "QiskitRuntimeService" not in source
    assert '"provider_calls": []' in source
