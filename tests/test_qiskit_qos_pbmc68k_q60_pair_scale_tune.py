from __future__ import annotations

import inspect
from pathlib import Path

import pytest

import qiskit_qos_pbmc68k_q60_pair_scale_tune as pair_tune


def _seed_result(seed: int, values: dict[float, float]) -> dict[str, object]:
    return {
        "seed": seed,
        "candidates": [
            {
                "pair_scale": scale,
                "cv_mean_balanced_accuracy": value,
                "cv_worst_balanced_accuracy": value - 0.05,
                "cv_std_balanced_accuracy": 0.02,
            }
            for scale, value in values.items()
        ],
    }


def test_pair_scale_tie_prefers_zero_ablation() -> None:
    rows = [
        _seed_result(11, {0.0: 0.6, 0.95: 0.6, 3.0: 0.55}),
        _seed_result(13, {0.0: 0.5, 0.95: 0.5, 3.0: 0.45}),
    ]

    result = pair_tune.aggregate_pair_scale_cv(
        rows, cv_mean_gate=0.5, cv_worst_seed_gate=0.45
    )

    assert result["chosen"]["pair_scale"] == pytest.approx(0.0)
    assert result["nonzero_strictly_beats_pair_zero"] is False
    assert result["passes_fresh_confirmation_gate"] is False


def test_nonzero_scale_must_beat_zero_and_quality_gates() -> None:
    rows = [
        _seed_result(11, {0.0: 0.55, 0.95: 0.65, 3.0: 0.60}),
        _seed_result(13, {0.0: 0.50, 0.95: 0.60, 3.0: 0.55}),
    ]

    result = pair_tune.aggregate_pair_scale_cv(
        rows, cv_mean_gate=0.60, cv_worst_seed_gate=0.60
    )

    assert result["chosen"]["pair_scale"] == pytest.approx(0.95)
    assert result["chosen_minus_pair_zero_cv_mean"] == pytest.approx(0.10)
    assert result["passes_fresh_confirmation_gate"] is True


def test_numerical_roundoff_is_not_an_entangling_win() -> None:
    rows = [
        _seed_result(11, {0.0: 0.6, 0.95: 0.6000000000000001}),
        _seed_result(13, {0.0: 0.5, 0.95: 0.5000000000000001}),
    ]

    result = pair_tune.aggregate_pair_scale_cv(
        rows, cv_mean_gate=0.5, cv_worst_seed_gate=0.45
    )

    assert result["chosen"]["pair_scale"] == pytest.approx(0.0)
    assert result["passes_fresh_confirmation_gate"] is False


def test_pair_scale_cv_api_has_no_test_inputs() -> None:
    parameters = inspect.signature(pair_tune.cross_validate_pair_scales).parameters
    assert "encoded_test" not in parameters
    assert "y_test" not in parameters


def test_source_has_no_provider_or_hardware_path() -> None:
    source = Path(pair_tune.__file__).read_text(encoding="utf-8")
    assert "validate_fireopal_batch" not in source
    assert "fireopal.execute" not in source
    assert '"--validate"' not in source
    assert "QiskitRuntimeService" not in source
    assert '"provider_calls": []' in source
