from __future__ import annotations

import inspect

import numpy as np

import qiskit_qos_pbmc_width_scaled_entangler_screen as screen


def test_pair_multiplier_scale_laws_are_frozen() -> None:
    assert screen.pair_multiplier("legacy", 40) == 1.0
    assert screen.pair_multiplier("sqrt_q", 40) == np.sqrt(40)
    assert screen.pair_multiplier("q_over_4", 40) == 10.0
    assert screen.pair_multiplier("q_over_2", 40) == 20.0
    assert screen.pair_multiplier("q", 40) == 40.0


def test_rzz_angle_statistics_scale_linearly_and_remain_finite() -> None:
    blocks = np.full((2, 4, 20), 1.0 / np.sqrt(20), dtype=np.float64)
    legacy = screen.rzz_angle_statistics(blocks, scale_law="legacy")
    scaled = screen.rzz_angle_statistics(blocks, scale_law="q")

    assert legacy["all_finite"] is True
    assert scaled["all_finite"] is True
    assert np.isclose(
        scaled["absolute_angle_quantiles"]["median"],
        20.0 * legacy["absolute_angle_quantiles"]["median"],
    )
    assert np.isclose(scaled["absolute_angle_quantiles"]["median"], 0.95)
    assert scaled["fraction_abs_above_pi"] == 0.0


def test_exact_scaled_q4_validation() -> None:
    rows = screen.exact_scaled_validation((4,), ("legacy", "q"))

    assert len(rows) == 2
    assert all(row["passed"] for row in rows)
    assert all(row["actual_depth"] == 18 for row in rows)
    assert max(row["max_abs_mps_minus_statevector"] for row in rows) < 1e-9


def _screen_row(
    width: int,
    scale_law: str,
    selected: int | None,
    chi32: bool,
) -> dict:
    return {
        "qubits": width,
        "scale_law": scale_law,
        "rzz_angle_statistics": {"fraction_abs_above_pi": 0.0},
        "mps_convergence": {
            "selected_lower_converged_bond_dimension": selected,
            "reference_bond_dimension": 256,
            "chi32_converged": chi32,
        },
    }


def test_preflight_advances_at_most_two_mildest_scaling_candidates() -> None:
    rows = []
    for law in ("sqrt_q", "q_over_4", "q_over_2"):
        rows.extend(
            [
                _screen_row(20, law, 32, True),
                _screen_row(30, law, 64, False),
                _screen_row(40, law, 128, False),
            ]
        )
    result = screen.evaluate_preflight_funnel(
        rows,
        widths=(20, 30, 40),
        scale_laws=("sqrt_q", "q_over_4", "q_over_2"),
        maximum_candidates=2,
    )

    assert result["survivors"] == ["sqrt_q", "q_over_4"]


def test_confirmation_gate_requires_fourfold_width_growth() -> None:
    passing = [
        _screen_row(20, "q_over_4", 32, True),
        _screen_row(30, "q_over_4", 64, False),
        _screen_row(40, "q_over_4", 128, False),
    ]
    flat = [
        _screen_row(20, "q_over_2", 128, False),
        _screen_row(30, "q_over_2", 128, False),
        _screen_row(40, "q_over_2", 128, False),
    ]
    for row in [*passing, *flat]:
        row["mps_convergence"]["reference_bond_dimension"] = 512
    result = screen.evaluate_confirmation_gate(
        [*passing, *flat],
        widths=(20, 30, 40),
        scale_laws=("q_over_4", "q_over_2"),
    )

    assert result["passed"] is True
    assert result["selected_scale_law"] == "q_over_4"
    assert result["eligible_scale_laws"] == ["q_over_4"]


def test_runner_is_local_and_keeps_depth_fixed() -> None:
    source = inspect.getsource(screen)

    assert "authenticate_qctrl_account" not in source
    assert ".execute(" not in source
    assert screen.DEFAULT_BLOCK_COUNT == 4
    assert screen.SCALE_LAWS == (
        "legacy",
        "sqrt_q",
        "q_over_4",
        "q_over_2",
        "q",
    )
