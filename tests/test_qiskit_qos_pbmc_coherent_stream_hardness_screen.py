from __future__ import annotations

import inspect

import numpy as np
from scipy import sparse

import qiskit_qos_pbmc_coherent_stream_hardness_screen as screen
import qiskit_qos_realdata_projected_kernel_gate as projected


def test_grid_panel_is_hardware_compatible_and_has_frozen_counts() -> None:
    q40 = screen.grid_panel_summary(40)
    q60 = screen.grid_panel_summary(60)

    assert q40["observable_count"] == 405
    assert q40["grid_rows"] == 5
    assert q40["grid_columns"] == 8
    assert q40["grid_edges"] == 67
    assert q40["grid_plaquettes"] == 28
    assert q60["observable_count"] == 627
    assert q60["grid_rows"] == 6
    assert q60["grid_columns"] == 10
    assert q60["grid_edges"] == 104
    assert q60["grid_plaquettes"] == 45
    for summary in (q40, q60):
        assert summary["all_homogeneous_xyz"] is True
        assert summary["measurement_circuits_per_sample"] == 3
        assert summary["largest_observable_support"] == 4
        assert summary["support_size_counts"].keys() == {"1", "2", "4"}


def test_coherent_stream_depth_and_no_midcircuit_operations() -> None:
    for block_count in (1, 2, 3, 4):
        blocks = np.asarray(
            [
                np.sin(np.linspace(-0.8, 0.7, 40) + 0.2 * block_index)
                for block_index in range(block_count)
            ]
        )
        circuit = screen.coherent_stream_circuit(blocks)

        assert circuit.depth() == screen.expected_logical_depth(40, block_count)
        assert circuit.depth() == 8 + 4 * (block_count - 1)
        assert circuit.count_ops()["rzz"] == sum(
            len(layer) for layer in screen.block_interaction_layers(40, block_count)
        )
        assert "measure" not in circuit.count_ops()
        assert "reset" not in circuit.count_ops()


def test_b1_is_exactly_the_frozen_grid_mixer_d8_circuit() -> None:
    vector = np.sin(np.linspace(-0.8, 0.7, 40))
    coherent = screen.coherent_stream_circuit(vector[None, :])
    frozen = projected.feature_map_circuit(vector, "grid_mixer_d8")

    assert coherent == frozen


def test_pair_multiplier_only_rescales_rzz_angles() -> None:
    blocks = np.asarray(
        [
            np.sin(np.linspace(-0.8, 0.7, 20) + 0.2 * block_index)
            for block_index in range(4)
        ]
    )
    baseline = screen.coherent_stream_circuit(blocks)
    scaled = screen.coherent_stream_circuit(blocks, pair_multiplier=5.0)
    baseline_angles = [
        float(item.operation.params[0])
        for item in baseline.data
        if item.operation.name == "rzz"
    ]
    scaled_angles = [
        float(item.operation.params[0])
        for item in scaled.data
        if item.operation.name == "rzz"
    ]

    assert baseline.depth() == scaled.depth() == 20
    assert np.allclose(scaled_angles, 5.0 * np.asarray(baseline_angles))


def test_block_hash_preserves_b1_and_normalizes_each_extra_block() -> None:
    x = sparse.csr_matrix(
        np.asarray(
            [
                [2.0, 0.0, 1.0, 3.0, 0.0, 1.0],
                [0.0, 4.0, 1.0, 0.0, 2.0, 3.0],
            ]
        )
    )
    legacy, _ = screen.pairwise.build_pairwise_hashed_matrix(
        x,
        feature_dim=4,
        hash_seed=7,
        value_mode="log-product",
        max_active_genes=6,
    )
    b1, _ = screen.build_coherent_blocks(
        x,
        num_qubits=4,
        block_count=1,
        hash_seed=7,
        value_mode="log-product",
        max_active_genes=6,
    )
    b4, stats = screen.build_coherent_blocks(
        x,
        num_qubits=4,
        block_count=4,
        hash_seed=7,
        value_mode="log-product",
        max_active_genes=6,
    )

    assert np.array_equal(b1[:, 0], legacy.astype(np.float64))
    nonzero_norms = np.linalg.norm(b4, axis=2)
    assert np.allclose(nonzero_norms[nonzero_norms > 0.0], 1.0)
    assert np.max(np.abs(b4)) <= 1.0
    assert stats["ambient_hash_buckets"] == 16


def test_exact_q4_parity_for_shallow_and_deep_streams() -> None:
    rows = screen.exact_small_width_validation(
        (4,), (1, 4), bond_dimension=64
    )

    assert len(rows) == 2
    assert all(row["passed"] for row in rows)
    assert rows[0]["b1_exactly_matches_frozen_grid_mixer_d8"] is True
    assert rows[0]["actual_depth"] == 8
    assert rows[1]["actual_depth"] == screen.expected_logical_depth(4, 4) == 18
    assert max(row["max_abs_mps_minus_statevector"] for row in rows) < 1e-9


def test_structural_gate_requires_width_scaling_not_just_large_chi() -> None:
    def row(width: int, block_count: int, selected: int | None, chi32: bool):
        return {
            "qubits": width,
            "block_count": block_count,
            "mps_convergence": {
                "selected_lower_converged_bond_dimension": selected,
                "reference_bond_dimension": 512,
                "chi32_converged": chi32,
            },
        }

    rows = [
        row(20, 1, 32, True),
        row(30, 1, 32, True),
        row(40, 1, 32, True),
        row(20, 2, 32, True),
        row(30, 2, 64, False),
        row(40, 2, 128, False),
        row(20, 3, 128, False),
        row(30, 3, 128, False),
        row(40, 3, 128, False),
    ]
    result = screen.evaluate_structural_gate(
        rows, widths=(20, 30, 40), block_counts=(1, 2, 3)
    )

    assert result["passed"] is True
    assert result["eligible_block_counts"] == [2]
    assert result["candidates"][2]["passed"] is False


def test_runner_has_no_provider_execution_or_exponential_state_allocation() -> None:
    source = inspect.getsource(screen)

    assert "authenticate_qctrl_account" not in source
    assert ".execute(" not in source
    assert "np.zeros(2 **" not in source
    assert screen.DEFAULT_WIDTHS == (20, 30, 40)
    assert screen.DEFAULT_BLOCK_COUNTS == (1, 2, 3, 4)
    assert screen.DEFAULT_BOND_DIMENSIONS == (16, 32, 64, 128, 256, 512)
