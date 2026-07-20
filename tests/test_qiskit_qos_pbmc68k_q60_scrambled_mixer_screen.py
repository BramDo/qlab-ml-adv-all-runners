from __future__ import annotations

import ast
import inspect

import numpy as np
import pytest
from qiskit.quantum_info import Statevector

import qiskit_qos_pbmc68k_q60_scrambled_mixer_screen as screen
import qiskit_qos_toy_model as toy


def test_architecture_depths_and_noncommuting_growth() -> None:
    values = np.linspace(-0.8, 0.9, 60)
    circuits = {
        name: screen.architecture_circuit(values, name)
        for name in screen.ARCHITECTURES
    }
    assert circuits["control_rx05_d6"].depth() == 6
    assert circuits["grid_mixer_d12"].depth() == 12
    assert circuits["scrambled_mixer_d16"].depth() == 16
    assert circuits["control_rx05_d6"].count_ops()["rzz"] == 59
    assert circuits["grid_mixer_d12"].count_ops()["rzz"] > 59
    assert circuits["scrambled_mixer_d16"].count_ops()["rzz"] > circuits["grid_mixer_d12"].count_ops()["rzz"]


@pytest.mark.parametrize("architecture", screen.ARCHITECTURES)
def test_every_interaction_layer_is_a_matching(architecture: str) -> None:
    for layer in screen.interaction_layers(architecture, 60):
        flattened = [qubit for edge in layer for qubit in edge]
        assert len(flattened) == len(set(flattened))


def test_hardness_proxies_expand_beyond_commuting_control() -> None:
    mappings = toy.pauli_feature_mappings(60, family="local")
    control = screen.structural_hardness("control_rx05_d6", mappings, 60)
    grid = screen.structural_hardness("grid_mixer_d12", mappings, 60)
    scrambled = screen.structural_hardness("scrambled_mixer_d16", mappings, 60)

    assert control["causal_cone"]["maximum"] == 4
    assert control["interaction_graph_treewidth_min_fill_upper_bound"] == 1
    assert grid["causal_cone"]["median"] >= 16
    assert grid["interaction_graph_treewidth_min_fill_upper_bound"] > 1
    assert scrambled["causal_cone"]["median"] > grid["causal_cone"]["median"]
    assert (
        scrambled["interaction_graph_treewidth_min_fill_upper_bound"]
        > grid["interaction_graph_treewidth_min_fill_upper_bound"]
    )


@pytest.mark.parametrize("architecture", screen.ARCHITECTURES)
def test_small_q_mps_matches_statevector(architecture: str) -> None:
    parity = screen.small_q_parity(architecture, num_qubits=8)
    assert parity["passed"] is True
    assert parity["statevector_norm"] == pytest.approx(1.0, abs=1e-12)
    assert parity["max_abs_mps64_minus_statevector"] <= 1e-9


def test_architecture_circuits_are_unitary_and_finite() -> None:
    values = np.linspace(-0.7, 0.6, 8)
    for architecture in screen.ARCHITECTURES:
        state = Statevector.from_instruction(
            screen.architecture_circuit(values, architecture)
        )
        assert np.all(np.isfinite(state.data))
        assert np.linalg.norm(state.data) == pytest.approx(1.0, abs=1e-12)


def test_runner_is_strictly_local_and_test_is_not_used_for_selection() -> None:
    source = inspect.getsource(screen)
    tree = ast.parse(source)
    imported_modules = {
        alias.name
        for node in ast.walk(tree)
        if isinstance(node, (ast.Import, ast.ImportFrom))
        for alias in node.names
    }
    assert "fireopal" not in imported_modules
    assert ".execute(" not in source
    assert ".get_result(" not in source
    assert '"provider_calls": []' in source
    assert '"execution_attempted": False' in source
    selection_source = inspect.getsource(screen.run_screen)
    winner_prefix = selection_source.split("final_winner_evaluation", maxsplit=1)[0]
    assert "encoded_test" not in winner_prefix
    assert "y_test" not in winner_prefix


def test_cli_defaults_to_all_three_local_candidates() -> None:
    args = screen.build_parser().parse_args([])
    assert args.architectures == screen.ARCHITECTURES
    assert args.bond_dimension == 64
    assert args.probe_bond_dimensions == (32, 128)
