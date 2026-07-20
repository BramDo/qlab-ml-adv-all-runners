from __future__ import annotations

import ast
import inspect

import qiskit_qos_pbmc68k_q60_multiscale_readout_screen as screen
import qiskit_qos_pbmc68k_q60_shallow_fireopal_validate as q60


def test_panels_are_nested_unique_and_expand_support() -> None:
    mappings, panels = screen.build_multiscale_panels()

    assert tuple(panels) == screen.PANEL_NAMES
    assert panels["local"] == list(range(len(panels["local"])))
    assert panels["multiscale_pairs"][: len(panels["local"])] == panels["local"]
    assert (
        panels["multiscale_strings"][: len(panels["multiscale_pairs"])]
        == panels["multiscale_pairs"]
    )
    assert len(panels["local"]) < len(panels["multiscale_pairs"])
    assert len(panels["multiscale_pairs"]) < len(panels["multiscale_strings"])
    assert panels["multiscale_strings"] == list(range(len(mappings)))

    keys = [screen._mapping_key(mapping) for mapping in mappings]
    assert len(keys) == len(set(keys))
    assert {len(mapping) for mapping in mappings} == {1, 2, 4, 8}


def test_every_observable_uses_an_existing_global_measurement_basis() -> None:
    mappings, panels = screen.build_multiscale_panels()

    for panel_name, indices in panels.items():
        summary = screen.panel_summary(mappings, indices)
        assert summary["all_homogeneous_xyz"] is True
        assert summary["compatible_with_existing_global_xyz_measurements"] is True
        for index in indices:
            mapping = mappings[index]
            assert set(mapping.values()) in ({"X"}, {"Y"}, {"Z"})
            assert q60.measurement_basis_for_mapping(mapping) in {"X", "Y", "Z"}


def test_multiscale_observables_expand_grid_causal_cones() -> None:
    mappings, panels = screen.build_multiscale_panels()
    local = screen.architecture_screen.structural_hardness(
        screen.ARCHITECTURE,
        [mappings[index] for index in panels["local"]],
        60,
    )
    multiscale = screen.architecture_screen.structural_hardness(
        screen.ARCHITECTURE,
        [mappings[index] for index in panels["multiscale_strings"]],
        60,
    )

    assert multiscale["causal_cone"]["maximum"] >= local["causal_cone"]["maximum"]
    assert multiscale["causal_cone"]["median"] >= local["causal_cone"]["median"]


def test_runner_is_local_only_and_panel_selection_cannot_see_fixed_test() -> None:
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

    panel_cv_source = inspect.getsource(screen._panel_cv)
    assert "encoded_test" not in panel_cv_source
    assert "y_test" not in panel_cv_source

    run_source = inspect.getsource(screen.run_screen)
    selection_prefix = run_source.split("final = _final_panel_evaluation", maxsplit=1)[0]
    assert "encoded_test" not in selection_prefix
    assert "y_test" not in selection_prefix


def test_cli_defaults_are_bounded_and_provider_free() -> None:
    args = screen.build_parser().parse_args([])

    assert args.bond_dimension == 64
    assert args.probe_bond_dimensions == (32, 128)
    assert args.selected_features == 24
    assert args.shot_intent == 128
    assert args.cv_splits == 4
    assert args.cv_repeats == 5
    assert args.shot_noise_replicates == 500
    assert not hasattr(args, "backend")
    assert not hasattr(args, "api_key")
