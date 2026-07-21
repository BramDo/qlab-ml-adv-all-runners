from __future__ import annotations

import ast
import inspect

import qiskit_qos_pbmc68k_q60_module_fireopal_validate as validate


def test_phase_scopes_fix_circuit_shot_batch_and_quantum_second_budgets() -> None:
    sentinel = validate.phase_scope("sentinel")
    large = validate.phase_scope("large")

    assert sentinel["measured_circuits"] == 192
    assert sentinel["batch_sizes"] == [192]
    assert sentinel["quantum_seconds_estimate"] == {
        "low": 30,
        "central": 35,
        "high": 40,
    }
    assert large["measured_circuits"] == 1536
    assert large["batch_sizes"] == [300, 300, 300, 300, 300, 36]
    assert large["quantum_seconds_estimate"] == {
        "low": 240,
        "central": 280,
        "high": 320,
    }
    assert sentinel["phase_cap"] + large["phase_cap"] == 450
    assert validate.SHOTS == 128
    assert validate.BACKEND == "ibm_fez"


def test_batch_ranges_are_contiguous_and_cover_large_phase() -> None:
    rows = validate.batch_ranges(1536, [300, 300, 300, 300, 300, 36])

    assert rows[0]["start_circuit_index"] == 0
    assert rows[-1]["stop_circuit_index_exclusive"] == 1536
    assert all(
        left["stop_circuit_index_exclusive"] == right["start_circuit_index"]
        for left, right in zip(rows, rows[1:])
    )
    assert max(row["circuit_count"] for row in rows) == 300


def test_validate_runner_cannot_execute_or_retrieve() -> None:
    source = inspect.getsource(validate)
    tree = ast.parse(source)
    calls = [
        node.attr for node in ast.walk(tree) if isinstance(node, ast.Attribute)
    ]

    assert "execute" not in calls
    assert "get_result" not in calls
    assert "validate" in calls
    assert '"execution_attempted": False' in source
    assert '"quantum_seconds_used": 0' in source
    args = validate.build_parser().parse_args([])
    assert args.validate is False
    assert not hasattr(args, "confirm_submit")
