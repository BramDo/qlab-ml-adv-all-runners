from __future__ import annotations

import ast
import inspect

import pytest

import qiskit_qos_pbmc68k_q60_module_fireopal_pilot as pilot


def test_execution_is_confined_to_submit_and_retrieval_cannot_resubmit() -> None:
    source = inspect.getsource(pilot)
    tree = ast.parse(source)
    calls_by_function: dict[str, list[str]] = {}
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            calls_by_function[node.name] = [
                child.attr
                for child in ast.walk(node)
                if isinstance(child, ast.Attribute)
            ]

    assert "execute" in calls_by_function["submit_pilot"]
    assert "execute" not in calls_by_function["plan_pilot"]
    assert "execute" not in calls_by_function["retrieve_pilot"]
    assert "get_result" in calls_by_function["retrieve_pilot"]
    assert "get_result" not in calls_by_function["submit_pilot"]
    assert '"automatic_resubmission": False' in source


@pytest.mark.parametrize("phase", ["sentinel", "large"])
def test_submit_requires_exact_phase_confirmation_before_artifact_access(phase: str) -> None:
    args = pilot.build_parser().parse_args(["submit", "--phase", phase])
    with pytest.raises(pilot.PilotError, match="requires --confirm-submit"):
        pilot.submit_pilot(args)


def test_quantum_second_caps_and_confirmations_are_phase_specific() -> None:
    assert pilot.FULL_STUDY_QUANTUM_SECONDS_CAP == 450
    assert "SENTINEL_192X128" in pilot.CONFIRMATIONS["sentinel"]
    assert "MAX50QS" in pilot.CONFIRMATIONS["sentinel"]
    assert "LARGE_1536X128" in pilot.CONFIRMATIONS["large"]
    assert "MAX400QS" in pilot.CONFIRMATIONS["large"]


def test_quantum_second_extraction_ignores_unrelated_durations() -> None:
    raw = {
        "execution_metadata": {"estimated_duration": 1420.0},
        "usage": {"quantum_seconds_used": 35.0},
    }

    assert pilot._quantum_seconds(raw) == [35.0]


def test_probability_results_preserve_order_and_generate_627_features() -> None:
    scope = pilot.validate.phase_scope("sentinel")
    all_zero = {"0" * pilot.pipeline.QUBITS: 1.0}
    manifest = []
    for circuit_index in range(scope["measured_circuits"]):
        base_index = circuit_index // 3
        manifest.append(
            {
                "base_circuit_index": base_index,
                "measurement_basis": pilot.validate.MEASUREMENT_BASES[circuit_index % 3],
                "split": "train" if base_index < 32 else "test",
                "sample_position": base_index if base_index < 32 else base_index - 32,
                "source_row_index": 1000 + base_index,
                "label_for_local_matched_analysis_only": -1 if base_index % 2 else 1,
            }
        )
    result = pilot.validate_hardware_results(
        [{"results": [all_zero for _ in range(scope["measured_circuits"])]}],
        manifest,
        phase="sentinel",
    )

    assert result["distribution_validation"]["passed"] is True
    assert result["observable_validation"]["passed"] is True
    assert result["observable_validation"]["observable_count_per_sample"] == 627
    assert result["observable_validation"]["feature_value_count"] == 64 * 627
    assert result["observable_validation"]["minimum"] == pytest.approx(1.0)
    assert result["observable_validation"]["maximum"] == pytest.approx(1.0)
    assert len(result["hardware_feature_rows"]) == 64
    assert result["classifier_analysis_performed"] is False
