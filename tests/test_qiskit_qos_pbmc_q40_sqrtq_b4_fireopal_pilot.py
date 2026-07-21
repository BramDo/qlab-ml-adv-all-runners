from __future__ import annotations

import ast
import inspect
from pathlib import Path

import pytest

import qiskit_qos_pbmc_q40_sqrtq_b4_fireopal_pilot as pilot


def test_pinned_bundle_and_validation_are_submission_ready() -> None:
    qasms, manifest, _, bundle = pilot._load_bundle(pilot.DEFAULT_BUNDLE)
    validation = pilot._validated_report(pilot.DEFAULT_VALIDATION, bundle)

    assert len(qasms) == 192
    assert len(manifest) == 192
    assert bundle["sha256"] == pilot.PINNED_BUNDLE_SHA256
    assert bundle["aggregate_qasm_sha256"] == pilot.PINNED_AGGREGATE_QASM_SHA256
    assert [row["measurement_basis"] for row in manifest[:3]] == ["X", "Y", "Z"]
    assert manifest[0]["split"] == "train"
    assert manifest[-1]["split"] == "test"
    assert validation["provider_validation_passed"] is True
    assert validation["warning_categories"] == {
        "measurement_error_high": 192,
        "x_gate_error_high": 189,
    }


def test_plan_is_provider_free_and_predeclares_readout(tmp_path: Path) -> None:
    target = tmp_path / "plan.json"
    args = pilot.build_parser().parse_args(["plan", "--plan", str(target)])
    result = pilot.plan_pilot(args)

    assert target.is_file()
    assert result["status"] == "authorized_and_ready_for_confirmed_submission"
    assert result["submission_attempted"] is False
    assert result["provider_calls"] == []
    assert result["pilot"]["measured_circuits"] == 192
    assert result["pilot"]["shots_per_circuit"] == 128
    assert result["pilot"]["total_requested_shots"] == 24576
    assert result["predeclared_readout"]["observable_count"] == 405
    assert len(result["predeclared_readout"]["observable_panel"]) == 405


def test_submit_requires_literal_and_existing_intent_blocks_resubmission(
    tmp_path: Path,
) -> None:
    args = pilot.build_parser().parse_args(["submit"])
    with pytest.raises(pilot.PilotError, match="requires --confirm-submit"):
        pilot.submit_pilot(args)

    intent = tmp_path / "intent.json"
    intent.write_text("{}", encoding="utf-8")
    args = pilot.build_parser().parse_args(
        [
            "submit",
            "--intent",
            str(intent),
            "--receipt",
            str(tmp_path / "receipt.json"),
            "--confirm-submit",
            pilot.SUBMIT_CONFIRMATION,
        ]
    )
    with pytest.raises(pilot.PilotError, match="refusing resubmission"):
        pilot.submit_pilot(args)


def test_execution_call_is_confined_to_submit_and_retrieve_cannot_submit() -> None:
    module_source = inspect.getsource(pilot)
    tree = ast.parse(module_source)
    calls_by_function: dict[str, list[str]] = {}
    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
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
    assert '"automatic_resubmission": False' in module_source


def test_probability_results_preserve_order_and_generate_bounded_panel() -> None:
    _, manifest, _, _ = pilot._load_bundle(pilot.DEFAULT_BUNDLE)
    all_zero = {"0" * pilot.PILOT_QUBITS: 1.0}
    result = pilot.validate_hardware_result(
        {"results": [all_zero for _ in range(pilot.PILOT_CIRCUITS)]}, manifest
    )

    assert result["distribution_validation"]["passed"] is True
    assert result["distribution_validation"]["semantics"] == ["probability"]
    assert result["observable_validation"]["passed"] is True
    assert result["observable_validation"]["observable_count_per_sample"] == 405
    assert result["observable_validation"]["feature_value_count"] == 64 * 405
    assert result["observable_validation"]["minimum"] == pytest.approx(1.0)
    assert result["observable_validation"]["maximum"] == pytest.approx(1.0)
    assert len(result["hardware_feature_rows"]) == 64
    assert result["classifier_analysis_performed"] is False
