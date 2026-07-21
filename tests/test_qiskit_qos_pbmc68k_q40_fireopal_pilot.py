from __future__ import annotations

import argparse
import ast
import gzip
import inspect
import json
from pathlib import Path
from unittest.mock import patch

import pytest

import qiskit_qos_pbmc68k_q40_fireopal_pilot as pilot


def _submit_artifacts(tmp_path: Path) -> tuple[Path, Path]:
    bundle = tmp_path / "pilot.json.gz"
    qasm = (
        "OPENQASM 2.0;\ninclude \"qelib1.inc\";\n"
        "qreg q[40];\ncreg c[40];\nmeasure q -> c;\n"
    )
    qasm_hash = pilot._sha256_bytes(qasm.encode("utf-8"))
    circuits = [
        {
            "circuit_index": index,
            "seed": pilot.PILOT_SEED,
            "base_circuit_index": index // 2,
            "feature_index": index % 2,
            "qasm_sha256": qasm_hash,
            "metrics": {"num_qubits": pilot.PILOT_QUBITS},
            "qasm": qasm,
        }
        for index in range(pilot.PILOT_CIRCUITS)
    ]
    with gzip.open(bundle, "wt", encoding="utf-8") as handle:
        json.dump({"circuits": circuits}, handle)
    _, manifest, bundle_info = pilot._load_bundle(bundle)
    plan = tmp_path / "plan.json"
    plan.write_text(
        json.dumps(
            {
                "kind": "pbmc68k_q40_fireopal_seed11_hardware_pilot_plan",
                "validated_subset_hash_match": True,
                "qasm_bundle": bundle_info,
                "qasm_hashes": [row["qasm_sha256"] for row in manifest],
            }
        ),
        encoding="utf-8",
    )
    return bundle, plan


def _submit_args(tmp_path: Path, *, confirmation: str) -> argparse.Namespace:
    bundle, plan = _submit_artifacts(tmp_path)
    return argparse.Namespace(
        bundle=bundle,
        plan=plan,
        intent=tmp_path / "intent.json",
        receipt=tmp_path / "receipt.json",
        qiskit_account="test-account",
        qctrl_notebook=None,
        instance=None,
        confirm_submit=confirmation,
    )


def test_submit_requires_exact_confirmation_before_any_side_effect(tmp_path: Path) -> None:
    args = _submit_args(tmp_path, confirmation="wrong")
    with patch.object(
        pilot.validate_runner,
        "_fire_opal_credentials_from_source",
        side_effect=AssertionError("credentials must not be resolved"),
    ):
        with pytest.raises(pilot.PilotError, match="Submission requires"):
            pilot.submit_pilot(args)
    assert not args.intent.exists()
    assert not args.receipt.exists()


class _FakeJob:
    action_id = "action-4242"

    def result(self) -> object:
        raise AssertionError("submit must never wait for a result")


class _FakeFireOpal:
    def __init__(self) -> None:
        self.calls: list[str] = []

    def show_supported_devices(self, *, credentials: object) -> dict[str, object]:
        self.calls.append("show_supported_devices")
        return {"supported_devices": [pilot.PILOT_BACKEND]}

    def execute(
        self,
        *,
        circuits: list[str],
        shot_count: int,
        credentials: object,
        backend_name: str,
    ) -> _FakeJob:
        self.calls.append("execute")
        assert len(circuits) == pilot.PILOT_CIRCUITS
        assert shot_count == pilot.PILOT_SHOTS
        assert backend_name == pilot.PILOT_BACKEND
        return _FakeJob()


def test_submit_persists_action_id_without_waiting(tmp_path: Path) -> None:
    args = _submit_args(tmp_path, confirmation=pilot.SUBMIT_CONFIRMATION)
    fake = _FakeFireOpal()
    with patch.object(
        pilot.validate_runner,
        "_fire_opal_credentials_from_source",
        return_value=(fake, object(), {"token_source": "test"}, "test"),
    ):
        receipt = pilot.submit_pilot(args)

    assert fake.calls == ["show_supported_devices", "execute"]
    assert receipt["action_id"] == "action-4242"
    assert receipt["result_waited_during_submit"] is False
    assert receipt["automatic_resubmission"] is False
    assert json.loads(args.receipt.read_text(encoding="utf-8"))["action_id"] == "action-4242"
    assert json.loads(args.intent.read_text(encoding="utf-8"))["status"] == "receipt_persisted"
    assert ".result(" not in inspect.getsource(pilot.submit_pilot)


def test_existing_intent_blocks_possible_resubmission(tmp_path: Path) -> None:
    args = _submit_args(tmp_path, confirmation=pilot.SUBMIT_CONFIRMATION)
    args.intent.write_text("{}", encoding="utf-8")
    with patch.object(
        pilot.validate_runner,
        "_fire_opal_credentials_from_source",
        side_effect=AssertionError("credentials must not be resolved"),
    ):
        with pytest.raises(pilot.PilotError, match="refusing possible resubmission"):
            pilot.submit_pilot(args)


class _FakeRetrievalFireOpal:
    def __init__(self) -> None:
        self.action_ids: list[str] = []

    def get_result(self, action_id: str) -> dict[str, object]:
        self.action_ids.append(action_id)
        return {"results": [{"0": 0.5, "1": 0.5}], "token": "must-redact"}

    def execute(self, **_: object) -> object:
        raise AssertionError("retrieve must never execute")


def test_retrieve_uses_action_id_redacts_and_never_executes(tmp_path: Path) -> None:
    receipt = tmp_path / "receipt.json"
    receipt.write_text(
        json.dumps(
            {
                "kind": "pbmc68k_q40_fireopal_seed11_submission_receipt",
                "action_id": "action-4242",
            }
        ),
        encoding="utf-8",
    )
    result = tmp_path / "result.json"
    args = argparse.Namespace(
        receipt=receipt,
        result=result,
        qctrl_notebook=None,
        force=False,
    )
    fake = _FakeRetrievalFireOpal()
    with patch.object(
        pilot,
        "_authenticated_fireopal_for_retrieval",
        return_value=(fake, "test"),
    ):
        artifact = pilot.retrieve_pilot(args)

    assert fake.action_ids == ["action-4242"]
    assert artifact["submission_attempted_in_this_mode"] is False
    assert artifact["raw_result"]["token"] == "[redacted]"
    assert ".execute(" not in inspect.getsource(pilot.retrieve_pilot)


def _distribution(expectation: float, num_qubits: int = 40) -> dict[str, float]:
    zero = "0" * num_qubits
    one_on_q0 = "0" * (num_qubits - 1) + "1"
    return {
        zero: 0.5 * (1.0 + expectation),
        one_on_q0: 0.5 * (1.0 - expectation),
    }


def _synthetic_result_and_manifest() -> tuple[list[dict[str, float]], list[dict[str, object]]]:
    distributions: list[dict[str, float]] = []
    manifest: list[dict[str, object]] = []
    for base_index in range(pilot.PILOT_BASE_CIRCUITS):
        if base_index == 0:
            role, split, label, position = "weighted_training_sketch", "train", None, None
            feature_values = (0.85, 0.35)
        else:
            role = "query"
            split = "train" if base_index <= 16 else "test"
            local_position = base_index - 1 if split == "train" else base_index - 17
            label = -1.0 if local_position < 8 else 1.0
            position = local_position
            sign = -1.0 if label < 0 else 1.0
            feature_values = (0.55 * sign, 0.25 * sign)
        for feature_index, (pauli, expectation) in enumerate(
            zip(("X", "Y"), feature_values, strict=True)
        ):
            distributions.append(_distribution(expectation))
            manifest.append(
                {
                    "base_circuit_index": base_index,
                    "feature_index": feature_index,
                    "pauli_mapping": [{"qubit": 0, "pauli": pauli}],
                    "role": role,
                    "split": split,
                    "sample_position": position,
                    "source_row_index": position,
                    "label": label,
                }
            )
    return distributions, manifest


def test_analyze_float_probabilities_preserves_order_and_bounds() -> None:
    distributions, manifest = _synthetic_result_and_manifest()
    analysis = pilot.analyze_distributions(distributions, manifest)

    assert analysis["distribution_validation"]["passed"] is True
    assert analysis["distribution_validation"]["semantics"] == ["probability"]
    assert analysis["observable_validation"]["all_within_minus_one_plus_one"] is True
    assert analysis["model"]["test_metrics"]["balanced_accuracy"] == pytest.approx(1.0)
    assert analysis["model"]["test_balanced_accuracy_uncertainty"]["replicates"] == 10_000


def test_invalid_probability_weight_is_rejected() -> None:
    with pytest.raises(pilot.PilotError, match="finite and non-negative"):
        pilot._validated_distribution(
            {"0": 1.1, "1": -0.1}, num_qubits=1, shots=128
        )


def test_prepare_cli_is_local_only_by_default() -> None:
    args = pilot.build_parser().parse_args(["prepare"])
    assert args.command == "prepare"
    assert not hasattr(args, "confirm_submit")


def test_provider_execute_and_get_result_are_confined_to_separate_modes() -> None:
    source = Path(pilot.__file__).read_text(encoding="utf-8")
    tree = ast.parse(source)
    locations: dict[str, list[str]] = {"execute": [], "get_result": []}

    class Visitor(ast.NodeVisitor):
        current_function = "<module>"

        def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
            previous = self.current_function
            self.current_function = node.name
            self.generic_visit(node)
            self.current_function = previous

        def visit_Attribute(self, node: ast.Attribute) -> None:
            if node.attr in locations:
                locations[node.attr].append(self.current_function)
            self.generic_visit(node)

    Visitor().visit(tree)
    assert locations["execute"] == ["submit_pilot"]
    assert locations["get_result"] == ["retrieve_pilot"]
