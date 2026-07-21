from __future__ import annotations

import argparse
import ast
import gzip
import inspect
import json
from pathlib import Path
from unittest.mock import patch

import pytest

import qiskit_qos_pbmc68k_q60_rx05_fireopal_pilot as pilot


def _fake_manifest() -> list[dict[str, object]]:
    return [
        {
            "circuit_index": index,
            "base_circuit_index": index // 3,
            "measurement_basis": pilot.PILOT_MEASUREMENT_BASES[index % 3],
        }
        for index in range(pilot.PILOT_CIRCUITS)
    ]


def _fake_payload() -> dict[str, object]:
    return {
        "seeds": [
            {
                "selected_observables": [
                    {
                        "candidate_index": index,
                        "measurement_basis": pilot.PILOT_MEASUREMENT_BASES[index % 3],
                        "pauli_mapping": [
                            {
                                "qubit": index % pilot.PILOT_QUBITS,
                                "pauli": pilot.PILOT_MEASUREMENT_BASES[index % 3],
                            }
                        ],
                    }
                    for index in range(24)
                ]
            }
        ]
    }


def _submit_artifacts(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[Path, Path]:
    bundle = tmp_path / "pilot.json.gz"
    qasm = (
        'OPENQASM 2.0;\ninclude "qelib1.inc";\n'
        "qreg q[60];\ncreg c[60];\nmeasure q -> c;\n"
    )
    qasm_hash = pilot._sha256_bytes(qasm.encode("utf-8"))
    circuits = [
        {
            "circuit_index": index,
            "seed": pilot.PILOT_SEED,
            "base_circuit_index": index // 3,
            "measurement_basis": pilot.PILOT_MEASUREMENT_BASES[index % 3],
            "qasm_sha256": qasm_hash,
            "metrics": {
                "num_qubits": pilot.PILOT_QUBITS,
                "num_clbits": pilot.PILOT_QUBITS,
            },
            "quantum_register_count": 1,
            "classical_register_count": 1,
            "virtual_qubits_only": True,
            "all_parameters_numeric": True,
            "round_trip_validated": True,
            "qasm": qasm,
        }
        for index in range(pilot.PILOT_CIRCUITS)
    ]
    payload = {**_fake_payload(), "circuits": circuits}
    with gzip.open(bundle, "wt", encoding="utf-8") as handle:
        json.dump(payload, handle)
    _, manifest, _, bundle_info = pilot._load_bundle(bundle)
    validation_hash = "a" * 64
    monkeypatch.setattr(pilot, "PINNED_VALIDATION_SHA256", validation_hash)
    monkeypatch.setattr(pilot, "PINNED_BUNDLE_SHA256", bundle_info["sha256"])
    monkeypatch.setattr(
        pilot, "PINNED_AGGREGATE_QASM_SHA256", bundle_info["aggregate_qasm_sha256"]
    )
    plan = tmp_path / "plan.json"
    plan.write_text(
        json.dumps(
            {
                "kind": "pbmc68k_q60_rx05_fireopal_seed11_hardware_pilot_plan",
                "status": "ready_for_separate_explicit_submission_authorization",
                "validation_report": {"sha256": validation_hash},
                "qasm_bundle": bundle_info,
                "qasm_hashes": [row["qasm_sha256"] for row in manifest],
            }
        ),
        encoding="utf-8",
    )
    return bundle, plan


def _submit_args(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, *, confirmation: str
) -> argparse.Namespace:
    bundle, plan = _submit_artifacts(tmp_path, monkeypatch)
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


def test_submit_requires_exact_confirmation_before_any_side_effect(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    args = _submit_args(tmp_path, monkeypatch, confirmation="wrong")
    with patch.object(
        pilot.q40_validate,
        "_fire_opal_credentials_from_source",
        side_effect=AssertionError("credentials must not be resolved"),
    ):
        with pytest.raises(pilot.PilotError, match="Submission requires"):
            pilot.submit_pilot(args)
    assert not args.intent.exists()
    assert not args.receipt.exists()


class _FakeJob:
    action_id = "4242"

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
        parameters: object,
    ) -> _FakeJob:
        self.calls.append("execute")
        assert len(circuits) == pilot.PILOT_CIRCUITS
        assert shot_count == pilot.PILOT_SHOTS
        assert backend_name == pilot.PILOT_BACKEND
        assert parameters is None
        return _FakeJob()


def test_submit_persists_action_id_without_waiting_or_secrets(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    args = _submit_args(
        tmp_path, monkeypatch, confirmation=pilot.SUBMIT_CONFIRMATION
    )
    fake = _FakeFireOpal()
    with patch.object(
        pilot.q40_validate,
        "_fire_opal_credentials_from_source",
        return_value=(
            fake,
            object(),
            {
                "token_source": "saved_qiskit_account",
                "instance_source": "saved_qiskit_account",
                "account": "test-account",
                "instance_sha256": "do-not-copy",
                "token": "secret-token",
            },
            "notebook_in_memory",
        ),
    ):
        receipt = pilot.submit_pilot(args)

    assert fake.calls == ["show_supported_devices", "execute"]
    assert receipt["action_id"] == "4242"
    assert receipt["result_waited_during_submit"] is False
    assert receipt["automatic_resubmission"] is False
    saved = args.receipt.read_text(encoding="utf-8")
    assert "secret-token" not in saved
    assert "do-not-copy" not in saved
    assert json.loads(saved)["action_id"] == "4242"
    assert json.loads(args.intent.read_text(encoding="utf-8"))["status"] == "receipt_persisted"
    assert ".result(" not in inspect.getsource(pilot.submit_pilot)


def test_existing_receipt_blocks_possible_resubmission(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    args = _submit_args(
        tmp_path, monkeypatch, confirmation=pilot.SUBMIT_CONFIRMATION
    )
    args.receipt.write_text("{}", encoding="utf-8")
    with patch.object(
        pilot.q40_validate,
        "_fire_opal_credentials_from_source",
        side_effect=AssertionError("credentials must not be resolved"),
    ):
        with pytest.raises(pilot.PilotError, match="refusing possible resubmission"):
            pilot.submit_pilot(args)


def test_missing_action_id_sets_recovery_lock_and_never_resubmits(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    args = _submit_args(
        tmp_path, monkeypatch, confirmation=pilot.SUBMIT_CONFIRMATION
    )
    fake = _FakeFireOpal()
    fake.execute = lambda **_: type("JobWithoutActionId", (), {})()  # type: ignore[method-assign]
    with patch.object(
        pilot.q40_validate,
        "_fire_opal_credentials_from_source",
        return_value=(fake, object(), {}, "test"),
    ):
        with pytest.raises(pilot.PilotError, match="do not resubmit"):
            pilot.submit_pilot(args)

    intent = json.loads(args.intent.read_text(encoding="utf-8"))
    assert intent["status"] == "submitted_but_action_id_missing"
    assert intent["execution_attempted"] is True
    assert not args.receipt.exists()


def test_bundle_hash_mismatch_blocks_before_credentials(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    args = _submit_args(
        tmp_path, monkeypatch, confirmation=pilot.SUBMIT_CONFIRMATION
    )
    plan = json.loads(args.plan.read_text(encoding="utf-8"))
    plan["qasm_bundle"]["sha256"] = "0" * 64
    args.plan.write_text(json.dumps(plan), encoding="utf-8")
    with patch.object(
        pilot.q40_validate,
        "_fire_opal_credentials_from_source",
        side_effect=AssertionError("credentials must not be resolved"),
    ):
        with pytest.raises(pilot.PilotError, match="hash differs"):
            pilot.submit_pilot(args)
    assert not args.intent.exists()


def _probability_distribution(expectation: float = 0.0) -> dict[str, float]:
    zero = "0" * pilot.PILOT_QUBITS
    one_on_q0 = "0" * (pilot.PILOT_QUBITS - 1) + "1"
    return {
        zero: 0.5 * (1.0 + expectation),
        one_on_q0: 0.5 * (1.0 - expectation),
    }


def test_float_probability_result_preserves_order_bit_order_and_bounds() -> None:
    raw = {
        "results": [
            _probability_distribution(0.25)
            for _ in range(pilot.PILOT_CIRCUITS)
        ]
    }
    validated = pilot.validate_hardware_result(raw, _fake_manifest(), _fake_payload())
    assert validated["distribution_validation"]["semantics"] == ["probability"]
    assert validated["distribution_validation"]["ordered_against_manifest"] is True
    assert validated["observable_validation"]["expectation_count"] == 65 * 24
    assert validated["observable_validation"]["all_within_minus_one_plus_one"] is True
    first = validated["observable_expectations"][0]
    assert first["base_circuit_index"] == 0
    assert first["observable_index"] == 0
    assert first["expectation"] == pytest.approx(0.25)


@pytest.mark.parametrize(
    ("distribution", "message"),
    [
        ({"0": 1.1, "1": -0.1}, "finite and non-negative"),
        ({"0": float("nan"), "1": 1.0}, "finite and non-negative"),
        ({"0": 0.0, "1": 0.0}, "zero total weight"),
        ({"0": 0.7, "1": 0.7}, "neither normalized probability"),
    ],
)
def test_invalid_probability_weights_are_rejected(
    distribution: dict[str, float], message: str
) -> None:
    with pytest.raises(pilot.PilotError, match=message):
        pilot._validated_distribution(
            distribution, num_qubits=1, shots=pilot.PILOT_SHOTS
        )


class _FakeRetrievalFireOpal:
    def __init__(self) -> None:
        self.action_ids: list[str] = []

    def get_result(self, action_id: str) -> dict[str, object]:
        self.action_ids.append(action_id)
        return {
            "results": [
                _probability_distribution() for _ in range(pilot.PILOT_CIRCUITS)
            ],
            "token": "must-redact",
        }

    def execute(self, **_: object) -> object:
        raise AssertionError("retrieve must never execute")


def test_retrieve_uses_action_id_redacts_and_never_executes(
    tmp_path: Path,
) -> None:
    receipt = tmp_path / "receipt.json"
    receipt.write_text(
        json.dumps(
            {
                "kind": "pbmc68k_q60_rx05_fireopal_seed11_submission_receipt",
                "action_id": "4242",
                "backend": pilot.PILOT_BACKEND,
                "circuit_count": pilot.PILOT_CIRCUITS,
                "shots_per_circuit": pilot.PILOT_SHOTS,
                "bundle_sha256": "bundle-hash",
            }
        ),
        encoding="utf-8",
    )
    result = tmp_path / "result.json"
    args = argparse.Namespace(
        receipt=receipt,
        result=result,
        plan=tmp_path / "plan.json",
        bundle=tmp_path / "bundle.json.gz",
        qctrl_notebook=None,
        force=False,
    )
    fake = _FakeRetrievalFireOpal()
    bundle_info = {"sha256": "bundle-hash"}
    with (
        patch.object(
            pilot,
            "_verify_plan_bundle",
            return_value=(
                {},
                ["qasm"] * pilot.PILOT_CIRCUITS,
                _fake_manifest(),
                _fake_payload(),
                bundle_info,
            ),
        ),
        patch.object(
            pilot,
            "_authenticated_fireopal_for_retrieval",
            return_value=(fake, "test"),
        ),
    ):
        artifact = pilot.retrieve_pilot(args)

    assert fake.action_ids == ["4242"]
    assert artifact["submission_attempted_in_this_mode"] is False
    assert artifact["raw_result"]["token"] == "[redacted]"
    assert ".execute(" not in inspect.getsource(pilot.retrieve_pilot)


def test_default_cli_mode_is_local_plan(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_plan = {
        "pilot": {
            "measured_circuits": pilot.PILOT_CIRCUITS,
            "total_requested_shots": pilot.PILOT_CIRCUITS * pilot.PILOT_SHOTS,
        },
        "validation_report": {"warning_count": 390},
    }
    with patch.object(pilot, "plan_pilot", return_value=fake_plan) as planned:
        assert pilot.main([]) == 0
    planned.assert_called_once()


def test_provider_calls_are_confined_to_separate_modes() -> None:
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
