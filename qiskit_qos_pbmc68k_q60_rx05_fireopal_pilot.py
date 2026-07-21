#!/usr/bin/env python3
"""Safe PBMC68k q60 RX(0.5x) Fire Opal seed-11 hardware pilot.

``plan`` is local-only and is also the default mode. ``submit`` is the only
mode that can call ``fireopal.execute``; it requires both the explicit
subcommand and an exact confirmation literal. ``retrieve`` only uses a saved
action ID and can never submit or resubmit hardware work.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import math
import os
import sys
import time
import uuid
from collections import Counter
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from qiskit import qasm2

import qiskit_qos_pbmc68k_q40_fireopal_validate as q40_validate


SCHEMA_VERSION = "1.0"
PILOT_SEED = 11
PILOT_QUBITS = 60
PILOT_BASE_CIRCUITS = 65
PILOT_CIRCUITS = 195
PILOT_SHOTS = 128
PILOT_BACKEND = "ibm_fez"
PILOT_MEASUREMENT_BASES = ("X", "Y", "Z")
SUBMIT_CONFIRMATION = "FIREOPAL_SUBMIT_Q60_RX05_SEED11_195X128_IBM_FEZ"

PINNED_VALIDATION_SHA256 = (
    "fc9d4f37e9cbd4e5def6d370e88e658a44836a1506704ad8cdad449b5baf8f57"
)
PINNED_BUNDLE_SHA256 = (
    "6e1bc418261ce27907242d80d9bbe067126399897c1ec2edc0cae772dfdad596"
)
PINNED_AGGREGATE_QASM_SHA256 = (
    "adfcc43949c440c53ed14f350ad06fac3553dde1caf4538afc476a74e4a638b8"
)

ARTIFACT_DIR = Path("fire_opal_pbmc68k_q60_shallow")
DEFAULT_VALIDATION = (
    ARTIFACT_DIR / "pbmc68k_q60_seed11_rx05_fireopal_validate.json"
)
DEFAULT_BUNDLE = (
    ARTIFACT_DIR / "pbmc68k_q60_seed11_rx05_fireopal_qasm2.json.gz"
)
DEFAULT_PLAN = (
    ARTIFACT_DIR / "pbmc68k_q60_seed11_rx05_fireopal_pilot_plan.json"
)
DEFAULT_INTENT = (
    ARTIFACT_DIR / "pbmc68k_q60_seed11_rx05_fireopal_submission_intent.json"
)
DEFAULT_RECEIPT = (
    ARTIFACT_DIR / "pbmc68k_q60_seed11_rx05_fireopal_submission_receipt.json"
)
DEFAULT_RESULT = (
    ARTIFACT_DIR / "pbmc68k_q60_seed11_rx05_fireopal_result.json"
)


class PilotError(RuntimeError):
    """A sanitized error that is safe to show or persist."""


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _aggregate_qasm_hash(qasm_hashes: Sequence[str]) -> str:
    encoded = json.dumps(list(qasm_hashes), separators=(",", ":")).encode("utf-8")
    return _sha256_bytes(encoded)


def _atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    try:
        temporary.write_text(
            json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8"
        )
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _load_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise PilotError(f"Required artifact is missing: {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise PilotError(f"Could not read JSON artifact {path} ({type(exc).__name__})") from None
    if not isinstance(payload, Mapping):
        raise PilotError(f"Expected a JSON object in {path}")
    return dict(payload)


def _measurement_mapping(circuit: Any) -> list[dict[str, int]]:
    return [
        {
            "qubit": circuit.find_bit(item.qubits[0]).index,
            "clbit": circuit.find_bit(item.clbits[0]).index,
        }
        for item in circuit.data
        if item.operation.name == "measure"
    ]


def _load_bundle(
    path: Path,
) -> tuple[list[str], list[dict[str, Any]], dict[str, Any], dict[str, Any]]:
    """Load and independently re-parse every ordered QASM payload."""

    if not path.is_file():
        raise PilotError(f"QASM bundle is missing: {path}")
    try:
        with gzip.open(path, "rt", encoding="utf-8") as handle:
            payload = json.load(handle)
    except (OSError, json.JSONDecodeError) as exc:
        raise PilotError(f"Could not read QASM bundle ({type(exc).__name__})") from None
    if not isinstance(payload, Mapping):
        raise PilotError("QASM bundle root must be a mapping")
    circuits = payload.get("circuits")
    if not isinstance(circuits, list) or len(circuits) != PILOT_CIRCUITS:
        raise PilotError(f"Pilot bundle must contain exactly {PILOT_CIRCUITS} circuits")

    expected_mapping = [
        {"qubit": index, "clbit": index} for index in range(PILOT_QUBITS)
    ]
    qasms: list[str] = []
    manifest: list[dict[str, Any]] = []
    for expected_index, raw_row in enumerate(circuits):
        if not isinstance(raw_row, Mapping):
            raise PilotError("QASM bundle circuit rows must be mappings")
        row = dict(raw_row)
        qasm = row.pop("qasm", None)
        if not isinstance(qasm, str) or not qasm.startswith("OPENQASM 2.0;"):
            raise PilotError("Pilot circuits must be OpenQASM 2 payloads")
        if int(row.get("circuit_index", -1)) != expected_index:
            raise PilotError("Pilot circuit order is not contiguous")
        if int(row.get("seed", -1)) != PILOT_SEED:
            raise PilotError("Pilot bundle contains a non-seed-11 circuit")
        if int(row.get("base_circuit_index", -1)) != expected_index // 3:
            raise PilotError("Pilot base-circuit order changed")
        expected_basis = PILOT_MEASUREMENT_BASES[expected_index % 3]
        if str(row.get("measurement_basis")) != expected_basis:
            raise PilotError("Pilot X/Y/Z measurement order changed")
        metrics = row.get("metrics")
        if not isinstance(metrics, Mapping):
            raise PilotError("Pilot manifest lacks circuit metrics")
        if (
            int(metrics.get("num_qubits", -1)) != PILOT_QUBITS
            or int(metrics.get("num_clbits", -1)) != PILOT_QUBITS
        ):
            raise PilotError("Pilot bundle contains a circuit with wrong width")
        required_flags = (
            row.get("virtual_qubits_only") is True,
            row.get("all_parameters_numeric") is True,
            row.get("round_trip_validated") is True,
            int(row.get("quantum_register_count", -1)) == 1,
            int(row.get("classical_register_count", -1)) == 1,
        )
        if not all(required_flags):
            raise PilotError("Pilot QASM manifest lost a local validation flag")
        actual_hash = _sha256_bytes(qasm.encode("utf-8"))
        if actual_hash != str(row.get("qasm_sha256", "")):
            raise PilotError("Pilot QASM content does not match its manifest hash")
        try:
            parsed = qasm2.loads(qasm)
        except Exception as exc:
            raise PilotError(
                f"Pilot QASM {expected_index} no longer parses ({type(exc).__name__})"
            ) from None
        if parsed.num_qubits != PILOT_QUBITS or parsed.num_clbits != PILOT_QUBITS:
            raise PilotError("Parsed pilot circuit has wrong register widths")
        actual_mapping = sorted(
            _measurement_mapping(parsed), key=lambda item: item["clbit"]
        )
        if actual_mapping != expected_mapping:
            raise PilotError("Parsed pilot measurement mapping is not q[i] to c[i]")
        qasms.append(qasm)
        manifest.append(row)

    qasm_hashes = [str(row["qasm_sha256"]) for row in manifest]
    aggregate = _aggregate_qasm_hash(qasm_hashes)
    info = {
        "path": str(path.resolve()),
        "bytes": path.stat().st_size,
        "sha256": _sha256_file(path),
        "circuits": len(qasms),
        "aggregate_qasm_sha256": aggregate,
        "independent_qasm_parse_passed": True,
        "measurement_width_and_mapping_passed": True,
    }
    return qasms, manifest, dict(payload), info


def _validated_report(path: Path, bundle_info: Mapping[str, Any]) -> dict[str, Any]:
    report = _load_json(path)
    if _sha256_file(path) != PINNED_VALIDATION_SHA256:
        raise PilotError("Validation report hash differs from the pinned passing report")
    if report.get("kind") != "pbmc68k_q60_rx05_fireopal_validate_only":
        raise PilotError("Unexpected q60 RX(0.5x) validation report kind")
    if report.get("status") != "pass" or report.get("execution_attempted") is not False:
        raise PilotError("Validation report is not a passing validate-only artifact")
    config = report.get("config")
    local = report.get("local_validation")
    provider = report.get("provider_validation")
    recorded_bundle = report.get("qasm_bundle")
    if not all(isinstance(value, Mapping) for value in (config, local, provider, recorded_bundle)):
        raise PilotError("Validation report is missing required sections")
    if (
        int(config.get("seed", -1)) != PILOT_SEED
        or int(config.get("qubits", -1)) != PILOT_QUBITS
        or str(config.get("backend")) != PILOT_BACKEND
        or int(config.get("future_readout_shots", -1)) != PILOT_SHOTS
        or str(config.get("architecture")) != "rx_0p5"
    ):
        raise PilotError("Validation report configuration differs from the pilot")
    if local.get("passed") is not True:
        raise PilotError("Validation report local gates did not pass")
    if (
        int(local.get("circuit_count", -1)) != PILOT_CIRCUITS
        or str(local.get("aggregate_sha256")) != PINNED_AGGREGATE_QASM_SHA256
    ):
        raise PilotError("Validation report circuit count or aggregate hash changed")
    if (
        provider.get("requested") is not True
        or provider.get("passed") is not True
        or provider.get("execution_attempted") is not False
        or str(provider.get("backend")) != PILOT_BACKEND
        or provider.get("backend_supported") is not True
    ):
        raise PilotError("Fire Opal validate-only gate is no longer passing")
    if (
        str(recorded_bundle.get("sha256")) != bundle_info["sha256"]
        or str(recorded_bundle.get("sha256")) != PINNED_BUNDLE_SHA256
        or int(recorded_bundle.get("circuits", -1)) != PILOT_CIRCUITS
    ):
        raise PilotError("Validated bundle identity differs from the pilot bundle")
    warnings = [str(value) for value in provider.get("warnings", [])]
    warning_categories = Counter(
        "measurement_error_high"
        if "measurement error is much higher" in value
        else "x_gate_error_high"
        if "X gate error is much higher" in value
        else "other"
        for value in warnings
    )
    return {
        "path": str(path.resolve()),
        "sha256": _sha256_file(path),
        "status": str(report["status"]),
        "provider_validation_passed": True,
        "validation_execution_attempted": False,
        "validation_quantum_seconds_used": int(report.get("quantum_seconds_used", 0)),
        "warning_count": len(warnings),
        "warning_categories": dict(sorted(warning_categories.items())),
        "max_logical_depth": int(local["max_logical_depth_before_measurement"]),
        "max_payload_depth": int(local["max_payload_depth"]),
        "total_qasm_bytes": int(local["total_qasm_bytes"]),
    }


def plan_pilot(args: argparse.Namespace) -> dict[str, Any]:
    """Create a local, zero-provider-call plan pinned to validated artifacts."""

    if args.plan.exists() and not args.force:
        raise PilotError(f"Refusing to overwrite existing artifact: {args.plan}")
    started = time.perf_counter()
    _, manifest, payload, bundle = _load_bundle(args.bundle)
    if bundle["sha256"] != PINNED_BUNDLE_SHA256:
        raise PilotError("QASM bundle file hash differs from the pinned validated bundle")
    if bundle["aggregate_qasm_sha256"] != PINNED_AGGREGATE_QASM_SHA256:
        raise PilotError("Ordered QASM aggregate hash differs from the pinned batch")
    validation = _validated_report(args.validation_report, bundle)
    seeds = payload.get("seeds")
    if not isinstance(seeds, list) or len(seeds) != 1:
        raise PilotError("QASM bundle must contain exactly one seed record")
    selected = seeds[0].get("selected_observables") if isinstance(seeds[0], Mapping) else None
    if not isinstance(selected, list) or len(selected) != 24:
        raise PilotError("QASM bundle must retain the frozen 24-observable selection")

    qasm_hashes = [str(row["qasm_sha256"]) for row in manifest]
    plan = {
        "schema_version": SCHEMA_VERSION,
        "kind": "pbmc68k_q60_rx05_fireopal_seed11_hardware_pilot_plan",
        "status": "ready_for_separate_explicit_submission_authorization",
        "captured_at_utc": _utc_now(),
        "environment": q40_validate.runtime_environment(),
        "authorized_scope": "build_and_local_plan_only",
        "submission_attempted": False,
        "execution_attempted": False,
        "quantum_seconds_used": 0,
        "provider_calls": [],
        "pilot": {
            "dataset": "PBMC68k",
            "architecture": "rx_0p5",
            "seed": PILOT_SEED,
            "qubits": PILOT_QUBITS,
            "backend": PILOT_BACKEND,
            "logical_base_circuits": PILOT_BASE_CIRCUITS,
            "measurement_bases": list(PILOT_MEASUREMENT_BASES),
            "measured_circuits": PILOT_CIRCUITS,
            "shots_per_circuit": PILOT_SHOTS,
            "total_requested_shots": PILOT_CIRCUITS * PILOT_SHOTS,
            "train_samples": 32,
            "test_samples": 32,
            "selected_observables": 24,
        },
        "validation_report": validation,
        "qasm_bundle": bundle,
        "qasm_hashes": qasm_hashes,
        "ordered_qasm_hash_count": len(qasm_hashes),
        "submission_boundary": {
            "explicit_submit_subcommand_required": True,
            "exact_confirmation_required": True,
            "confirmation_literal": SUBMIT_CONFIRMATION,
            "intent_lock_written_before_execute": True,
            "automatic_resubmission": False,
            "result_wait_during_submit": False,
            "allowed_submit_api_calls": [
                "fireopal.show_supported_devices",
                "fireopal.execute",
            ],
        },
        "result_validation": {
            "expected_distribution_count": PILOT_CIRCUITS,
            "accepts_normalized_float_probabilities": True,
            "accepts_exact_integer_shot_counts": True,
            "finite_non_negative_required": True,
            "observable_expectation_bounds": [-1.0, 1.0],
            "bit_order": "Qiskit little-endian: rightmost bit is qubit 0",
        },
        "comparison_plan": {
            "matched_direct_baseline_required_for_improvement_claim": True,
            "same_qasm_backend_shots_and_calibration_window_required": True,
            "current_plan_supports_fireopal_improvement_claim": False,
        },
        "elapsed_seconds": time.perf_counter() - started,
        "claim_boundary": (
            "This is a dry hardware-pilot plan, not an execution result. The 390 "
            "provider warnings indicate unusually high measurement and X-gate errors. "
            "Without an identical time-matched direct baseline it cannot establish a "
            "Fire Opal improvement or quantum advantage."
        ),
    }
    _atomic_write_json(args.plan, plan)
    return plan


def _verify_plan_bundle(
    plan_path: Path, bundle_path: Path
) -> tuple[dict[str, Any], list[str], list[dict[str, Any]], dict[str, Any], dict[str, Any]]:
    plan = _load_json(plan_path)
    qasms, manifest, payload, bundle = _load_bundle(bundle_path)
    if plan.get("kind") != "pbmc68k_q60_rx05_fireopal_seed11_hardware_pilot_plan":
        raise PilotError("Unexpected pilot plan kind")
    if plan.get("status") != "ready_for_separate_explicit_submission_authorization":
        raise PilotError("Pilot plan is not submission-ready")
    validation = plan.get("validation_report")
    planned_bundle = plan.get("qasm_bundle")
    if not isinstance(validation, Mapping) or not isinstance(planned_bundle, Mapping):
        raise PilotError("Pilot plan lacks pinned validation metadata")
    if str(validation.get("sha256")) != PINNED_VALIDATION_SHA256:
        raise PilotError("Pilot plan validation hash is not the pinned passing hash")
    if (
        str(planned_bundle.get("sha256")) != bundle["sha256"]
        or bundle["sha256"] != PINNED_BUNDLE_SHA256
    ):
        raise PilotError("Pilot bundle file hash differs from the plan or pin")
    if (
        str(planned_bundle.get("aggregate_qasm_sha256"))
        != bundle["aggregate_qasm_sha256"]
        or bundle["aggregate_qasm_sha256"] != PINNED_AGGREGATE_QASM_SHA256
    ):
        raise PilotError("Pilot ordered QASM aggregate differs from the plan or pin")
    planned_hashes = [str(value) for value in plan.get("qasm_hashes", [])]
    actual_hashes = [str(row["qasm_sha256"]) for row in manifest]
    if planned_hashes != actual_hashes or len(planned_hashes) != PILOT_CIRCUITS:
        raise PilotError("Pilot ordered QASM manifest differs from the plan")
    return plan, qasms, manifest, payload, bundle


def _credential_source_labels(value: Mapping[str, Any]) -> dict[str, Any]:
    """Retain provenance labels but never credential values or full CRNs."""

    allowed = ("token_source", "instance_source", "account")
    return {key: value.get(key) for key in allowed}


def submit_pilot(args: argparse.Namespace) -> dict[str, Any]:
    """Submit exactly once, persist the action ID, and never wait for a result."""

    if args.confirm_submit != SUBMIT_CONFIRMATION:
        raise PilotError(f"Submission requires --confirm-submit {SUBMIT_CONFIRMATION}")
    if args.intent.exists() or args.receipt.exists():
        raise PilotError("Submission intent or receipt exists; refusing possible resubmission")
    _, qasms, _, _, bundle = _verify_plan_bundle(args.plan, args.bundle)
    intent_id = str(uuid.uuid4())
    intent = {
        "schema_version": SCHEMA_VERSION,
        "kind": "pbmc68k_q60_rx05_fireopal_seed11_submission_intent",
        "intent_id": intent_id,
        "created_at_utc": _utc_now(),
        "status": "preflight_locked",
        "plan_path": str(args.plan.resolve()),
        "plan_sha256": _sha256_file(args.plan),
        "bundle_path": str(args.bundle.resolve()),
        "bundle_sha256": bundle["sha256"],
        "aggregate_qasm_sha256": bundle["aggregate_qasm_sha256"],
        "backend": PILOT_BACKEND,
        "circuit_count": PILOT_CIRCUITS,
        "shots_per_circuit": PILOT_SHOTS,
        "total_requested_shots": PILOT_CIRCUITS * PILOT_SHOTS,
        "execution_attempted": False,
        "automatic_resubmission": False,
    }
    _atomic_write_json(args.intent, intent)

    try:
        fireopal, credentials, credential_source, qctrl_source = (
            q40_validate._fire_opal_credentials_from_source(
                args.qiskit_account, args.qctrl_notebook, args.instance
            )
        )
        devices = q40_validate._workflow_result(
            q40_validate._safe_provider_call(
                "Fire Opal supported-device discovery",
                fireopal.show_supported_devices,
                credentials=credentials,
            )
        )
        supported = [str(value) for value in devices.get("supported_devices", [])]
        if PILOT_BACKEND not in supported:
            intent.update(
                {
                    "status": "backend_not_supported",
                    "backend_supported": False,
                    "updated_at_utc": _utc_now(),
                }
            )
            _atomic_write_json(args.intent, intent)
            raise PilotError("ibm_fez is not currently supported by Fire Opal")

        intent.update(
            {
                "status": "execute_call_started",
                "backend_supported": True,
                "execution_attempted": True,
                "updated_at_utc": _utc_now(),
            }
        )
        _atomic_write_json(args.intent, intent)
        job = q40_validate._safe_provider_call(
            "Fire Opal q60 RX(0.5x) seed-11 hardware submission",
            fireopal.execute,
            circuits=list(qasms),
            shot_count=PILOT_SHOTS,
            credentials=credentials,
            backend_name=PILOT_BACKEND,
            parameters=None,
        )
        action_id = getattr(job, "action_id", None)
        if action_id is None:
            intent.update(
                {
                    "status": "submitted_but_action_id_missing",
                    "updated_at_utc": _utc_now(),
                }
            )
            _atomic_write_json(args.intent, intent)
            raise PilotError("Execute may have been accepted but no action ID returned; do not resubmit")

        receipt = {
            "schema_version": SCHEMA_VERSION,
            "kind": "pbmc68k_q60_rx05_fireopal_seed11_submission_receipt",
            "status": "submitted_not_retrieved",
            "submitted_at_utc": _utc_now(),
            "intent_id": intent_id,
            "action_id": str(action_id),
            "backend": PILOT_BACKEND,
            "circuit_count": PILOT_CIRCUITS,
            "shots_per_circuit": PILOT_SHOTS,
            "total_requested_shots": PILOT_CIRCUITS * PILOT_SHOTS,
            "bundle_path": str(args.bundle.resolve()),
            "bundle_sha256": bundle["sha256"],
            "aggregate_qasm_sha256": bundle["aggregate_qasm_sha256"],
            "plan_path": str(args.plan.resolve()),
            "plan_sha256": _sha256_file(args.plan),
            "credential_source": _credential_source_labels(credential_source),
            "qctrl_auth_source": str(qctrl_source),
            "api_calls": ["fireopal.show_supported_devices", "fireopal.execute"],
            "execution_attempted": True,
            "result_waited_during_submit": False,
            "result_retrieved": False,
            "quantum_seconds_used": None,
            "automatic_resubmission": False,
            "claim_boundary": "Submission receipt only; no result retrieved or analyzed.",
        }
        _atomic_write_json(args.receipt, receipt)
        intent.update(
            {
                "status": "receipt_persisted",
                "action_id": str(action_id),
                "receipt_path": str(args.receipt.resolve()),
                "updated_at_utc": _utc_now(),
            }
        )
        _atomic_write_json(args.intent, intent)
        return receipt
    except (q40_validate.RunnerError, PilotError) as exc:
        if args.intent.exists():
            persisted = _load_json(args.intent)
            if persisted.get("status") not in {
                "backend_not_supported",
                "submitted_but_action_id_missing",
                "receipt_persisted",
            }:
                persisted.update(
                    {
                        "status": "submit_failed_sanitized",
                        "failure_type": type(exc).__name__,
                        "updated_at_utc": _utc_now(),
                    }
                )
                _atomic_write_json(args.intent, persisted)
        raise PilotError(str(exc)) from None


_SECRET_KEY_PARTS = (
    "api_key",
    "apikey",
    "authorization",
    "credential",
    "crn",
    "instance",
    "secret",
    "token",
)


def _json_safe_redacted(value: Any, *, parent_key: str = "") -> Any:
    if any(part in parent_key.lower() for part in _SECRET_KEY_PARTS):
        return "[redacted]"
    if isinstance(value, Mapping):
        return {
            str(key): _json_safe_redacted(item, parent_key=str(key))
            for key, item in value.items()
        }
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [_json_safe_redacted(item) for item in value]
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return f"<{type(value).__name__}>"


def _authenticated_fireopal_for_retrieval(
    qctrl_notebook: Path | None,
) -> tuple[Any, str]:
    key, source = q40_validate._qctrl_api_key(qctrl_notebook)
    import fireopal

    q40_validate._safe_provider_call(
        "Q-CTRL authentication", fireopal.authenticate_qctrl_account, api_key=key
    )
    return fireopal, source


def _result_distributions(raw: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    results = raw.get("results")
    if isinstance(results, list) and all(isinstance(item, Mapping) for item in results):
        return list(results)
    execution_results = raw.get("execution_results")
    if isinstance(execution_results, list):
        extracted: list[Mapping[str, Any]] = []
        for row in execution_results:
            if not isinstance(row, Mapping) or not isinstance(row.get("meas"), Mapping):
                raise PilotError("Execution result row lacks a 'meas' distribution")
            extracted.append(row["meas"])
        return extracted
    raise PilotError("Fire Opal payload has no supported distribution list")


def _validated_distribution(
    distribution: Mapping[str, Any], *, num_qubits: int, shots: int
) -> tuple[dict[str, float], dict[str, Any]]:
    cleaned: dict[str, float] = {}
    for raw_key, raw_value in distribution.items():
        bitstring = str(raw_key).replace(" ", "")
        if not bitstring or any(bit not in "01" for bit in bitstring):
            raise PilotError("Result distribution contains a non-binary key")
        if len(bitstring) > num_qubits:
            raise PilotError("Result bitstring is wider than the circuit")
        bitstring = bitstring.zfill(num_qubits)
        try:
            value = float(raw_value)
        except (TypeError, ValueError):
            raise PilotError("Result weights must be numeric") from None
        if not math.isfinite(value) or value < 0.0:
            raise PilotError("Result weights must be finite and non-negative")
        cleaned[bitstring] = cleaned.get(bitstring, 0.0) + value
    total = float(sum(cleaned.values()))
    if total <= 0.0:
        raise PilotError("Result distribution has zero total weight")
    if abs(total - 1.0) <= 1e-4:
        semantics = "probability"
        deviation = abs(total - 1.0)
    elif abs(total - shots) <= 1e-6 and all(
        abs(value - round(value)) <= 1e-9 for value in cleaned.values()
    ):
        semantics = "integer_counts"
        deviation = abs(total - shots)
    else:
        raise PilotError("Result is neither normalized probability nor exact shot counts")
    return cleaned, {
        "semantics": semantics,
        "total_weight": total,
        "normalization_deviation": deviation,
        "outcomes": len(cleaned),
    }


def _pauli_expectation(
    distribution: Mapping[str, float], mapping: Sequence[Mapping[str, Any]]
) -> float:
    if not mapping:
        raise PilotError("Selected observable has an empty Pauli mapping")
    qubits = [int(item["qubit"]) for item in mapping]
    if len(set(qubits)) != len(qubits) or any(
        qubit < 0 or qubit >= PILOT_QUBITS for qubit in qubits
    ):
        raise PilotError("Selected observable has invalid qubit indices")
    total = float(sum(distribution.values()))
    expectation = 0.0
    for bitstring, weight in distribution.items():
        parity = sum(int(bitstring[PILOT_QUBITS - 1 - qubit]) for qubit in qubits) % 2
        expectation += (1.0 if parity == 0 else -1.0) * weight / total
    if not math.isfinite(expectation) or abs(expectation) > 1.0 + 1e-9:
        raise PilotError("Pauli expectation is outside [-1, 1]")
    return float(max(-1.0, min(1.0, expectation)))


def validate_hardware_result(
    raw: Mapping[str, Any],
    manifest: Sequence[Mapping[str, Any]],
    bundle_payload: Mapping[str, Any],
) -> dict[str, Any]:
    distributions = _result_distributions(raw)
    if len(distributions) != PILOT_CIRCUITS or len(manifest) != PILOT_CIRCUITS:
        raise PilotError("Fire Opal result and manifest must each contain 195 circuits")
    cleaned_rows: list[dict[str, float]] = []
    checks: list[dict[str, Any]] = []
    by_base_basis: dict[tuple[int, str], dict[str, float]] = {}
    for distribution, row in zip(distributions, manifest, strict=True):
        cleaned, check = _validated_distribution(
            distribution, num_qubits=PILOT_QUBITS, shots=PILOT_SHOTS
        )
        key = (int(row["base_circuit_index"]), str(row["measurement_basis"]))
        if key in by_base_basis:
            raise PilotError("Duplicate base-circuit measurement result")
        by_base_basis[key] = cleaned
        cleaned_rows.append(cleaned)
        checks.append(check)

    seeds = bundle_payload.get("seeds")
    if not isinstance(seeds, list) or len(seeds) != 1 or not isinstance(seeds[0], Mapping):
        raise PilotError("Bundle lacks the seed observable metadata")
    selected = seeds[0].get("selected_observables")
    if not isinstance(selected, list) or len(selected) != 24:
        raise PilotError("Bundle lacks the 24 selected observables")
    expectation_rows: list[dict[str, Any]] = []
    for base_index in range(PILOT_BASE_CIRCUITS):
        for observable_index, raw_observable in enumerate(selected):
            if not isinstance(raw_observable, Mapping):
                raise PilotError("Selected observable row is invalid")
            basis = str(raw_observable.get("measurement_basis"))
            mapping = raw_observable.get("pauli_mapping")
            if basis not in PILOT_MEASUREMENT_BASES or not isinstance(mapping, list):
                raise PilotError("Selected observable basis or mapping is invalid")
            distribution = by_base_basis.get((base_index, basis))
            if distribution is None:
                raise PilotError("Missing measurement basis for selected observable")
            expectation_rows.append(
                {
                    "base_circuit_index": base_index,
                    "observable_index": observable_index,
                    "candidate_index": int(raw_observable["candidate_index"]),
                    "measurement_basis": basis,
                    "expectation": _pauli_expectation(distribution, mapping),
                }
            )
    values = [float(row["expectation"]) for row in expectation_rows]
    return {
        "distribution_validation": {
            "passed": True,
            "circuit_count": len(cleaned_rows),
            "semantics": sorted({str(row["semantics"]) for row in checks}),
            "all_finite_non_negative": True,
            "all_normalized_or_exact_shot_counts": True,
            "max_normalization_deviation": float(
                max(row["normalization_deviation"] for row in checks)
            ),
            "minimum_outcomes": int(min(row["outcomes"] for row in checks)),
            "maximum_outcomes": int(max(row["outcomes"] for row in checks)),
            "ordered_against_manifest": True,
        },
        "observable_validation": {
            "passed": True,
            "expectation_count": len(expectation_rows),
            "minimum": min(values),
            "maximum": max(values),
            "all_within_minus_one_plus_one": True,
            "bit_order": "Qiskit little-endian: rightmost bit is qubit 0",
        },
        "observable_expectations": expectation_rows,
    }


def retrieve_pilot(args: argparse.Namespace) -> dict[str, Any]:
    """Retrieve an existing action ID; this path contains no execute call."""

    if args.result.exists() and not args.force:
        raise PilotError(f"Refusing to overwrite result artifact: {args.result}")
    receipt = _load_json(args.receipt)
    if receipt.get("kind") != "pbmc68k_q60_rx05_fireopal_seed11_submission_receipt":
        raise PilotError("Unexpected submission receipt kind")
    action_id = receipt.get("action_id")
    if not action_id or not str(action_id).isnumeric():
        raise PilotError("Submission receipt has no valid numeric action ID")
    if (
        int(receipt.get("circuit_count", -1)) != PILOT_CIRCUITS
        or int(receipt.get("shots_per_circuit", -1)) != PILOT_SHOTS
        or str(receipt.get("backend")) != PILOT_BACKEND
    ):
        raise PilotError("Submission receipt scope differs from this pilot")
    _, _, manifest, payload, bundle = _verify_plan_bundle(args.plan, args.bundle)
    if str(receipt.get("bundle_sha256")) != bundle["sha256"]:
        raise PilotError("Submission receipt bundle hash differs from the pinned bundle")

    fireopal, qctrl_source = _authenticated_fireopal_for_retrieval(args.qctrl_notebook)
    raw = q40_validate._safe_provider_call(
        "Fire Opal q60 RX(0.5x) result retrieval", fireopal.get_result, str(action_id)
    )
    if not isinstance(raw, Mapping):
        raise PilotError("Fire Opal result payload is not a mapping")
    validation = validate_hardware_result(raw, manifest, payload)
    artifact = {
        "schema_version": SCHEMA_VERSION,
        "kind": "pbmc68k_q60_rx05_fireopal_seed11_hardware_result",
        "status": "retrieved_and_structurally_validated_uncompared",
        "retrieved_at_utc": _utc_now(),
        "action_id": str(action_id),
        "receipt_path": str(args.receipt.resolve()),
        "receipt_sha256_before_update": _sha256_file(args.receipt),
        "bundle_sha256": bundle["sha256"],
        "qctrl_auth_source": str(qctrl_source),
        "api_calls": ["fireopal.get_result"],
        "submission_attempted_in_this_mode": False,
        "automatic_resubmission": False,
        **validation,
        "raw_result": _json_safe_redacted(raw),
        "claim_boundary": (
            "Retrieved and structurally validated provider data. No identical "
            "time-matched direct baseline has been compared, so this does not "
            "establish a Fire Opal improvement or quantum advantage."
        ),
    }
    _atomic_write_json(args.result, artifact)
    receipt.update(
        {
            "status": "result_retrieved",
            "result_retrieved": True,
            "result_path": str(args.result.resolve()),
            "result_sha256": _sha256_file(args.result),
            "retrieved_at_utc": artifact["retrieved_at_utc"],
        }
    )
    _atomic_write_json(args.receipt, receipt)
    return artifact


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    plan = subparsers.add_parser("plan", help="local-only pinned dry-run plan")
    plan.add_argument("--validation-report", type=Path, default=DEFAULT_VALIDATION)
    plan.add_argument("--bundle", type=Path, default=DEFAULT_BUNDLE)
    plan.add_argument("--plan", type=Path, default=DEFAULT_PLAN)
    plan.add_argument("--force", action="store_true")

    submit = subparsers.add_parser("submit", help="submit once without waiting")
    submit.add_argument("--bundle", type=Path, default=DEFAULT_BUNDLE)
    submit.add_argument("--plan", type=Path, default=DEFAULT_PLAN)
    submit.add_argument("--intent", type=Path, default=DEFAULT_INTENT)
    submit.add_argument("--receipt", type=Path, default=DEFAULT_RECEIPT)
    submit.add_argument("--qiskit-account", default="default-ibm-cloud")
    submit.add_argument("--qctrl-notebook", type=Path)
    submit.add_argument("--instance")
    submit.add_argument("--confirm-submit", default="")

    retrieve = subparsers.add_parser(
        "retrieve", help="retrieve by saved action ID; never submit"
    )
    retrieve.add_argument("--bundle", type=Path, default=DEFAULT_BUNDLE)
    retrieve.add_argument("--plan", type=Path, default=DEFAULT_PLAN)
    retrieve.add_argument("--receipt", type=Path, default=DEFAULT_RECEIPT)
    retrieve.add_argument("--result", type=Path, default=DEFAULT_RESULT)
    retrieve.add_argument("--qctrl-notebook", type=Path)
    retrieve.add_argument("--force", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    command_args = list(sys.argv[1:] if argv is None else argv)
    if not command_args:
        command_args = ["plan"]
    args = build_parser().parse_args(command_args)
    if args.command == "plan":
        plan = plan_pilot(args)
        print("PBMC68k q60 RX(0.5x) Fire Opal seed-11 pilot planned locally")
        print(f"- circuits: {plan['pilot']['measured_circuits']}")
        print(f"- shots if separately authorized later: {plan['pilot']['total_requested_shots']}")
        print("- provider calls: 0")
        print("- hardware submission attempted: False")
        print(f"- validation warnings retained: {plan['validation_report']['warning_count']}")
        print(f"- plan: {args.plan}")
        return 0
    if args.command == "submit":
        receipt = submit_pilot(args)
        print("Fire Opal q60 RX(0.5x) seed-11 pilot submitted; no result wait performed")
        print(f"- action ID: {receipt['action_id']}")
        print(f"- receipt: {args.receipt}")
        return 0
    if args.command == "retrieve":
        artifact = retrieve_pilot(args)
        print("Fire Opal q60 RX(0.5x) result retrieved without resubmission")
        print(f"- action ID: {artifact['action_id']}")
        print(f"- result: {args.result}")
        return 0
    raise PilotError(f"Unsupported command: {args.command}")


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except PilotError as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(2) from None
