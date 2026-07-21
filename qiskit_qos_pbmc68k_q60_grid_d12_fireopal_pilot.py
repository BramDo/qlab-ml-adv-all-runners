#!/usr/bin/env python3
"""Safe Fire Opal hardware pilot for the frozen q60 grid-d12 batch.

``plan`` is local-only. ``submit`` is the only mode that can call
``fireopal.execute`` and requires an exact confirmation literal. ``retrieve``
uses only the persisted action ID and can never submit or resubmit work.
"""

from __future__ import annotations

import argparse
import gzip
import json
import math
import sys
import time
import uuid
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

import qiskit_qos_pbmc68k_q40_fireopal_validate as q40_validate
import qiskit_qos_pbmc68k_q60_rx05_fireopal_pilot as safe_base
import qiskit_qos_pbmc68k_q60_scrambled_mixer_screen as architecture
import qiskit_qos_pbmc68k_q60_shallow_fireopal_validate as q60


SCHEMA_VERSION = "1.0"
PILOT_SEED = 11
PILOT_QUBITS = 60
PILOT_BASE_CIRCUITS = 65
PILOT_CIRCUITS = 195
PILOT_SHOTS = 128
PILOT_BACKEND = "ibm_fez"
PILOT_ARCHITECTURE = "grid_mixer_d12"
PILOT_PANEL = "multiscale_pairs"
PILOT_MEASUREMENT_BASES = ("X", "Y", "Z")
SUBMIT_CONFIRMATION = "FIREOPAL_SUBMIT_Q60_GRID_D12_SEED11_195X128_IBM_FEZ"

PINNED_VALIDATION_SHA256 = (
    "60a4db24f5fbec0c26ca375f672014afb7038fb4a79376750578e1ee750b519a"
)
PINNED_BUNDLE_SHA256 = (
    "908affac079a1e83dba64ba8171163a200aa46906185c77299dc32c45ff24aae"
)
PINNED_AGGREGATE_QASM_SHA256 = (
    "0922c860b665ff7795e33fe04cb9665e8a362d70d8f5320f81c6f36b62fd0d51"
)

ARTIFACT_DIR = Path("fire_opal_pbmc68k_q60_shallow")
DEFAULT_VALIDATION = ARTIFACT_DIR / (
    "pbmc68k_q60_seed11_grid_d12_multiscale_pairs_fireopal_validate.json"
)
DEFAULT_BUNDLE = ARTIFACT_DIR / (
    "pbmc68k_q60_seed11_grid_d12_multiscale_pairs_fireopal_qasm2.json.gz"
)
DEFAULT_PLAN = ARTIFACT_DIR / (
    "pbmc68k_q60_seed11_grid_d12_multiscale_pairs_fireopal_pilot_plan.json"
)
DEFAULT_INTENT = ARTIFACT_DIR / (
    "pbmc68k_q60_seed11_grid_d12_multiscale_pairs_submission_intent.json"
)
DEFAULT_RECEIPT = ARTIFACT_DIR / (
    "pbmc68k_q60_seed11_grid_d12_multiscale_pairs_submission_receipt.json"
)
DEFAULT_RESULT = ARTIFACT_DIR / (
    "pbmc68k_q60_seed11_grid_d12_multiscale_pairs_hardware_result.json"
)

PilotError = safe_base.PilotError


def _load_bundle(
    path: Path,
) -> tuple[list[str], list[dict[str, Any]], dict[str, Any], dict[str, Any]]:
    if not path.is_file():
        raise PilotError(f"QASM bundle is missing: {path}")
    try:
        with gzip.open(path, "rt", encoding="utf-8") as handle:
            payload = json.load(handle)
    except (OSError, json.JSONDecodeError) as exc:
        raise PilotError(f"Could not read QASM bundle ({type(exc).__name__})") from None
    if not isinstance(payload, Mapping):
        raise PilotError("QASM bundle root must be a mapping")
    if payload.get("kind") != (
        "pbmc68k_q60_grid_d12_multiscale_pairs_numeric_qasm2_batch"
    ):
        raise PilotError("Unexpected QASM bundle kind")
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
        expected_basis = PILOT_MEASUREMENT_BASES[expected_index % 3]
        if (
            int(row.get("circuit_index", -1)) != expected_index
            or int(row.get("seed", -1)) != PILOT_SEED
            or int(row.get("base_circuit_index", -1)) != expected_index // 3
            or str(row.get("measurement_basis")) != expected_basis
        ):
            raise PilotError("Pilot circuit or X/Y/Z ordering changed")
        metrics = row.get("metrics")
        if not isinstance(metrics, Mapping) or (
            int(metrics.get("num_qubits", -1)) != PILOT_QUBITS
            or int(metrics.get("num_clbits", -1)) != PILOT_QUBITS
        ):
            raise PilotError("Pilot bundle contains a circuit with wrong width")
        if not all(
            (
                row.get("virtual_qubits_only") is True,
                row.get("all_parameters_numeric") is True,
                row.get("round_trip_validated") is True,
                int(row.get("quantum_register_count", -1)) == 1,
                int(row.get("classical_register_count", -1)) == 1,
            )
        ):
            raise PilotError("Pilot QASM manifest lost a validation flag")
        actual_hash = safe_base._sha256_bytes(qasm.encode("utf-8"))
        if actual_hash != str(row.get("qasm_sha256", "")):
            raise PilotError("Pilot QASM content differs from its manifest hash")
        try:
            parsed = safe_base.qasm2.loads(qasm)
        except Exception as exc:
            raise PilotError(
                f"Pilot QASM {expected_index} no longer parses ({type(exc).__name__})"
            ) from None
        actual_mapping = sorted(
            safe_base._measurement_mapping(parsed), key=lambda item: item["clbit"]
        )
        if (
            parsed.num_qubits != PILOT_QUBITS
            or parsed.num_clbits != PILOT_QUBITS
            or actual_mapping != expected_mapping
        ):
            raise PilotError("Parsed pilot register or measurement mapping changed")
        qasms.append(qasm)
        manifest.append(row)
    aggregate = safe_base._aggregate_qasm_hash(
        [str(row["qasm_sha256"]) for row in manifest]
    )
    info = {
        "path": str(path.resolve()),
        "bytes": path.stat().st_size,
        "sha256": safe_base._sha256_file(path),
        "circuits": len(qasms),
        "aggregate_qasm_sha256": aggregate,
        "independent_qasm_parse_passed": True,
        "measurement_width_and_mapping_passed": True,
    }
    return qasms, manifest, dict(payload), info


def _validated_report(path: Path, bundle: Mapping[str, Any]) -> dict[str, Any]:
    report = safe_base._load_json(path)
    if safe_base._sha256_file(path) != PINNED_VALIDATION_SHA256:
        raise PilotError("Validation report hash differs from the passing pin")
    if report.get("kind") != (
        "pbmc68k_q60_grid_d12_multiscale_pairs_fireopal_validate_only"
    ):
        raise PilotError("Unexpected grid-d12 validation report kind")
    config = report.get("config")
    local = report.get("local_validation")
    provider = report.get("provider_validation")
    recorded_bundle = report.get("qasm_bundle")
    if not all(
        isinstance(value, Mapping)
        for value in (config, local, provider, recorded_bundle)
    ):
        raise PilotError("Validation report is missing required sections")
    if (
        report.get("status") != "pass"
        or report.get("execution_attempted") is not False
        or int(config.get("seed", -1)) != PILOT_SEED
        or int(config.get("qubits", -1)) != PILOT_QUBITS
        or str(config.get("backend")) != PILOT_BACKEND
        or int(config.get("shots_per_circuit_for_future_hardware", -1))
        != PILOT_SHOTS
        or str(config.get("architecture")) != PILOT_ARCHITECTURE
        or str(config.get("panel")) != PILOT_PANEL
    ):
        raise PilotError("Validation configuration differs from the pilot")
    if (
        local.get("passed") is not True
        or int(local.get("circuit_count", -1)) != PILOT_CIRCUITS
        or str(local.get("aggregate_sha256"))
        != PINNED_AGGREGATE_QASM_SHA256
    ):
        raise PilotError("Local validation or aggregate QASM pin changed")
    if (
        provider.get("requested") is not True
        or provider.get("passed") is not True
        or provider.get("execution_attempted") is not False
        or provider.get("backend_supported") is not True
        or str(provider.get("backend")) != PILOT_BACKEND
        or int(provider.get("circuits_validated", -1)) != PILOT_CIRCUITS
    ):
        raise PilotError("Fire Opal validate-only gate is not passing")
    if (
        str(recorded_bundle.get("sha256")) != bundle["sha256"]
        or bundle["sha256"] != PINNED_BUNDLE_SHA256
    ):
        raise PilotError("Validated bundle identity differs from the pin")
    warnings = [str(value) for value in provider.get("warnings", [])]
    categories = Counter(
        "measurement_error_high"
        if "measurement error is much higher" in value
        else "x_gate_error_high"
        if "X gate error is much higher" in value
        else "other"
        for value in warnings
    )
    return {
        "path": str(path.resolve()),
        "sha256": safe_base._sha256_file(path),
        "provider_validation_passed": True,
        "warning_count": len(warnings),
        "warning_categories": dict(sorted(categories.items())),
        "max_logical_depth": int(local["max_logical_depth_before_measurement"]),
        "max_payload_depth": int(local["max_payload_depth"]),
        "total_qasm_bytes": int(local["total_qasm_bytes"]),
        "advantage_signal": report["advantage_signal"],
    }


def plan_pilot(args: argparse.Namespace) -> dict[str, Any]:
    if args.plan.exists() and not args.force:
        raise PilotError(f"Refusing to overwrite existing artifact: {args.plan}")
    started = time.perf_counter()
    _, manifest, payload, bundle = _load_bundle(args.bundle)
    if (
        bundle["sha256"] != PINNED_BUNDLE_SHA256
        or bundle["aggregate_qasm_sha256"] != PINNED_AGGREGATE_QASM_SHA256
    ):
        raise PilotError("QASM bundle differs from the validated pins")
    validation = _validated_report(args.validation_report, bundle)
    seed_batch = payload.get("seed_batch")
    selected = (
        seed_batch.get("selected_observables")
        if isinstance(seed_batch, Mapping)
        else None
    )
    if not isinstance(selected, list) or len(selected) != 24:
        raise PilotError("QASM bundle must retain 24 frozen observables")
    qasm_hashes = [str(row["qasm_sha256"]) for row in manifest]
    plan = {
        "schema_version": SCHEMA_VERSION,
        "kind": "pbmc68k_q60_grid_d12_fireopal_seed11_hardware_pilot_plan",
        "status": "authorized_and_ready_for_confirmed_submission",
        "captured_at_utc": safe_base._utc_now(),
        "environment": q40_validate.runtime_environment(),
        "user_authorization": "explicit_submit_authorization_received",
        "submission_attempted": False,
        "provider_calls": [],
        "pilot": {
            "dataset": "PBMC68k",
            "architecture": PILOT_ARCHITECTURE,
            "panel": PILOT_PANEL,
            "seed": PILOT_SEED,
            "qubits": PILOT_QUBITS,
            "backend": PILOT_BACKEND,
            "logical_base_circuits": PILOT_BASE_CIRCUITS,
            "measurement_bases": list(PILOT_MEASUREMENT_BASES),
            "measured_circuits": PILOT_CIRCUITS,
            "shots_per_circuit": PILOT_SHOTS,
            "total_requested_shots": PILOT_CIRCUITS * PILOT_SHOTS,
            "selected_observables": len(selected),
        },
        "validation_report": validation,
        "qasm_bundle": bundle,
        "qasm_hashes": qasm_hashes,
        "submission_boundary": {
            "explicit_submit_subcommand_required": True,
            "exact_confirmation_required": True,
            "confirmation_literal": SUBMIT_CONFIRMATION,
            "intent_lock_written_before_execute": True,
            "automatic_resubmission": False,
            "result_wait_during_submit": False,
        },
        "comparison_plan": {
            "classical_fixed_test_balanced_accuracy": 0.59375,
            "ideal_quantum_fixed_test_balanced_accuracy": 0.53125,
            "positive_training_cv_delta": 0.03125,
            "paper_goal": "hardware_feasibility_with_exploratory_advantage_signal",
        },
        "elapsed_seconds": time.perf_counter() - started,
        "claim_boundary": (
            "This authorized submission plan records an advantage-motivated hardware "
            "experiment. It is not yet a hardware result or proof of quantum advantage."
        ),
    }
    safe_base._atomic_write_json(args.plan, plan)
    return plan


def _verify_plan_bundle(
    plan_path: Path, bundle_path: Path
) -> tuple[dict[str, Any], list[str], list[dict[str, Any]], dict[str, Any], dict[str, Any]]:
    plan = safe_base._load_json(plan_path)
    qasms, manifest, payload, bundle = _load_bundle(bundle_path)
    if (
        plan.get("kind")
        != "pbmc68k_q60_grid_d12_fireopal_seed11_hardware_pilot_plan"
        or plan.get("status") != "authorized_and_ready_for_confirmed_submission"
        or plan.get("user_authorization")
        != "explicit_submit_authorization_received"
    ):
        raise PilotError("Pilot plan is not explicitly submission-ready")
    validation = plan.get("validation_report")
    planned_bundle = plan.get("qasm_bundle")
    if not isinstance(validation, Mapping) or not isinstance(planned_bundle, Mapping):
        raise PilotError("Pilot plan lacks pinned validation metadata")
    if str(validation.get("sha256")) != PINNED_VALIDATION_SHA256:
        raise PilotError("Pilot plan validation hash differs from the pin")
    if (
        str(planned_bundle.get("sha256")) != bundle["sha256"]
        or bundle["sha256"] != PINNED_BUNDLE_SHA256
        or str(planned_bundle.get("aggregate_qasm_sha256"))
        != bundle["aggregate_qasm_sha256"]
        or bundle["aggregate_qasm_sha256"] != PINNED_AGGREGATE_QASM_SHA256
    ):
        raise PilotError("Pilot bundle differs from the plan or pin")
    planned_hashes = [str(value) for value in plan.get("qasm_hashes", [])]
    actual_hashes = [str(row["qasm_sha256"]) for row in manifest]
    if planned_hashes != actual_hashes or len(planned_hashes) != PILOT_CIRCUITS:
        raise PilotError("Ordered QASM manifest differs from the plan")
    return plan, qasms, manifest, payload, bundle


def submit_pilot(args: argparse.Namespace) -> dict[str, Any]:
    if args.confirm_submit != SUBMIT_CONFIRMATION:
        raise PilotError(f"Submission requires --confirm-submit {SUBMIT_CONFIRMATION}")
    if args.intent.exists() or args.receipt.exists():
        raise PilotError("Submission intent or receipt exists; refusing resubmission")
    _, qasms, _, _, bundle = _verify_plan_bundle(args.plan, args.bundle)
    intent_id = str(uuid.uuid4())
    intent = {
        "schema_version": SCHEMA_VERSION,
        "kind": "pbmc68k_q60_grid_d12_fireopal_submission_intent",
        "intent_id": intent_id,
        "created_at_utc": safe_base._utc_now(),
        "status": "preflight_locked",
        "plan_path": str(args.plan.resolve()),
        "plan_sha256": safe_base._sha256_file(args.plan),
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
    safe_base._atomic_write_json(args.intent, intent)
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
                    "updated_at_utc": safe_base._utc_now(),
                }
            )
            safe_base._atomic_write_json(args.intent, intent)
            raise PilotError("ibm_fez is not currently supported by Fire Opal")
        intent.update(
            {
                "status": "execute_call_started",
                "backend_supported": True,
                "execution_attempted": True,
                "updated_at_utc": safe_base._utc_now(),
            }
        )
        safe_base._atomic_write_json(args.intent, intent)
        job = q40_validate._safe_provider_call(
            "Fire Opal q60 grid-d12 seed-11 hardware submission",
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
                    "updated_at_utc": safe_base._utc_now(),
                }
            )
            safe_base._atomic_write_json(args.intent, intent)
            raise PilotError(
                "Execute may have been accepted but no action ID returned; do not resubmit"
            )
        receipt = {
            "schema_version": SCHEMA_VERSION,
            "kind": "pbmc68k_q60_grid_d12_fireopal_submission_receipt",
            "status": "submitted_not_retrieved",
            "submitted_at_utc": safe_base._utc_now(),
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
            "plan_sha256": safe_base._sha256_file(args.plan),
            "credential_source": safe_base._credential_source_labels(
                credential_source
            ),
            "qctrl_auth_source": str(qctrl_source),
            "api_calls": ["fireopal.show_supported_devices", "fireopal.execute"],
            "execution_attempted": True,
            "result_waited_during_submit": False,
            "result_retrieved": False,
            "quantum_seconds_used": None,
            "automatic_resubmission": False,
            "claim_boundary": "Submission receipt only; no result retrieved yet.",
        }
        safe_base._atomic_write_json(args.receipt, receipt)
        intent.update(
            {
                "status": "receipt_persisted",
                "action_id": str(action_id),
                "receipt_path": str(args.receipt.resolve()),
                "updated_at_utc": safe_base._utc_now(),
            }
        )
        safe_base._atomic_write_json(args.intent, intent)
        return receipt
    except (q40_validate.RunnerError, PilotError) as exc:
        if args.intent.exists():
            persisted = safe_base._load_json(args.intent)
            if persisted.get("status") not in {
                "backend_not_supported",
                "submitted_but_action_id_missing",
                "receipt_persisted",
            }:
                persisted.update(
                    {
                        "status": "submit_failed_sanitized",
                        "failure_type": type(exc).__name__,
                        "updated_at_utc": safe_base._utc_now(),
                    }
                )
                safe_base._atomic_write_json(args.intent, persisted)
        raise PilotError(str(exc)) from None


def _pauli_expectation(
    distribution: Mapping[str, float], mapping: Sequence[Mapping[str, Any]]
) -> float:
    qubits = [int(item["qubit"]) for item in mapping]
    if not qubits or len(qubits) != len(set(qubits)):
        raise PilotError("Selected observable has invalid support")
    total = float(sum(distribution.values()))
    expectation = 0.0
    for bitstring, weight in distribution.items():
        parity = sum(
            int(bitstring[PILOT_QUBITS - 1 - qubit]) for qubit in qubits
        ) % 2
        expectation += (1.0 if parity == 0 else -1.0) * weight / total
    if not math.isfinite(expectation) or abs(expectation) > 1.0 + 1e-9:
        raise PilotError("Pauli expectation is outside [-1, 1]")
    return float(max(-1.0, min(1.0, expectation)))


def validate_hardware_result(
    raw: Mapping[str, Any],
    manifest: Sequence[Mapping[str, Any]],
    bundle_payload: Mapping[str, Any],
) -> dict[str, Any]:
    distributions = safe_base._result_distributions(raw)
    if len(distributions) != PILOT_CIRCUITS or len(manifest) != PILOT_CIRCUITS:
        raise PilotError("Fire Opal result and manifest must each contain 195 circuits")
    checks: list[dict[str, Any]] = []
    by_base_basis: dict[tuple[int, str], dict[str, float]] = {}
    for distribution, row in zip(distributions, manifest, strict=True):
        cleaned, check = safe_base._validated_distribution(
            distribution, num_qubits=PILOT_QUBITS, shots=PILOT_SHOTS
        )
        key = (int(row["base_circuit_index"]), str(row["measurement_basis"]))
        if key in by_base_basis:
            raise PilotError("Duplicate base-circuit measurement result")
        by_base_basis[key] = cleaned
        checks.append(check)
    seed_batch = bundle_payload.get("seed_batch")
    selected = (
        seed_batch.get("selected_observables")
        if isinstance(seed_batch, Mapping)
        else None
    )
    if not isinstance(selected, list) or len(selected) != 24:
        raise PilotError("Bundle lacks the 24 selected observables")
    feature_matrix = np.empty((PILOT_BASE_CIRCUITS, len(selected)), dtype=np.float64)
    expectation_rows: list[dict[str, Any]] = []
    for base_index in range(PILOT_BASE_CIRCUITS):
        for observable_index, observable in enumerate(selected):
            if not isinstance(observable, Mapping):
                raise PilotError("Selected observable row is invalid")
            basis = str(observable.get("measurement_basis"))
            mapping = observable.get("pauli_mapping")
            if basis not in PILOT_MEASUREMENT_BASES or not isinstance(mapping, list):
                raise PilotError("Selected observable basis or mapping is invalid")
            distribution = by_base_basis.get((base_index, basis))
            if distribution is None:
                raise PilotError("Missing measurement basis result")
            value = _pauli_expectation(distribution, mapping)
            feature_matrix[base_index, observable_index] = value
            expectation_rows.append(
                {
                    "base_circuit_index": base_index,
                    "observable_index": observable_index,
                    "master_index": int(observable["master_index"]),
                    "measurement_basis": basis,
                    "expectation": value,
                }
            )
    base_metadata: dict[int, Mapping[str, Any]] = {}
    for row in manifest:
        base_metadata.setdefault(int(row["base_circuit_index"]), row)
    train_labels = np.asarray(
        [float(base_metadata[index]["label"]) for index in range(1, 33)]
    )
    test_labels = np.asarray(
        [float(base_metadata[index]["label"]) for index in range(33, 65)]
    )
    head_train = q60._head_features(feature_matrix[0], feature_matrix[1:33])
    head_test = q60._head_features(feature_matrix[0], feature_matrix[33:65])
    train_scores, test_scores = q60._ridge_scores(
        head_train, head_test, train_labels
    )
    values = feature_matrix.ravel()
    return {
        "distribution_validation": {
            "passed": True,
            "circuit_count": len(distributions),
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
            "minimum": float(np.min(values)),
            "maximum": float(np.max(values)),
            "all_within_minus_one_plus_one": True,
            "bit_order": "Qiskit little-endian: rightmost bit is qubit 0",
        },
        "hardware_classifier": {
            "train": architecture._balanced_metrics(train_labels, train_scores),
            "fixed_test": architecture._balanced_metrics(test_labels, test_scores),
            "ideal_quantum_fixed_test_balanced_accuracy": 0.53125,
            "classical_fixed_test_balanced_accuracy": 0.59375,
        },
        "observable_expectations": expectation_rows,
    }


def retrieve_pilot(args: argparse.Namespace) -> dict[str, Any]:
    if args.result.exists() and not args.force:
        raise PilotError(f"Refusing to overwrite result artifact: {args.result}")
    receipt = safe_base._load_json(args.receipt)
    if receipt.get("kind") != "pbmc68k_q60_grid_d12_fireopal_submission_receipt":
        raise PilotError("Unexpected submission receipt kind")
    action_id = receipt.get("action_id")
    if not action_id or not str(action_id).isnumeric():
        raise PilotError("Submission receipt has no numeric action ID")
    _, _, manifest, payload, bundle = _verify_plan_bundle(args.plan, args.bundle)
    if str(receipt.get("bundle_sha256")) != bundle["sha256"]:
        raise PilotError("Submission receipt bundle differs from the pin")
    fireopal, qctrl_source = safe_base._authenticated_fireopal_for_retrieval(
        args.qctrl_notebook
    )
    raw = q40_validate._safe_provider_call(
        "Fire Opal q60 grid-d12 result retrieval",
        fireopal.get_result,
        str(action_id),
    )
    if not isinstance(raw, Mapping):
        raise PilotError("Fire Opal result payload is not a mapping")
    validation = validate_hardware_result(raw, manifest, payload)
    artifact = {
        "schema_version": SCHEMA_VERSION,
        "kind": "pbmc68k_q60_grid_d12_fireopal_hardware_result",
        "status": "retrieved_and_structurally_validated",
        "retrieved_at_utc": safe_base._utc_now(),
        "action_id": str(action_id),
        "receipt_path": str(args.receipt.resolve()),
        "bundle_sha256": bundle["sha256"],
        "qctrl_auth_source": str(qctrl_source),
        "api_calls": ["fireopal.get_result"],
        "submission_attempted_in_this_mode": False,
        "automatic_resubmission": False,
        **validation,
        "raw_result": safe_base._json_safe_redacted(raw),
        "claim_boundary": (
            "This is a Fire Opal hardware result with matched ideal and classical "
            "references. It can demonstrate hardware feasibility; quantum advantage "
            "requires the reported hardware statistics to support that stronger claim."
        ),
    }
    safe_base._atomic_write_json(args.result, artifact)
    receipt.update(
        {
            "status": "result_retrieved",
            "result_retrieved": True,
            "result_path": str(args.result.resolve()),
            "result_sha256": safe_base._sha256_file(args.result),
            "retrieved_at_utc": artifact["retrieved_at_utc"],
            "claim_boundary": (
                "Result retrieved and structurally validated; see the result "
                "artifact for classifier metrics and interpretation limits."
            ),
        }
    )
    safe_base._atomic_write_json(args.receipt, receipt)
    return artifact


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    plan = subparsers.add_parser("plan", help="local-only pinned pilot plan")
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
    retrieve = subparsers.add_parser("retrieve", help="retrieve saved action only")
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
        print("PBMC68k q60 grid-d12 Fire Opal pilot planned locally")
        print(f"- circuits: {plan['pilot']['measured_circuits']}")
        print(f"- total requested shots: {plan['pilot']['total_requested_shots']}")
        print("- provider calls: 0")
        print(f"- plan: {args.plan}")
        return 0
    if args.command == "submit":
        receipt = submit_pilot(args)
        print("Fire Opal q60 grid-d12 pilot submitted; no result wait performed")
        print(f"- action ID: {receipt['action_id']}")
        print(f"- receipt: {args.receipt}")
        return 0
    if args.command == "retrieve":
        artifact = retrieve_pilot(args)
        print("Fire Opal q60 grid-d12 result retrieved without resubmission")
        print(f"- action ID: {artifact['action_id']}")
        print(
            "- hardware fixed-test balanced accuracy: "
            f"{artifact['hardware_classifier']['fixed_test']['balanced_accuracy']:.6f}"
        )
        print(f"- result: {args.result}")
        return 0
    raise PilotError(f"Unsupported command: {args.command}")


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except PilotError as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(2) from None
