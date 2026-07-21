#!/usr/bin/env python3
"""Safe Fire Opal pilot for the frozen q60 PBMC68k module route.

``plan`` is provider-free.  ``submit`` is the only function that can call
``fireopal.execute`` and requires an exact phase-specific confirmation.
``retrieve`` reads only persisted action IDs and cannot submit or resubmit.
"""

from __future__ import annotations

import argparse
import gzip
import json
import math
import sys
import time
import uuid
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

import qiskit_qos_pbmc68k_q40_fireopal_validate as safe_validate
import qiskit_qos_pbmc68k_q60_module_fireopal_validate as validate
import qiskit_qos_pbmc68k_q60_module_pipeline as pipeline
import qiskit_qos_pbmc68k_q60_rx05_fireopal_pilot as safe_base


SCHEMA_VERSION = "1.0"
KIND_PREFIX = "pbmc68k_q60_coexpression_modules_b4_fireopal"
ARTIFACT_DIR = validate.ARTIFACT_DIR
CONFIRMATIONS = {
    "sentinel": "FIREOPAL_SUBMIT_Q60_MODULE_B4_SEED11_SENTINEL_192X128_IBM_FEZ_MAX50QS",
    "large": "FIREOPAL_SUBMIT_Q60_MODULE_B4_SEED11_LARGE_1536X128_IBM_FEZ_MAX400QS",
}
FULL_STUDY_QUANTUM_SECONDS_CAP = 450


PilotError = safe_base.PilotError


def default_path(phase: str, artifact: str) -> Path:
    suffixes = {
        "plan": "hardware_plan.json",
        "intent": "submission_intent.json",
        "receipt": "submission_receipt.json",
        "result": "hardware_result.json",
    }
    return ARTIFACT_DIR / f"pbmc68k_q60_modules_b4_seed11_{phase}_{suffixes[artifact]}"


def _load_bundle(
    path: Path, *, phase: str
) -> tuple[list[str], list[dict[str, Any]], dict[str, Any], dict[str, Any]]:
    scope = validate.phase_scope(phase)
    if not path.is_file():
        raise PilotError(f"QASM bundle is missing: {path}")
    try:
        with gzip.open(path, "rt", encoding="utf-8") as handle:
            payload = json.load(handle)
    except (OSError, json.JSONDecodeError) as exc:
        raise PilotError(f"Could not read QASM bundle ({type(exc).__name__})") from None
    if not isinstance(payload, Mapping):
        raise PilotError("QASM bundle root must be a mapping")
    config = payload.get("config", {})
    circuits = payload.get("circuits")
    batches = payload.get("batches")
    if (
        payload.get("kind")
        != "pbmc68k_q60_coexpression_modules_b4_numeric_qasm2_bundle"
        or not isinstance(circuits, list)
        or not isinstance(batches, list)
        or len(circuits) != int(scope["measured_circuits"])
        or int(config.get("qubits", -1)) != pipeline.QUBITS
        or int(config.get("block_count", -1)) != pipeline.BLOCK_COUNT
        or int(config.get("seed", -1)) != pipeline.SEED
        or str(config.get("backend")) != validate.BACKEND
        or int(config.get("shots", -1)) != validate.SHOTS
    ):
        raise PilotError("QASM bundle differs from the frozen phase")
    expected_batches = validate.batch_ranges(
        int(scope["measured_circuits"]), scope["batch_sizes"]
    )
    if [int(row.get("circuit_count", -1)) for row in batches] != [
        row["circuit_count"] for row in expected_batches
    ]:
        raise PilotError("QASM bundle batch partition changed")

    expected_mapping = [
        {"qubit": index, "clbit": index} for index in range(pipeline.QUBITS)
    ]
    qasms: list[str] = []
    manifest: list[dict[str, Any]] = []
    for expected_index, raw_row in enumerate(circuits):
        if not isinstance(raw_row, Mapping):
            raise PilotError("QASM circuit rows must be mappings")
        row = dict(raw_row)
        qasm = row.pop("qasm", None)
        expected_basis = validate.MEASUREMENT_BASES[expected_index % 3]
        expected_split = (
            "train" if expected_index // 3 < int(scope["train_samples"]) else "test"
        )
        logical = row.get("logical_metrics_before_measurement", {})
        metrics = row.get("metrics", {})
        if (
            not isinstance(qasm, str)
            or not qasm.startswith("OPENQASM 2.0;")
            or int(row.get("circuit_index", -1)) != expected_index
            or int(row.get("base_circuit_index", -1)) != expected_index // 3
            or int(row.get("seed", -1)) != pipeline.SEED
            or str(row.get("measurement_basis")) != expected_basis
            or str(row.get("split")) != expected_split
            or int(logical.get("depth", -1)) != pipeline.EXPECTED_LOGICAL_DEPTH
            or int(logical.get("two_qubit_gates", -1))
            != pipeline.EXPECTED_LOGICAL_TWO_QUBIT_GATES
            or int(metrics.get("num_qubits", -1)) != pipeline.QUBITS
            or int(metrics.get("num_clbits", -1)) != pipeline.QUBITS
            or row.get("virtual_qubits_only") is not True
            or row.get("all_parameters_numeric") is not True
            or row.get("round_trip_validated") is not True
            or row.get("labels_in_provider_payload") is not False
        ):
            raise PilotError("QASM manifest order or validation invariant changed")
        if safe_base._sha256_bytes(qasm.encode("utf-8")) != str(
            row.get("qasm_sha256", "")
        ):
            raise PilotError("QASM content differs from its manifest hash")
        try:
            parsed = safe_base.qasm2.loads(qasm)
        except Exception as exc:
            raise PilotError(
                f"QASM {expected_index} no longer parses ({type(exc).__name__})"
            ) from None
        actual_mapping = sorted(
            safe_base._measurement_mapping(parsed), key=lambda item: item["clbit"]
        )
        if (
            parsed.num_qubits != pipeline.QUBITS
            or parsed.num_clbits != pipeline.QUBITS
            or actual_mapping != expected_mapping
        ):
            raise PilotError("Parsed QASM register or measurement mapping changed")
        qasms.append(qasm)
        manifest.append(row)

    hashes = [str(row["qasm_sha256"]) for row in manifest]
    aggregate = safe_base._aggregate_qasm_hash(hashes)
    if aggregate != str(payload.get("aggregate_qasm_sha256")):
        raise PilotError("Aggregate QASM hash differs from the bundle")
    info = {
        "path": str(path.resolve()),
        "bytes": path.stat().st_size,
        "sha256": safe_base._sha256_file(path),
        "circuits": len(qasms),
        "aggregate_qasm_sha256": aggregate,
        "batches": [dict(row) for row in batches],
        "independent_qasm_parse_passed": True,
        "measurement_width_and_mapping_passed": True,
    }
    return qasms, manifest, dict(payload), info


def _validated_report(
    path: Path, bundle: Mapping[str, Any], *, phase: str
) -> dict[str, Any]:
    report = safe_base._load_json(path)
    local = report.get("local_validation", {})
    provider = report.get("provider_validation", {})
    recorded_bundle = report.get("qasm_bundle", {})
    scope = validate.phase_scope(phase)
    if (
        report.get("kind") != validate.KIND
        or report.get("status") != "pass"
        or report.get("phase") != phase
        or report.get("execution_attempted") is not False
        or int(report.get("quantum_seconds_used", -1)) != 0
        or local.get("passed") is not True
        or int(local.get("circuit_count", -1)) != int(scope["measured_circuits"])
        or str(local.get("aggregate_qasm_sha256"))
        != str(bundle["aggregate_qasm_sha256"])
        or provider.get("requested") is not True
        or provider.get("passed") is not True
        or provider.get("execution_attempted") is not False
        or provider.get("backend_supported") is not True
        or int(provider.get("circuits_validated", -1))
        != int(scope["measured_circuits"])
        or str(recorded_bundle.get("sha256")) != str(bundle["sha256"])
    ):
        raise PilotError("Fire Opal validate-only report is not submission-ready")
    return {
        "path": str(path.resolve()),
        "sha256": safe_base._sha256_file(path),
        "captured_at_utc": report.get("captured_at_utc"),
        "provider_validation_passed": True,
        "validation_action_ids": [
            row.get("validation_action_id") for row in provider.get("batches", [])
        ],
        "warning_count": len(provider.get("warnings", [])),
        "large_phase_gate": report.get("large_phase_gate"),
    }


def _sentinel_prerequisite(path: Path | None, phase: str) -> dict[str, Any] | None:
    if phase != "large":
        return None
    if path is None:
        raise PilotError("Large phase requires --sentinel-result")
    result = safe_base._load_json(path)
    if (
        result.get("kind") != f"{KIND_PREFIX}_hardware_result"
        or result.get("phase") != "sentinel"
        or result.get("status") != "retrieved_and_structurally_validated"
        or result.get("distribution_validation", {}).get("passed") is not True
        or result.get("observable_validation", {}).get("passed") is not True
    ):
        raise PilotError("Sentinel result is not a successful structural prerequisite")
    return {
        "path": str(path.resolve()),
        "sha256": safe_base._sha256_file(path),
        "action_ids": result.get("action_ids", []),
        "provider_reported_quantum_seconds": result.get(
            "provider_reported_quantum_seconds"
        ),
    }


def _phase_paths(args: argparse.Namespace) -> tuple[Path, Path, Path, Path, Path]:
    phase = str(args.phase)
    return (
        args.bundle or validate.default_bundle(phase),
        args.validation_report or validate.default_report(phase),
        args.plan or default_path(phase, "plan"),
        args.intent or default_path(phase, "intent"),
        args.receipt or default_path(phase, "receipt"),
    )


def plan_pilot(args: argparse.Namespace) -> dict[str, Any]:
    phase = str(args.phase)
    bundle_path, validation_path, plan_path, _, _ = _phase_paths(args)
    if plan_path.exists() and not args.force:
        raise PilotError(f"Refusing to overwrite existing artifact: {plan_path}")
    started = time.perf_counter()
    _, manifest, payload, bundle = _load_bundle(bundle_path, phase=phase)
    validation_report = _validated_report(
        validation_path, bundle, phase=phase
    )
    sentinel = _sentinel_prerequisite(args.sentinel_result, phase)
    scope = validate.phase_scope(phase)
    panel = pipeline.observable_mappings()
    qasm_hashes = [str(row["qasm_sha256"]) for row in manifest]
    estimate = scope["quantum_seconds_estimate"]
    plan = {
        "schema_version": SCHEMA_VERSION,
        "kind": f"{KIND_PREFIX}_{phase}_hardware_plan",
        "status": "ready_for_separately_confirmed_submission",
        "captured_at_utc": safe_base._utc_now(),
        "environment": safe_validate.runtime_environment(),
        "submission_attempted": False,
        "provider_calls": [],
        "phase": phase,
        "hardware_scope": {
            "dataset": "PBMC68k",
            "architecture": pipeline.ARCHITECTURE,
            "seed": pipeline.SEED,
            "qubits": pipeline.QUBITS,
            "backend": validate.BACKEND,
            "train_samples": scope["train_samples"],
            "test_samples": scope["test_samples"],
            "base_circuits": scope["base_circuits"],
            "measured_circuits": scope["measured_circuits"],
            "batch_sizes": scope["batch_sizes"],
            "shots_per_circuit": validate.SHOTS,
            "total_requested_shots": int(scope["measured_circuits"]) * validate.SHOTS,
        },
        "quantum_seconds_budget": {
            "basis": "scaled from provider-reported 26 quantum seconds for the matching 40q 192x128 run",
            "estimated_low": estimate["low"],
            "estimated_central": estimate["central"],
            "estimated_high": estimate["high"],
            "phase_cap": scope["phase_cap"],
            "full_study_cap": FULL_STUDY_QUANTUM_SECONDS_CAP,
            "cap_is_declared_not_provider_enforced": True,
            "retries_included": False,
        },
        "validation_report": validation_report,
        "qasm_bundle": bundle,
        "qasm_hashes": qasm_hashes,
        "specification": payload.get("specification"),
        "large_phase_gate": payload.get("large_phase_gate"),
        "sentinel_prerequisite": sentinel,
        "submission_boundary": {
            "separate_phase_authorization_required": True,
            "explicit_submit_subcommand_required": True,
            "exact_confirmation_required": True,
            "confirmation_literal": CONFIRMATIONS[phase],
            "intent_lock_written_before_each_execute": True,
            "each_action_id_persisted_before_next_batch": True,
            "automatic_resubmission": False,
            "backend_switch_allowed": False,
            "shot_increase_allowed": False,
            "result_wait_during_submit": False,
        },
        "predeclared_readout": {
            "observable_count": len(panel),
            "measurement_bases": list(validate.MEASUREMENT_BASES),
            "bit_order": "Qiskit little-endian: rightmost bit is qubit 0",
            "model_selection": "four-fold stratified training-only CV, seed 6011",
            "matched_classical_models": ["linear SVC", "RBF SVC"],
        },
        "elapsed_seconds": time.perf_counter() - started,
        "claim_boundary": (
            "This plan fixes a hardware-feasibility experiment. Only a later frozen "
            "large-split comparison can become a task-bound empirical advantage candidate."
        ),
    }
    safe_base._atomic_write_json(plan_path, plan)
    return plan


def _verify_plan_bundle(
    plan_path: Path, bundle_path: Path, validation_path: Path, *, phase: str
) -> tuple[dict[str, Any], list[str], list[dict[str, Any]], dict[str, Any]]:
    plan = safe_base._load_json(plan_path)
    qasms, manifest, _, bundle = _load_bundle(bundle_path, phase=phase)
    if (
        plan.get("kind") != f"{KIND_PREFIX}_{phase}_hardware_plan"
        or plan.get("status") != "ready_for_separately_confirmed_submission"
        or plan.get("phase") != phase
        or plan.get("submission_attempted") is not False
    ):
        raise PilotError("Hardware plan is not ready for separate confirmation")
    validation_report = _validated_report(validation_path, bundle, phase=phase)
    planned_validation = plan.get("validation_report", {})
    planned_bundle = plan.get("qasm_bundle", {})
    if (
        str(planned_validation.get("sha256")) != validation_report["sha256"]
        or str(planned_bundle.get("sha256")) != bundle["sha256"]
        or str(planned_bundle.get("aggregate_qasm_sha256"))
        != bundle["aggregate_qasm_sha256"]
        or [str(value) for value in plan.get("qasm_hashes", [])]
        != [str(row["qasm_sha256"]) for row in manifest]
    ):
        raise PilotError("Plan, validation, and ordered QASM bundle identities differ")
    if phase == "large":
        prerequisite = plan.get("sentinel_prerequisite")
        if not isinstance(prerequisite, Mapping):
            raise PilotError("Large plan lost its sentinel prerequisite")
        sentinel_path = Path(str(prerequisite.get("path", "")))
        if (
            not sentinel_path.is_file()
            or safe_base._sha256_file(sentinel_path)
            != str(prerequisite.get("sha256", ""))
        ):
            raise PilotError("Large plan sentinel prerequisite changed after planning")
    return plan, qasms, manifest, bundle


def submit_pilot(args: argparse.Namespace) -> dict[str, Any]:
    """Submit the frozen phase once; persist every action before continuing."""

    phase = str(args.phase)
    if args.confirm_submit != CONFIRMATIONS[phase]:
        raise PilotError(
            f"Submission requires --confirm-submit {CONFIRMATIONS[phase]}"
        )
    bundle_path, validation_path, plan_path, intent_path, receipt_path = _phase_paths(args)
    if intent_path.exists() or receipt_path.exists():
        raise PilotError("Submission intent or receipt exists; refusing resubmission")
    plan, qasms, _, bundle = _verify_plan_bundle(
        plan_path, bundle_path, validation_path, phase=phase
    )
    scope = validate.phase_scope(phase)
    batches = bundle["batches"]
    intent_id = str(uuid.uuid4())
    intent = {
        "schema_version": SCHEMA_VERSION,
        "kind": f"{KIND_PREFIX}_{phase}_submission_intent",
        "intent_id": intent_id,
        "created_at_utc": safe_base._utc_now(),
        "status": "preflight_locked",
        "phase": phase,
        "plan_path": str(plan_path.resolve()),
        "plan_sha256": safe_base._sha256_file(plan_path),
        "bundle_path": str(bundle_path.resolve()),
        "bundle_sha256": bundle["sha256"],
        "aggregate_qasm_sha256": bundle["aggregate_qasm_sha256"],
        "backend": validate.BACKEND,
        "circuit_count": scope["measured_circuits"],
        "batch_sizes": scope["batch_sizes"],
        "shots_per_circuit": validate.SHOTS,
        "total_requested_shots": int(scope["measured_circuits"]) * validate.SHOTS,
        "quantum_seconds_phase_cap": scope["phase_cap"],
        "execution_attempted": False,
        "completed_batch_actions": [],
        "automatic_resubmission": False,
    }
    safe_base._atomic_write_json(intent_path, intent)
    receipt: dict[str, Any] | None = None
    try:
        fireopal, credentials, credential_source, qctrl_source = (
            safe_validate._fire_opal_credentials_from_source(
                args.qiskit_account, args.qctrl_notebook, args.instance
            )
        )
        devices = safe_validate._workflow_result(
            safe_validate._safe_provider_call(
                "Fire Opal supported-device discovery",
                fireopal.show_supported_devices,
                credentials=credentials,
            )
        )
        supported = [str(value) for value in devices.get("supported_devices", [])]
        if validate.BACKEND not in supported:
            intent.update(
                {
                    "status": "backend_not_supported",
                    "backend_supported": False,
                    "updated_at_utc": safe_base._utc_now(),
                }
            )
            safe_base._atomic_write_json(intent_path, intent)
            raise PilotError("ibm_fez is not currently supported by Fire Opal")
        receipt = {
            "schema_version": SCHEMA_VERSION,
            "kind": f"{KIND_PREFIX}_{phase}_submission_receipt",
            "status": "submission_in_progress",
            "submitted_at_utc": safe_base._utc_now(),
            "phase": phase,
            "intent_id": intent_id,
            "backend": validate.BACKEND,
            "circuit_count": scope["measured_circuits"],
            "batch_sizes": scope["batch_sizes"],
            "shots_per_circuit": validate.SHOTS,
            "total_requested_shots": int(scope["measured_circuits"]) * validate.SHOTS,
            "bundle_path": str(bundle_path.resolve()),
            "bundle_sha256": bundle["sha256"],
            "aggregate_qasm_sha256": bundle["aggregate_qasm_sha256"],
            "plan_path": str(plan_path.resolve()),
            "plan_sha256": safe_base._sha256_file(plan_path),
            "credential_source": safe_base._credential_source_labels(credential_source),
            "qctrl_auth_source": str(qctrl_source),
            "batch_actions": [],
            "api_calls": ["fireopal.show_supported_devices"],
            "execution_attempted": False,
            "result_waited_during_submit": False,
            "result_retrieved": False,
            "quantum_seconds_used": None,
            "quantum_seconds_phase_cap": scope["phase_cap"],
            "automatic_resubmission": False,
        }
        safe_base._atomic_write_json(receipt_path, receipt)
        for batch in batches:
            batch_index = int(batch["batch_index"])
            start = int(batch["start_circuit_index"])
            stop = int(batch["stop_circuit_index_exclusive"])
            intent.update(
                {
                    "status": "execute_call_started",
                    "backend_supported": True,
                    "execution_attempted": True,
                    "active_batch_index": batch_index,
                    "updated_at_utc": safe_base._utc_now(),
                }
            )
            safe_base._atomic_write_json(intent_path, intent)
            job = safe_validate._safe_provider_call(
                "Fire Opal q60 module hardware submission",
                fireopal.execute,
                circuits=list(qasms[start:stop]),
                shot_count=validate.SHOTS,
                credentials=credentials,
                backend_name=validate.BACKEND,
                parameters=None,
            )
            action_id = getattr(job, "action_id", None)
            if action_id is None:
                intent.update(
                    {
                        "status": "submitted_but_action_id_missing",
                        "ambiguous_batch_index": batch_index,
                        "updated_at_utc": safe_base._utc_now(),
                    }
                )
                safe_base._atomic_write_json(intent_path, intent)
                raise PilotError(
                    "Execute may have been accepted but no action ID returned; do not resubmit"
                )
            action = {
                **dict(batch),
                "action_id": str(action_id),
                "submitted_at_utc": safe_base._utc_now(),
            }
            receipt["batch_actions"].append(action)
            receipt["api_calls"].append("fireopal.execute")
            receipt["execution_attempted"] = True
            safe_base._atomic_write_json(receipt_path, receipt)
            intent["completed_batch_actions"] = list(receipt["batch_actions"])
            safe_base._atomic_write_json(intent_path, intent)
        receipt.update(
            {
                "status": "submitted_not_retrieved",
                "submission_completed_at_utc": safe_base._utc_now(),
                "action_ids": [row["action_id"] for row in receipt["batch_actions"]],
                "claim_boundary": "Submission receipt only; no result retrieved yet.",
            }
        )
        safe_base._atomic_write_json(receipt_path, receipt)
        intent.update(
            {
                "status": "receipt_persisted",
                "active_batch_index": None,
                "receipt_path": str(receipt_path.resolve()),
                "updated_at_utc": safe_base._utc_now(),
            }
        )
        safe_base._atomic_write_json(intent_path, intent)
        return receipt
    except (safe_validate.RunnerError, PilotError) as exc:
        if intent_path.exists():
            persisted = safe_base._load_json(intent_path)
            if persisted.get("status") not in {
                "backend_not_supported",
                "submitted_but_action_id_missing",
                "receipt_persisted",
            }:
                persisted.update(
                    {
                        "status": "submit_failed_sanitized_no_automatic_retry",
                        "failure_type": type(exc).__name__,
                        "updated_at_utc": safe_base._utc_now(),
                    }
                )
                safe_base._atomic_write_json(intent_path, persisted)
        if receipt is not None:
            receipt.update(
                {
                    "status": "partial_or_failed_submission_do_not_resubmit",
                    "failure_type": type(exc).__name__,
                    "updated_at_utc": safe_base._utc_now(),
                }
            )
            safe_base._atomic_write_json(receipt_path, receipt)
        raise PilotError(str(exc)) from None


def _observable_panel() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for index, mapping in enumerate(pipeline.observable_mappings()):
        bases = {str(value) for value in mapping.values()}
        if len(bases) != 1 or next(iter(bases)) not in validate.MEASUREMENT_BASES:
            raise PilotError("Observable is not recoverable from one global basis")
        rows.append(
            {
                "observable_index": int(index),
                "measurement_basis": next(iter(bases)),
                "pauli_mapping": [
                    {"qubit": int(qubit), "pauli": str(pauli)}
                    for qubit, pauli in sorted(mapping.items())
                ],
            }
        )
    return rows


def _pauli_expectation(
    distribution: Mapping[str, float], mapping: Sequence[Mapping[str, Any]]
) -> float:
    qubits = [int(item["qubit"]) for item in mapping]
    if not qubits or len(qubits) != len(set(qubits)):
        raise PilotError("Observable support is invalid")
    total = float(sum(distribution.values()))
    expectation = 0.0
    for bitstring, weight in distribution.items():
        parity = sum(
            int(bitstring[pipeline.QUBITS - 1 - qubit]) for qubit in qubits
        ) % 2
        expectation += (1.0 if parity == 0 else -1.0) * float(weight) / total
    if not math.isfinite(expectation) or abs(expectation) > 1.0 + 1e-9:
        raise PilotError("Pauli expectation is outside [-1, 1]")
    return float(max(-1.0, min(1.0, expectation)))


def validate_hardware_results(
    raw_batches: Sequence[Mapping[str, Any]],
    manifest: Sequence[Mapping[str, Any]],
    *,
    phase: str,
) -> dict[str, Any]:
    scope = validate.phase_scope(phase)
    distributions: list[Mapping[str, Any]] = []
    for raw in raw_batches:
        distributions.extend(safe_base._result_distributions(raw))
    expected = int(scope["measured_circuits"])
    if len(distributions) != expected or len(manifest) != expected:
        raise PilotError("Fire Opal results and manifest have different circuit counts")
    checks: list[dict[str, Any]] = []
    by_base_basis: dict[tuple[int, str], dict[str, float]] = {}
    for distribution, row in zip(distributions, manifest, strict=True):
        cleaned, check = safe_base._validated_distribution(
            distribution, num_qubits=pipeline.QUBITS, shots=validate.SHOTS
        )
        key = (int(row["base_circuit_index"]), str(row["measurement_basis"]))
        if key in by_base_basis:
            raise PilotError("Duplicate base-circuit measurement result")
        by_base_basis[key] = cleaned
        checks.append(check)
    panel = _observable_panel()
    feature_matrix = np.empty(
        (int(scope["base_circuits"]), len(panel)), dtype=np.float64
    )
    for base_index in range(int(scope["base_circuits"])):
        for observable in panel:
            basis = str(observable["measurement_basis"])
            distribution = by_base_basis.get((base_index, basis))
            if distribution is None:
                raise PilotError("Missing measurement-basis result")
            feature_matrix[base_index, int(observable["observable_index"])] = (
                _pauli_expectation(distribution, observable["pauli_mapping"])
            )
    if not np.all(np.isfinite(feature_matrix)) or np.max(np.abs(feature_matrix)) > 1.0 + 1e-9:
        raise PilotError("Hardware features are non-finite or outside [-1, 1]")
    base_metadata: dict[int, Mapping[str, Any]] = {}
    for row in manifest:
        base_metadata.setdefault(int(row["base_circuit_index"]), row)
    feature_rows = [
        {
            "base_circuit_index": int(index),
            "split": str(base_metadata[index]["split"]),
            "sample_position": int(base_metadata[index]["sample_position"]),
            "source_row_index": int(base_metadata[index]["source_row_index"]),
            "label_for_matched_analysis": int(
                base_metadata[index]["label_for_local_matched_analysis_only"]
            ),
            "features": [float(value) for value in feature_matrix[index]],
        }
        for index in range(int(scope["base_circuits"]))
    ]
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
            "observable_count_per_sample": len(panel),
            "feature_value_count": int(feature_matrix.size),
            "minimum": float(np.min(feature_matrix)),
            "maximum": float(np.max(feature_matrix)),
            "all_within_minus_one_plus_one": True,
            "bit_order": "Qiskit little-endian: rightmost bit is qubit 0",
        },
        "hardware_feature_rows": feature_rows,
        "classifier_analysis_performed": False,
    }


def _quantum_seconds(value: Any, *, parent: str = "") -> list[float]:
    rows: list[float] = []
    if isinstance(value, Mapping):
        for key, item in value.items():
            normalized = str(key).lower().replace("-", "_").replace(" ", "_")
            if normalized in {"quantum_seconds", "quantum_seconds_used", "qpu_seconds"}:
                try:
                    number = float(item)
                except (TypeError, ValueError):
                    number = math.nan
                if math.isfinite(number) and number >= 0.0:
                    rows.append(number)
            else:
                rows.extend(_quantum_seconds(item, parent=normalized))
    elif isinstance(value, (list, tuple)):
        for item in value:
            rows.extend(_quantum_seconds(item, parent=parent))
    return rows


def retrieve_pilot(args: argparse.Namespace) -> dict[str, Any]:
    phase = str(args.phase)
    bundle_path, validation_path, plan_path, _, receipt_path = _phase_paths(args)
    result_path = args.result or default_path(phase, "result")
    if result_path.exists() and not args.force:
        raise PilotError(f"Refusing to overwrite result artifact: {result_path}")
    receipt = safe_base._load_json(receipt_path)
    scope = validate.phase_scope(phase)
    actions = receipt.get("batch_actions")
    if (
        receipt.get("kind") != f"{KIND_PREFIX}_{phase}_submission_receipt"
        or receipt.get("status") != "submitted_not_retrieved"
        or not isinstance(actions, list)
        or len(actions) != len(scope["batch_sizes"])
    ):
        raise PilotError("Submission receipt is incomplete or has an unexpected kind")
    plan, _, manifest, bundle = _verify_plan_bundle(
        plan_path, bundle_path, validation_path, phase=phase
    )
    if (
        str(receipt.get("bundle_sha256")) != bundle["sha256"]
        or str(receipt.get("plan_sha256")) != safe_base._sha256_file(plan_path)
    ):
        raise PilotError("Submission receipt differs from the frozen plan or bundle")
    fireopal, qctrl_source = safe_base._authenticated_fireopal_for_retrieval(
        args.qctrl_notebook
    )
    raw_batches: list[Mapping[str, Any]] = []
    for action in actions:
        action_id = str(action.get("action_id", ""))
        if not action_id.isnumeric():
            raise PilotError("Submission receipt contains a non-numeric action ID")
        raw = safe_validate._safe_provider_call(
            "Fire Opal q60 module result retrieval",
            fireopal.get_result,
            action_id,
        )
        if not isinstance(raw, Mapping):
            raise PilotError("Fire Opal result payload is not a mapping")
        raw_batches.append(raw)
    validation_result = validate_hardware_results(
        raw_batches, manifest, phase=phase
    )
    quantum_seconds_by_batch: list[dict[str, Any]] = []
    unambiguous_batch_values: list[float] = []
    for batch_index, raw in enumerate(raw_batches):
        values = sorted(set(_quantum_seconds(raw)))
        resolved = values[0] if len(values) == 1 else None
        quantum_seconds_by_batch.append(
            {
                "batch_index": int(batch_index),
                "fields_found": values,
                "resolved_quantum_seconds": resolved,
                "ambiguous": len(values) > 1,
            }
        )
        if resolved is not None:
            unambiguous_batch_values.append(float(resolved))
    quantum_seconds_total = (
        float(sum(unambiguous_batch_values))
        if len(unambiguous_batch_values) == len(raw_batches)
        else None
    )
    artifact = {
        "schema_version": SCHEMA_VERSION,
        "kind": f"{KIND_PREFIX}_hardware_result",
        "status": "retrieved_and_structurally_validated",
        "retrieved_at_utc": safe_base._utc_now(),
        "phase": phase,
        "action_ids": [str(row["action_id"]) for row in actions],
        "receipt_path": str(receipt_path.resolve()),
        "bundle_sha256": bundle["sha256"],
        "plan_sha256": safe_base._sha256_file(plan_path),
        "qctrl_auth_source": str(qctrl_source),
        "api_calls": ["fireopal.get_result" for _ in actions],
        "submission_attempted_in_this_mode": False,
        "automatic_resubmission": False,
        "provider_reported_quantum_seconds": quantum_seconds_total,
        "quantum_seconds_by_batch": quantum_seconds_by_batch,
        "quantum_seconds_cap_exceeded": (
            bool(quantum_seconds_total > float(scope["phase_cap"]))
            if quantum_seconds_total is not None
            else None
        ),
        "declared_phase_cap": scope["phase_cap"],
        **validation_result,
        "raw_result_batches": [safe_base._json_safe_redacted(raw) for raw in raw_batches],
        "claim_boundary": (
            "This is a structurally validated Fire Opal hardware result. Frozen "
            "training-only classifier analysis remains a separate local-only step."
        ),
    }
    safe_base._atomic_write_json(result_path, artifact)
    receipt.update(
        {
            "status": "result_retrieved",
            "result_retrieved": True,
            "result_path": str(result_path.resolve()),
            "result_sha256": safe_base._sha256_file(result_path),
            "retrieved_at_utc": artifact["retrieved_at_utc"],
            "quantum_seconds_used": quantum_seconds_total,
        }
    )
    safe_base._atomic_write_json(receipt_path, receipt)
    return artifact


def _common_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--phase", choices=("sentinel", "large"), default="sentinel")
    parser.add_argument("--bundle", type=Path)
    parser.add_argument("--validation-report", type=Path)
    parser.add_argument("--plan", type=Path)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    plan = subparsers.add_parser("plan", help="create a provider-free locked hardware plan")
    _common_arguments(plan)
    plan.add_argument("--sentinel-result", type=Path)
    plan.add_argument("--intent", type=Path)
    plan.add_argument("--receipt", type=Path)
    plan.add_argument("--force", action="store_true")
    submit = subparsers.add_parser("submit", help="submit the frozen phase once")
    _common_arguments(submit)
    submit.add_argument("--intent", type=Path)
    submit.add_argument("--receipt", type=Path)
    submit.add_argument("--qiskit-account", default="default-ibm-cloud")
    submit.add_argument("--qctrl-notebook", type=Path)
    submit.add_argument("--instance")
    submit.add_argument("--confirm-submit", default="")
    retrieve = subparsers.add_parser("retrieve", help="retrieve persisted actions only")
    _common_arguments(retrieve)
    retrieve.add_argument("--intent", type=Path)
    retrieve.add_argument("--receipt", type=Path)
    retrieve.add_argument("--result", type=Path)
    retrieve.add_argument("--qctrl-notebook", type=Path)
    retrieve.add_argument("--force", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(list(sys.argv[1:] if argv is None else argv))
    if args.command == "plan":
        plan = plan_pilot(args)
        print("PBMC68k q60 module Fire Opal hardware phase planned locally")
        print(f"- phase: {plan['phase']}")
        print(f"- circuits: {plan['hardware_scope']['measured_circuits']}")
        print(
            "- estimated quantum seconds: "
            f"{plan['quantum_seconds_budget']['estimated_low']}-"
            f"{plan['quantum_seconds_budget']['estimated_high']} "
            f"(central {plan['quantum_seconds_budget']['estimated_central']})"
        )
        print("- provider calls: 0")
        print(f"- plan: {args.plan or default_path(args.phase, 'plan')}")
        return 0
    if args.command == "submit":
        receipt = submit_pilot(args)
        print("Fire Opal q60 module phase submitted; no result wait performed")
        print(f"- phase: {receipt['phase']}")
        print(f"- action IDs: {receipt['action_ids']}")
        print(f"- receipt: {args.receipt or default_path(args.phase, 'receipt')}")
        return 0
    artifact = retrieve_pilot(args)
    print("Fire Opal q60 module result retrieved without resubmission")
    print(f"- phase: {artifact['phase']}")
    print(f"- action IDs: {artifact['action_ids']}")
    print(
        f"- provider-reported quantum seconds: {artifact['provider_reported_quantum_seconds']}"
    )
    print(f"- result: {args.result or default_path(args.phase, 'result')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
