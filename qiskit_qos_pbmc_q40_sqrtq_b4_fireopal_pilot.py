#!/usr/bin/env python3
"""Safe Fire Opal hardware pilot for the frozen q40 sqrt(q), B=4 batch.

``plan`` is provider-free. ``submit`` is the only mode that can call
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
import qiskit_qos_pbmc_coherent_stream_hardness_screen as coherent


SCHEMA_VERSION = "1.0"
PILOT_SEED = 11
PILOT_QUBITS = 40
PILOT_BLOCKS = 4
PILOT_BASE_CIRCUITS = 64
PILOT_CIRCUITS = 192
PILOT_SHOTS = 128
PILOT_BACKEND = "ibm_fez"
PILOT_ARCHITECTURE = "coherent_stream_b4_width_scaled_entangler"
PILOT_SCALE_LAW = "sqrt_q"
PILOT_MEASUREMENT_BASES = ("X", "Y", "Z")
SUBMIT_CONFIRMATION = "FIREOPAL_SUBMIT_Q40_SQRTQ_B4_SEED11_192X128_IBM_FEZ"

PINNED_VALIDATION_SHA256 = (
    "1de0dd6eaa95c03fafedbf1edc2669c34d1e9c25ee43d508f0b90488af631d18"
)
PINNED_BUNDLE_SHA256 = (
    "1251e74493acd198b7766abfcf75995ccb188a14dde886bcbb07ec4af6232eeb"
)
PINNED_AGGREGATE_QASM_SHA256 = (
    "f2571d6e35a7c2d4b11b66d70158bccec32484ec347f6ce1d66e3f2cf5383c92"
)

ARTIFACT_DIR = Path("fire_opal_pbmc68k_q40_sqrtq_b4")
DEFAULT_VALIDATION = ARTIFACT_DIR / (
    "pbmc68k_q40_sqrtq_b4_seed11_fireopal_validate.json"
)
DEFAULT_BUNDLE = ARTIFACT_DIR / (
    "pbmc68k_q40_sqrtq_b4_seed11_fireopal_qasm2.json.gz"
)
DEFAULT_PLAN = ARTIFACT_DIR / (
    "pbmc68k_q40_sqrtq_b4_seed11_fireopal_hardware_plan.json"
)
DEFAULT_INTENT = ARTIFACT_DIR / (
    "pbmc68k_q40_sqrtq_b4_seed11_fireopal_submission_intent.json"
)
DEFAULT_RECEIPT = ARTIFACT_DIR / (
    "pbmc68k_q40_sqrtq_b4_seed11_fireopal_submission_receipt.json"
)
DEFAULT_RESULT = ARTIFACT_DIR / (
    "pbmc68k_q40_sqrtq_b4_seed11_fireopal_hardware_result.json"
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
    if payload.get("kind") != "pbmc68k_q40_sqrtq_b4_seed11_numeric_qasm2_batch":
        raise PilotError("Unexpected QASM bundle kind")
    config = payload.get("config")
    circuits = payload.get("circuits")
    if not isinstance(config, Mapping) or not isinstance(circuits, list):
        raise PilotError("QASM bundle lacks configuration or circuits")
    if (
        len(circuits) != PILOT_CIRCUITS
        or int(config.get("qubits", -1)) != PILOT_QUBITS
        or int(config.get("block_count", -1)) != PILOT_BLOCKS
        or int(config.get("seed", -1)) != PILOT_SEED
        or str(config.get("scale_law")) != PILOT_SCALE_LAW
        or str(config.get("backend")) != PILOT_BACKEND
    ):
        raise PilotError("QASM bundle configuration differs from the frozen pilot")

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
        expected_split = "train" if expected_index // 3 < 32 else "test"
        if (
            int(row.get("circuit_index", -1)) != expected_index
            or int(row.get("seed", -1)) != PILOT_SEED
            or int(row.get("base_circuit_index", -1)) != expected_index // 3
            or str(row.get("measurement_basis")) != expected_basis
            or str(row.get("split")) != expected_split
        ):
            raise PilotError("Pilot circuit or train/test X/Y/Z ordering changed")
        metrics = row.get("metrics")
        logical = row.get("logical_metrics_before_measurement")
        if (
            not isinstance(metrics, Mapping)
            or not isinstance(logical, Mapping)
            or int(metrics.get("num_qubits", -1)) != PILOT_QUBITS
            or int(metrics.get("num_clbits", -1)) != PILOT_QUBITS
            or int(logical.get("depth", -1)) != 20
            or int(logical.get("two_qubit_gates", -1)) != 87
        ):
            raise PilotError("Pilot bundle contains a circuit with wrong metrics")
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
    if report.get("kind") != "pbmc68k_q40_sqrtq_b4_fireopal_validate_only":
        raise PilotError("Unexpected q40 sqrt(q) validation report kind")
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
        or int(config.get("block_count", -1)) != PILOT_BLOCKS
        or str(config.get("backend")) != PILOT_BACKEND
        or int(config.get("shots_per_circuit_for_future_hardware", -1))
        != PILOT_SHOTS
        or str(config.get("architecture")) != PILOT_ARCHITECTURE
        or str(config.get("scale_law")) != PILOT_SCALE_LAW
    ):
        raise PilotError("Validation configuration differs from the pilot")
    if (
        local.get("passed") is not True
        or int(local.get("circuit_count", -1)) != PILOT_CIRCUITS
        or str(local.get("aggregate_qasm_sha256"))
        != PINNED_AGGREGATE_QASM_SHA256
        or list(local.get("logical_depths", [])) != [20]
        or list(local.get("logical_two_qubit_gate_counts", [])) != [87]
    ):
        raise PilotError("Local validation or aggregate QASM pin changed")
    if (
        provider.get("requested") is not True
        or provider.get("passed") is not True
        or provider.get("execution_attempted") is not False
        or provider.get("backend_supported") is not True
        or str(provider.get("backend")) != PILOT_BACKEND
        or int(provider.get("circuits_validated", -1)) != PILOT_CIRCUITS
        or provider.get("errors") not in ([], None)
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
        "captured_at_utc": report.get("captured_at_utc"),
        "provider_validation_passed": True,
        "warning_count": len(warnings),
        "warning_categories": dict(sorted(categories.items())),
        "calibration_warning_acknowledged_by_later_hardware_authorization": True,
        "logical_depth": 20,
        "logical_two_qubit_gates": 87,
        "max_payload_depth": int(local["max_payload_depth"]),
        "total_qasm_bytes": int(local["total_qasm_bytes"]),
    }


def _observable_panel() -> list[dict[str, Any]]:
    mappings = coherent.grid_aligned_mappings(PILOT_QUBITS)
    rows: list[dict[str, Any]] = []
    for index, mapping in enumerate(mappings):
        bases = {str(value) for value in mapping.values()}
        if len(bases) != 1 or next(iter(bases)) not in PILOT_MEASUREMENT_BASES:
            raise PilotError("Grid observable is not recoverable from one global basis")
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
    if len(rows) != 405:
        raise PilotError("Frozen q40 grid panel must contain 405 observables")
    return rows


def plan_pilot(args: argparse.Namespace) -> dict[str, Any]:
    if args.plan.exists() and not args.force:
        raise PilotError(f"Refusing to overwrite existing artifact: {args.plan}")
    started = time.perf_counter()
    _, manifest, _, bundle = _load_bundle(args.bundle)
    if (
        bundle["sha256"] != PINNED_BUNDLE_SHA256
        or bundle["aggregate_qasm_sha256"] != PINNED_AGGREGATE_QASM_SHA256
    ):
        raise PilotError("QASM bundle differs from the validated pins")
    validation = _validated_report(args.validation_report, bundle)
    panel = _observable_panel()
    qasm_hashes = [str(row["qasm_sha256"]) for row in manifest]
    plan = {
        "schema_version": SCHEMA_VERSION,
        "kind": "pbmc68k_q40_sqrtq_b4_fireopal_seed11_hardware_pilot_plan",
        "status": "authorized_and_ready_for_confirmed_submission",
        "captured_at_utc": safe_base._utc_now(),
        "environment": q40_validate.runtime_environment(),
        "user_authorization": "explicit_hardware_run_authorization_received",
        "submission_attempted": False,
        "provider_calls": [],
        "pilot": {
            "dataset": "PBMC68k",
            "architecture": PILOT_ARCHITECTURE,
            "scale_law": PILOT_SCALE_LAW,
            "block_count": PILOT_BLOCKS,
            "seed": PILOT_SEED,
            "qubits": PILOT_QUBITS,
            "backend": PILOT_BACKEND,
            "train_samples": 32,
            "test_samples": 32,
            "logical_base_circuits": PILOT_BASE_CIRCUITS,
            "measurement_bases": list(PILOT_MEASUREMENT_BASES),
            "measured_circuits": PILOT_CIRCUITS,
            "shots_per_circuit": PILOT_SHOTS,
            "total_requested_shots": PILOT_CIRCUITS * PILOT_SHOTS,
        },
        "validation_report": validation,
        "qasm_bundle": bundle,
        "qasm_hashes": qasm_hashes,
        "submission_boundary": {
            "explicit_submit_subcommand_required": True,
            "exact_confirmation_required": True,
            "confirmation_literal": SUBMIT_CONFIRMATION,
            "intent_lock_written_before_execute": True,
            "action_id_persisted_before_any_result_wait": True,
            "automatic_resubmission": False,
            "result_wait_during_submit": False,
        },
        "predeclared_readout": {
            "observable_family": "all homogeneous X/Y/Z grid-aligned one- and two-qubit Paulis",
            "observable_count": len(panel),
            "observable_panel": panel,
            "bit_order": "Qiskit little-endian: rightmost bit is qubit 0",
            "primary_metric": "fixed-test balanced accuracy",
            "model_selection": "four-fold stratified training-only CV, seed 6011",
            "test_set_used_once_after_training_only_model_selection": True,
            "classical_frontier_must_be_recomputed_on_identical_32/32_split": True,
        },
        "elapsed_seconds": time.perf_counter() - started,
        "claim_boundary": (
            "This is an authorized hardware-feasibility experiment for a label-free "
            "structural candidate. Validation warnings report current device calibration "
            "risk. Submission and any later result do not by themselves prove classical "
            "hardness, predictive superiority, or quantum advantage."
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
        != "pbmc68k_q40_sqrtq_b4_fireopal_seed11_hardware_pilot_plan"
        or plan.get("status") != "authorized_and_ready_for_confirmed_submission"
        or plan.get("user_authorization")
        != "explicit_hardware_run_authorization_received"
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
    """Submit exactly once, persist the action ID, and never wait for a result."""

    if args.confirm_submit != SUBMIT_CONFIRMATION:
        raise PilotError(f"Submission requires --confirm-submit {SUBMIT_CONFIRMATION}")
    if args.intent.exists() or args.receipt.exists():
        raise PilotError("Submission intent or receipt exists; refusing resubmission")
    _, qasms, _, _, bundle = _verify_plan_bundle(args.plan, args.bundle)
    intent_id = str(uuid.uuid4())
    intent = {
        "schema_version": SCHEMA_VERSION,
        "kind": "pbmc68k_q40_sqrtq_b4_fireopal_submission_intent",
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
            "Fire Opal q40 sqrt(q) B=4 seed-11 hardware submission",
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
            "kind": "pbmc68k_q40_sqrtq_b4_fireopal_submission_receipt",
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
        raise PilotError("Observable has invalid support")
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
) -> dict[str, Any]:
    distributions = safe_base._result_distributions(raw)
    if len(distributions) != PILOT_CIRCUITS or len(manifest) != PILOT_CIRCUITS:
        raise PilotError("Fire Opal result and manifest must each contain 192 circuits")
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

    panel = _observable_panel()
    feature_matrix = np.empty((PILOT_BASE_CIRCUITS, len(panel)), dtype=np.float64)
    for base_index in range(PILOT_BASE_CIRCUITS):
        for observable in panel:
            basis = str(observable["measurement_basis"])
            distribution = by_base_basis.get((base_index, basis))
            if distribution is None:
                raise PilotError("Missing measurement-basis result")
            feature_matrix[base_index, int(observable["observable_index"])] = (
                _pauli_expectation(distribution, observable["pauli_mapping"])
            )
    if not np.all(np.isfinite(feature_matrix)) or np.max(np.abs(feature_matrix)) > 1.0 + 1e-9:
        raise PilotError("Hardware feature matrix is non-finite or outside [-1, 1]")

    base_metadata: dict[int, Mapping[str, Any]] = {}
    for row in manifest:
        base_metadata.setdefault(int(row["base_circuit_index"]), row)
    feature_rows = [
        {
            "base_circuit_index": int(index),
            "split": str(base_metadata[index]["split"]),
            "sample_position": int(base_metadata[index]["sample_position"]),
            "source_row_index": int(base_metadata[index]["source_row_index"]),
            "label_for_matched_analysis": float(
                base_metadata[index]["label_for_local_matched_analysis_only"]
            ),
            "features": [float(value) for value in feature_matrix[index]],
        }
        for index in range(PILOT_BASE_CIRCUITS)
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


def retrieve_pilot(args: argparse.Namespace) -> dict[str, Any]:
    if args.result.exists() and not args.force:
        raise PilotError(f"Refusing to overwrite result artifact: {args.result}")
    receipt = safe_base._load_json(args.receipt)
    if receipt.get("kind") != "pbmc68k_q40_sqrtq_b4_fireopal_submission_receipt":
        raise PilotError("Unexpected submission receipt kind")
    action_id = receipt.get("action_id")
    if not action_id or not str(action_id).isnumeric():
        raise PilotError("Submission receipt has no numeric action ID")
    _, _, manifest, _, bundle = _verify_plan_bundle(args.plan, args.bundle)
    if str(receipt.get("bundle_sha256")) != bundle["sha256"]:
        raise PilotError("Submission receipt bundle differs from the pin")
    fireopal, qctrl_source = safe_base._authenticated_fireopal_for_retrieval(
        args.qctrl_notebook
    )
    raw = q40_validate._safe_provider_call(
        "Fire Opal q40 sqrt(q) B=4 result retrieval",
        fireopal.get_result,
        str(action_id),
    )
    if not isinstance(raw, Mapping):
        raise PilotError("Fire Opal result payload is not a mapping")
    validation = validate_hardware_result(raw, manifest)
    artifact = {
        "schema_version": SCHEMA_VERSION,
        "kind": "pbmc68k_q40_sqrtq_b4_fireopal_hardware_result",
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
            "This is a structurally validated Fire Opal hardware result. The frozen "
            "training-only classifier analysis and matched classical comparison remain "
            "separate; this artifact alone is not evidence of quantum advantage."
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
        print("PBMC68k q40 sqrt(q) B=4 Fire Opal pilot planned locally")
        print(f"- circuits: {plan['pilot']['measured_circuits']}")
        print(f"- total requested shots: {plan['pilot']['total_requested_shots']}")
        print("- provider calls: 0")
        print(f"- plan: {args.plan}")
        return 0
    if args.command == "submit":
        receipt = submit_pilot(args)
        print("Fire Opal q40 sqrt(q) B=4 pilot submitted; no result wait performed")
        print(f"- action ID: {receipt['action_id']}")
        print(f"- receipt: {args.receipt}")
        return 0
    if args.command == "retrieve":
        artifact = retrieve_pilot(args)
        print("Fire Opal q40 sqrt(q) B=4 result retrieved without resubmission")
        print(f"- action ID: {artifact['action_id']}")
        print(
            "- feature values: "
            f"{artifact['observable_validation']['feature_value_count']}"
        )
        print(f"- result: {args.result}")
        return 0
    raise PilotError(f"Unknown command: {args.command}")


if __name__ == "__main__":
    raise SystemExit(main())
