#!/usr/bin/env python3
"""Safe PBMC68k q40 Fire Opal seed-11 hardware pilot.

``prepare`` is local-only. ``submit`` is the sole mode that can call
``fireopal.execute`` and requires an exact confirmation literal. Retrieval and
analysis are separate, so a retrieval failure can never resubmit hardware work.
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
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

import qiskit_qos_hash_streaming_genomics_runner as genomics_runner
import qiskit_qos_pbmc68k_q40_fireopal_validate as validate_runner
import qiskit_qos_toy_model as toy


SCHEMA_VERSION = "1.0"
PILOT_SEED = 11
PILOT_QUBITS = 40
PILOT_CIRCUITS = 66
PILOT_BASE_CIRCUITS = 33
PILOT_FEATURE_MAPPINGS = 2
PILOT_SHOTS = 128
PILOT_BACKEND = "ibm_fez"
BOOTSTRAP_SEED = 20260718
BOOTSTRAP_SAMPLES = 10_000
SUBMIT_CONFIRMATION = "FIREOPAL_SUBMIT_SEED11_66X128"

ARTIFACT_DIR = Path("fire_opal_pbmc68k_q40")
DEFAULT_BUNDLE = ARTIFACT_DIR / "pbmc68k_q40_seed11_fireopal_pilot_qasm2.json.gz"
DEFAULT_PLAN = ARTIFACT_DIR / "pbmc68k_q40_seed11_fireopal_pilot_plan.json"
DEFAULT_INTENT = ARTIFACT_DIR / "pbmc68k_q40_seed11_fireopal_submission_intent.json"
DEFAULT_RECEIPT = ARTIFACT_DIR / "pbmc68k_q40_seed11_fireopal_submission_receipt.json"
DEFAULT_RESULT = ARTIFACT_DIR / "pbmc68k_q40_seed11_fireopal_result.json"
DEFAULT_ANALYSIS = ARTIFACT_DIR / "pbmc68k_q40_seed11_fireopal_analysis.json"
DEFAULT_VALIDATED_BUNDLE = (
    ARTIFACT_DIR / "pbmc68k_q40_seed11_seed13_qasm2_provider_validate.json.gz"
)
DEFAULT_IBM_BASELINE = Path(
    "qiskit_qos_pbmc68k_pairwise_quantum_q40_hw_16x16_"
    "ibm_fez_legacylayout_seed11.json"
)


class PilotError(RuntimeError):
    """A sanitized error that is safe to display or persist."""


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
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise PilotError(f"Expected a JSON object in {path}")
    return dict(payload)


def _aggregate_qasm_hash(qasm_hashes: Sequence[str]) -> str:
    return _sha256_bytes(
        json.dumps(list(qasm_hashes), separators=(",", ":")).encode("utf-8")
    )


def _load_bundle(
    path: Path,
) -> tuple[list[str], list[dict[str, Any]], dict[str, Any]]:
    if not path.is_file():
        raise PilotError(f"QASM bundle is missing: {path}")
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        payload = json.load(handle)
    circuits = payload.get("circuits") if isinstance(payload, Mapping) else None
    if not isinstance(circuits, list) or len(circuits) != PILOT_CIRCUITS:
        raise PilotError(f"Pilot bundle must contain exactly {PILOT_CIRCUITS} circuits")

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
        metrics = row.get("metrics")
        if not isinstance(metrics, Mapping) or int(metrics.get("num_qubits", -1)) != PILOT_QUBITS:
            raise PilotError("Pilot bundle contains a non-40q circuit")
        actual_hash = _sha256_bytes(qasm.encode("utf-8"))
        if actual_hash != str(row.get("qasm_sha256", "")):
            raise PilotError("Pilot QASM content does not match its manifest hash")
        qasms.append(qasm)
        manifest.append(row)

    qasm_hashes = [str(row["qasm_sha256"]) for row in manifest]
    info = {
        "path": str(path.resolve()),
        "bytes": path.stat().st_size,
        "sha256": _sha256_file(path),
        "circuits": len(qasms),
        "aggregate_qasm_sha256": _aggregate_qasm_hash(qasm_hashes),
    }
    return qasms, manifest, info


def _validated_seed11_reference(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise PilotError(f"Validated 132-circuit bundle is missing: {path}")
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        payload = json.load(handle)
    circuits = payload.get("circuits") if isinstance(payload, Mapping) else None
    if not isinstance(circuits, list):
        raise PilotError("Validated reference bundle has no circuit list")
    seed_rows = [row for row in circuits if int(row.get("seed", -1)) == PILOT_SEED]
    if len(seed_rows) != PILOT_CIRCUITS:
        raise PilotError("Validated reference does not contain 66 seed-11 circuits")
    return {
        "path": str(path.resolve()),
        "sha256": _sha256_file(path),
        "full_circuit_count": len(circuits),
        "seed11_circuit_count": len(seed_rows),
        "seed11_qasm_hashes": [str(row.get("qasm_sha256", "")) for row in seed_rows],
    }


def _load_ibm_reference(path: Path) -> dict[str, Any]:
    payload = _load_json(path)
    config = payload.get("config")
    runs = payload.get("runs")
    if not isinstance(config, Mapping) or not isinstance(runs, list) or len(runs) != 1:
        raise PilotError("IBM reference artifact has an unexpected structure")
    expected = {
        "seed": PILOT_SEED,
        "hash_seed": PILOT_SEED,
        "max_train_samples": 16,
        "max_test_samples": 16,
        "readout_shots": PILOT_SHOTS,
        "quantum_head": "ridge",
        "readout_family": "local",
        "backend_name": PILOT_BACKEND,
    }
    if any(config.get(key) != value for key, value in expected.items()):
        raise PilotError("IBM reference configuration does not match the pilot")
    if config.get("qubits") != [PILOT_QUBITS]:
        raise PilotError("IBM reference qubit count does not match the pilot")
    run = runs[0]
    if not isinstance(run, Mapping):
        raise PilotError("IBM reference run must be a mapping")
    return {
        "path": str(path.resolve()),
        "sha256": _sha256_file(path),
        "relationship": "historical_configuration_matched_not_time_matched",
        "test_balanced_accuracy_quantum": float(run["test_balanced_accuracy_quantum"]),
        "test_balanced_accuracy_classical_hashed_ridge": float(
            run["test_balanced_accuracy_classical_hashed_ridge"]
        ),
        "test_balanced_accuracy_classical_hashed_linearsvc": float(
            run["test_balanced_accuracy_classical_hashed_linearsvc"]
        ),
        "quantum_head_method": str(run["quantum_head_method"]),
        "readout_feature_count": int(run["readout_feature_count"]),
    }


def _prepare_namespace(args: argparse.Namespace) -> argparse.Namespace:
    return argparse.Namespace(
        cache_dir=args.cache_dir,
        positive_label="CD4+/CD25 T Reg",
        negative_label="CD4+/CD45RO+ Memory",
        qubits=PILOT_QUBITS,
        seeds=(PILOT_SEED,),
        train_fraction=0.67,
        max_train_samples=16,
        max_test_samples=16,
        max_active_genes=256,
        value_mode="log-product",
        feature_mapping_limit=PILOT_FEATURE_MAPPINGS,
        readout_shots=PILOT_SHOTS,
        seed_transpiler=validate_runner.DEFAULT_SEED_TRANSPILER,
        backend=PILOT_BACKEND,
    )


def prepare_pilot(args: argparse.Namespace) -> dict[str, Any]:
    """Create a local plan whose QASM hashes match the validated 132-batch."""
    for path in (args.bundle, args.plan):
        if path.exists() and not args.force:
            raise PilotError(f"Refusing to overwrite existing artifact: {path}")
    started = time.perf_counter()
    qasms, prepared = validate_runner.prepare_batch(_prepare_namespace(args))
    if len(qasms) != PILOT_CIRCUITS:
        raise PilotError("Seed-11 preparation did not produce 66 circuits")
    local = prepared["local_validation"]
    required_flags = (
        "within_batch_limit",
        "all_target_qubits",
        "all_one_classical_register",
        "all_virtual_qubits_only",
        "all_parameters_numeric",
        "all_round_trips_validated",
    )
    if not all(local.get(name) is True for name in required_flags):
        raise PilotError("Seed-11 local QASM validation did not pass every gate")

    validate_runner.write_qasm_bundle(args.bundle, qasms, prepared)
    _, manifest, bundle = _load_bundle(args.bundle)
    reference = _validated_seed11_reference(args.validated_bundle)
    qasm_hashes = [str(row["qasm_sha256"]) for row in manifest]
    if qasm_hashes != reference["seed11_qasm_hashes"]:
        raise PilotError("Pilot differs from the provider-validated seed-11 subset")
    baseline = _load_ibm_reference(args.ibm_reference)
    plan = {
        "schema_version": SCHEMA_VERSION,
        "kind": "pbmc68k_q40_fireopal_seed11_hardware_pilot_plan",
        "status": "ready_for_separate_explicit_submission_authorization",
        "captured_at_utc": _utc_now(),
        "environment": validate_runner.runtime_environment(),
        "authorized_scope": "build_and_local_prepare_only",
        "submission_attempted": False,
        "execution_attempted": False,
        "quantum_seconds_used": 0,
        "pilot": {
            "dataset": "PBMC68k",
            "seed": PILOT_SEED,
            "qubits": PILOT_QUBITS,
            "backend": PILOT_BACKEND,
            "logical_base_circuits": PILOT_BASE_CIRCUITS,
            "feature_mappings": PILOT_FEATURE_MAPPINGS,
            "measured_circuits": PILOT_CIRCUITS,
            "shots_per_circuit": PILOT_SHOTS,
            "total_requested_shots": PILOT_CIRCUITS * PILOT_SHOTS,
            "quantum_head": "ridge",
            "train_samples": 16,
            "test_samples": 16,
        },
        "local_validation": local,
        "qasm_bundle": bundle,
        "validated_132_bundle_reference": {
            key: value for key, value in reference.items() if key != "seed11_qasm_hashes"
        },
        "validated_subset_hash_match": True,
        "qasm_hashes": qasm_hashes,
        "source": prepared["source"],
        "seed_metadata": prepared["seeds"][0],
        "historical_ibm_reference": baseline,
        "submission_boundary": {
            "confirmation_required": True,
            "confirmation_literal": SUBMIT_CONFIRMATION,
            "submit_writes_intent_lock_before_execute": True,
            "automatic_resubmission": False,
            "result_wait_during_submit": False,
            "allowed_submit_api_calls": [
                "fireopal.show_supported_devices",
                "fireopal.execute",
            ],
        },
        "analysis": {
            "bootstrap_seed": BOOTSTRAP_SEED,
            "bootstrap_samples": BOOTSTRAP_SAMPLES,
            "probability_normalization_checked": True,
            "observable_bounds_checked": True,
            "bit_order": "Qiskit little-endian qubit indexing",
        },
        "elapsed_seconds": time.perf_counter() - started,
        "claim_boundary": (
            "This prepares a hardware pilot but contains no hardware result. "
            "The historical IBM reference is configuration-matched, not time-matched, "
            "so it cannot establish a Fire Opal improvement."
        ),
    }
    _atomic_write_json(args.plan, plan)
    return plan


def _verify_plan_bundle(
    plan_path: Path, bundle_path: Path
) -> tuple[dict[str, Any], list[str], list[dict[str, Any]], dict[str, Any]]:
    plan = _load_json(plan_path)
    qasms, manifest, bundle = _load_bundle(bundle_path)
    if plan.get("kind") != "pbmc68k_q40_fireopal_seed11_hardware_pilot_plan":
        raise PilotError("Unexpected pilot plan kind")
    if plan.get("validated_subset_hash_match") is not True:
        raise PilotError("Pilot plan lacks a validated subset hash match")
    planned_bundle = plan.get("qasm_bundle")
    if not isinstance(planned_bundle, Mapping):
        raise PilotError("Pilot plan has no QASM bundle metadata")
    if str(planned_bundle.get("sha256")) != bundle["sha256"]:
        raise PilotError("Pilot bundle file hash differs from the plan")
    planned_hashes = [str(value) for value in plan.get("qasm_hashes", [])]
    actual_hashes = [str(row["qasm_sha256"]) for row in manifest]
    if planned_hashes != actual_hashes:
        raise PilotError("Pilot QASM manifest differs from the plan")
    return plan, qasms, manifest, bundle


def submit_pilot(args: argparse.Namespace) -> dict[str, Any]:
    """Submit once, persist the action ID, and never wait for a result."""
    if args.confirm_submit != SUBMIT_CONFIRMATION:
        raise PilotError(
            f"Submission requires --confirm-submit {SUBMIT_CONFIRMATION}"
        )
    if args.intent.exists() or args.receipt.exists():
        raise PilotError(
            "Submission intent or receipt exists; refusing possible resubmission"
        )
    _, qasms, _, bundle = _verify_plan_bundle(args.plan, args.bundle)
    intent_id = str(uuid.uuid4())
    intent = {
        "schema_version": SCHEMA_VERSION,
        "kind": "pbmc68k_q40_fireopal_seed11_submission_intent",
        "intent_id": intent_id,
        "created_at_utc": _utc_now(),
        "status": "preflight_locked",
        "plan_path": str(args.plan.resolve()),
        "plan_sha256": _sha256_file(args.plan),
        "bundle_path": str(args.bundle.resolve()),
        "bundle_sha256": bundle["sha256"],
        "backend": PILOT_BACKEND,
        "circuit_count": PILOT_CIRCUITS,
        "shots_per_circuit": PILOT_SHOTS,
        "execution_attempted": False,
        "automatic_resubmission": False,
    }
    _atomic_write_json(args.intent, intent)

    try:
        fireopal, credentials, credential_source, qctrl_source = (
            validate_runner._fire_opal_credentials_from_source(
                args.qiskit_account, args.qctrl_notebook, args.instance
            )
        )
        devices = validate_runner._workflow_result(
            validate_runner._safe_provider_call(
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
        job = validate_runner._safe_provider_call(
            "Fire Opal seed-11 hardware submission",
            fireopal.execute,
            circuits=list(qasms),
            shot_count=PILOT_SHOTS,
            credentials=credentials,
            backend_name=PILOT_BACKEND,
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
            raise PilotError(
                "Execute was accepted but no action ID returned; do not resubmit"
            )

        receipt = {
            "schema_version": SCHEMA_VERSION,
            "kind": "pbmc68k_q40_fireopal_seed11_submission_receipt",
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
            "credential_source": credential_source,
            "qctrl_auth_source": qctrl_source,
            "api_calls": ["fireopal.show_supported_devices", "fireopal.execute"],
            "execution_attempted": True,
            "result_waited_during_submit": False,
            "result_retrieved": False,
            "quantum_seconds_used": None,
            "automatic_resubmission": False,
            "claim_boundary": "Submission receipt only; no result retrieved.",
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
    except (validate_runner.RunnerError, PilotError) as exc:
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
    if isinstance(value, np.generic):
        return value.item()
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return f"<{type(value).__name__}>"


def _authenticated_fireopal_for_retrieval(
    qctrl_notebook: Path | None,
) -> tuple[Any, str]:
    key, source = validate_runner._qctrl_api_key(qctrl_notebook)
    import fireopal

    validate_runner._safe_provider_call(
        "Q-CTRL authentication", fireopal.authenticate_qctrl_account, api_key=key
    )
    return fireopal, source


def retrieve_pilot(args: argparse.Namespace) -> dict[str, Any]:
    """Retrieve an existing action ID; this path contains no execute call."""
    if args.result.exists() and not args.force:
        raise PilotError(f"Refusing to overwrite result artifact: {args.result}")
    receipt = _load_json(args.receipt)
    if receipt.get("kind") != "pbmc68k_q40_fireopal_seed11_submission_receipt":
        raise PilotError("Unexpected submission receipt kind")
    action_id = receipt.get("action_id")
    if not action_id:
        raise PilotError("Submission receipt has no action ID")

    fireopal, qctrl_source = _authenticated_fireopal_for_retrieval(
        args.qctrl_notebook
    )
    raw = validate_runner._safe_provider_call(
        "Fire Opal seed-11 result retrieval", fireopal.get_result, action_id
    )
    if not isinstance(raw, Mapping):
        raise PilotError("Fire Opal result payload is not a mapping")
    artifact = {
        "schema_version": SCHEMA_VERSION,
        "kind": "pbmc68k_q40_fireopal_seed11_hardware_result",
        "status": "retrieved_unanalyzed",
        "retrieved_at_utc": _utc_now(),
        "action_id": str(action_id),
        "receipt_path": str(args.receipt.resolve()),
        "receipt_sha256_before_update": _sha256_file(args.receipt),
        "qctrl_auth_source": qctrl_source,
        "api_calls": ["fireopal.get_result"],
        "submission_attempted_in_this_mode": False,
        "automatic_resubmission": False,
        "raw_result": _json_safe_redacted(raw),
        "claim_boundary": "Retrieved provider data; analysis is separate.",
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


def _result_distributions(
    result_artifact: Mapping[str, Any],
) -> list[Mapping[str, Any]]:
    raw = result_artifact.get("raw_result")
    if not isinstance(raw, Mapping):
        raise PilotError("Result artifact has no raw result mapping")
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
        value = float(raw_value)
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
        raise PilotError(
            "Result is neither normalized probability nor exact shot counts"
        )
    return cleaned, {
        "semantics": semantics,
        "total_weight": total,
        "normalization_deviation": deviation,
        "outcomes": len(cleaned),
    }


def _stratified_bootstrap(
    labels: np.ndarray, predictions: np.ndarray
) -> dict[str, Any]:
    labels = np.asarray(labels, dtype=np.float64)
    predictions = np.asarray(predictions, dtype=bool)
    positive = np.flatnonzero(labels > 0.0)
    negative = np.flatnonzero(labels < 0.0)
    if len(positive) == 0 or len(negative) == 0:
        raise PilotError("Bootstrap requires both test classes")
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    values = np.empty(BOOTSTRAP_SAMPLES, dtype=np.float64)
    for index in range(BOOTSTRAP_SAMPLES):
        sampled = np.concatenate(
            [
                rng.choice(positive, size=len(positive), replace=True),
                rng.choice(negative, size=len(negative), replace=True),
            ]
        )
        metrics = genomics_runner.binary_prediction_metrics(
            labels[sampled], predictions[sampled]
        )
        values[index] = float(metrics["balanced_accuracy"])
    return {
        "method": "stratified_nonparametric_bootstrap_over_test_cells",
        "seed": BOOTSTRAP_SEED,
        "replicates": BOOTSTRAP_SAMPLES,
        "standard_error": float(np.std(values, ddof=1)),
        "ci95_percentile": [
            float(np.percentile(values, 2.5)),
            float(np.percentile(values, 97.5)),
        ],
        "does_not_include_shot_or_hardware_drift_uncertainty": True,
    }


def analyze_distributions(
    distributions: Sequence[Mapping[str, Any]],
    manifest: Sequence[Mapping[str, Any]],
    *,
    num_qubits: int = PILOT_QUBITS,
    shots: int = PILOT_SHOTS,
) -> dict[str, Any]:
    """Turn Fire Opal probability mappings into the existing ridge-head metrics."""
    if len(distributions) != len(manifest):
        raise PilotError("Result and manifest circuit counts differ")
    feature_rows: dict[int, dict[int, float]] = {}
    base_metadata: dict[int, dict[str, Any]] = {}
    checks: list[dict[str, Any]] = []
    expectations: list[float] = []
    for distribution, raw_row in zip(distributions, manifest, strict=True):
        row = dict(raw_row)
        cleaned, check = _validated_distribution(
            distribution, num_qubits=num_qubits, shots=shots
        )
        mapping_rows = row.get("pauli_mapping")
        if not isinstance(mapping_rows, list):
            raise PilotError("Manifest row lacks a Pauli mapping")
        mapping = {
            int(item["qubit"]): str(item["pauli"])
            for item in mapping_rows
            if isinstance(item, Mapping)
        }
        expectation = toy.expectation_from_counts(
            cleaned, mapping=mapping, num_qubits=num_qubits
        )
        if not math.isfinite(expectation) or abs(expectation) > 1.0 + 1e-9:
            raise PilotError("Pauli expectation is outside [-1, 1]")
        base_index = int(row["base_circuit_index"])
        feature_index = int(row["feature_index"])
        if feature_index in feature_rows.setdefault(base_index, {}):
            raise PilotError("Duplicate base/feature result row")
        feature_rows[base_index][feature_index] = expectation
        base_metadata.setdefault(
            base_index,
            {
                "role": row.get("role"),
                "split": row.get("split"),
                "sample_position": row.get("sample_position"),
                "source_row_index": row.get("source_row_index"),
                "label": row.get("label"),
            },
        )
        checks.append(check)
        expectations.append(expectation)

    base_indices = sorted(feature_rows)
    expected_features = set(range(PILOT_FEATURE_MAPPINGS))
    if any(set(feature_rows[index]) != expected_features for index in base_indices):
        raise PilotError("A logical circuit is missing a feature result")
    features = np.asarray(
        [
            [feature_rows[index][feature] for feature in range(PILOT_FEATURE_MAPPINGS)]
            for index in base_indices
        ],
        dtype=np.float64,
    )
    if features.shape[0] != PILOT_BASE_CIRCUITS:
        raise PilotError("Pilot analysis requires 33 logical circuits")
    if base_metadata[base_indices[0]]["role"] != "weighted_training_sketch":
        raise PilotError("First logical circuit is not the training sketch")

    train_indices = sorted(
        index
        for index in base_indices
        if base_metadata[index]["role"] == "query"
        and base_metadata[index]["split"] == "train"
    )
    test_indices = sorted(
        index
        for index in base_indices
        if base_metadata[index]["role"] == "query"
        and base_metadata[index]["split"] == "test"
    )
    if len(train_indices) != 16 or len(test_indices) != 16:
        raise PilotError("Pilot analysis requires a 16/16 split")
    feature_by_base = {
        index: features[position] for position, index in enumerate(base_indices)
    }
    model_features = feature_by_base[base_indices[0]].copy()
    query_train = np.asarray([feature_by_base[index] for index in train_indices])
    query_test = np.asarray([feature_by_base[index] for index in test_indices])
    y_train = np.asarray([float(base_metadata[index]["label"]) for index in train_indices])
    y_test = np.asarray([float(base_metadata[index]["label"]) for index in test_indices])

    head_train_raw = np.asarray(
        [toy.quantum_head_feature_vector(model_features, row) for row in query_train],
        dtype=np.float64,
    )
    head_test_raw = np.asarray(
        [toy.quantum_head_feature_vector(model_features, row) for row in query_test],
        dtype=np.float64,
    )
    head_train, head_test = toy.standardize(head_train_raw, head_test_raw)
    head_weights = toy.ridge_linear_classifier(head_train, y_train)
    train_scores = np.asarray(head_train @ head_weights, dtype=np.float64)
    test_scores = np.asarray(head_test @ head_weights, dtype=np.float64)
    orientation_flipped = False
    if toy.pearson_corr(train_scores, y_train) < 0.0:
        train_scores *= -1.0
        test_scores *= -1.0
        model_features *= -1.0
        orientation_flipped = True
    threshold = 0.5 * (
        float(np.mean(train_scores[y_train > 0.0]))
        + float(np.mean(train_scores[y_train < 0.0]))
    )
    train_predictions = train_scores >= threshold
    test_predictions = test_scores >= threshold
    train_metrics = genomics_runner.binary_prediction_metrics(y_train, train_predictions)
    test_metrics = genomics_runner.binary_prediction_metrics(y_test, test_predictions)

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
        },
        "observable_validation": {
            "passed": True,
            "expectation_count": len(expectations),
            "minimum": float(min(expectations)),
            "maximum": float(max(expectations)),
            "all_within_minus_one_plus_one": True,
            "bit_order": "Qiskit little-endian qubit indexing",
        },
        "model": {
            "quantum_head": "ridge",
            "feature_mapping_count": PILOT_FEATURE_MAPPINGS,
            "orientation_flipped": orientation_flipped,
            "threshold": float(threshold),
            "model_features": model_features.tolist(),
            "head_weights": np.asarray(head_weights, dtype=np.float64).tolist(),
            "train_scores": train_scores.tolist(),
            "test_scores": test_scores.tolist(),
            "train_labels": y_train.tolist(),
            "test_labels": y_test.tolist(),
            "train_metrics": train_metrics,
            "test_metrics": test_metrics,
            "test_balanced_accuracy_uncertainty": _stratified_bootstrap(
                y_test, test_predictions
            ),
        },
    }


def analyze_pilot(args: argparse.Namespace) -> dict[str, Any]:
    if args.analysis.exists() and not args.force:
        raise PilotError(f"Refusing to overwrite analysis: {args.analysis}")
    plan, _, manifest, bundle = _verify_plan_bundle(args.plan, args.bundle)
    result = _load_json(args.result)
    if result.get("kind") != "pbmc68k_q40_fireopal_seed11_hardware_result":
        raise PilotError("Unexpected Fire Opal result artifact kind")
    distributions = _result_distributions(result)
    if len(distributions) != PILOT_CIRCUITS:
        raise PilotError("Fire Opal result must contain 66 distributions")
    core = analyze_distributions(distributions, manifest)
    baseline = _load_ibm_reference(args.ibm_reference)
    fireopal_balanced = float(core["model"]["test_metrics"]["balanced_accuracy"])
    analysis = {
        "schema_version": SCHEMA_VERSION,
        "kind": "pbmc68k_q40_fireopal_seed11_hardware_analysis",
        "status": "pass",
        "captured_at_utc": _utc_now(),
        "environment": validate_runner.runtime_environment(),
        "action_id": result.get("action_id"),
        "bundle": bundle,
        "plan_sha256": _sha256_file(args.plan),
        "result_sha256": _sha256_file(args.result),
        "pilot": plan["pilot"],
        **core,
        "historical_ibm_reference": baseline,
        "comparison": {
            "fireopal_test_balanced_accuracy": fireopal_balanced,
            "historical_ibm_test_balanced_accuracy_quantum": baseline[
                "test_balanced_accuracy_quantum"
            ],
            "difference_fireopal_minus_historical_ibm": (
                fireopal_balanced - baseline["test_balanced_accuracy_quantum"]
            ),
            "is_time_matched": False,
            "supports_fireopal_improvement_claim": False,
        },
        "claim_boundary": (
            "Seed-11 pilot only. The bootstrap covers test-cell resampling, not "
            "shot noise or hardware drift. The IBM reference is historical and not "
            "time-matched, so no causal Fire Opal improvement is established."
        ),
    }
    _atomic_write_json(args.analysis, analysis)
    return analysis


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare = subparsers.add_parser("prepare", help="local-only preparation")
    prepare.add_argument("--cache-dir", type=Path, default=Path("data_cache/pbmc68k"))
    prepare.add_argument("--bundle", type=Path, default=DEFAULT_BUNDLE)
    prepare.add_argument("--plan", type=Path, default=DEFAULT_PLAN)
    prepare.add_argument(
        "--validated-bundle", type=Path, default=DEFAULT_VALIDATED_BUNDLE
    )
    prepare.add_argument("--ibm-reference", type=Path, default=DEFAULT_IBM_BASELINE)
    prepare.add_argument("--force", action="store_true")

    submit = subparsers.add_parser(
        "submit", help="submit once and return without waiting"
    )
    submit.add_argument("--bundle", type=Path, default=DEFAULT_BUNDLE)
    submit.add_argument("--plan", type=Path, default=DEFAULT_PLAN)
    submit.add_argument("--intent", type=Path, default=DEFAULT_INTENT)
    submit.add_argument("--receipt", type=Path, default=DEFAULT_RECEIPT)
    submit.add_argument("--qiskit-account", default="default-ibm-cloud")
    submit.add_argument("--qctrl-notebook", type=Path)
    submit.add_argument("--instance")
    submit.add_argument("--confirm-submit", default="")

    retrieve = subparsers.add_parser(
        "retrieve", help="retrieve by action ID; never submit"
    )
    retrieve.add_argument("--receipt", type=Path, default=DEFAULT_RECEIPT)
    retrieve.add_argument("--result", type=Path, default=DEFAULT_RESULT)
    retrieve.add_argument("--qctrl-notebook", type=Path)
    retrieve.add_argument("--force", action="store_true")

    analyze = subparsers.add_parser("analyze", help="offline result analysis")
    analyze.add_argument("--bundle", type=Path, default=DEFAULT_BUNDLE)
    analyze.add_argument("--plan", type=Path, default=DEFAULT_PLAN)
    analyze.add_argument("--result", type=Path, default=DEFAULT_RESULT)
    analyze.add_argument("--analysis", type=Path, default=DEFAULT_ANALYSIS)
    analyze.add_argument("--ibm-reference", type=Path, default=DEFAULT_IBM_BASELINE)
    analyze.add_argument("--force", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "prepare":
        plan = prepare_pilot(args)
        print("PBMC68k q40 Fire Opal seed-11 pilot prepared locally")
        print(f"- circuits: {plan['pilot']['measured_circuits']}")
        print(f"- shots if authorized later: {plan['pilot']['total_requested_shots']}")
        print("- provider calls: 0")
        print("- hardware submission attempted: False")
        print(f"- plan: {args.plan}")
        return 0
    if args.command == "submit":
        receipt = submit_pilot(args)
        print("Fire Opal seed-11 pilot submitted; no result wait performed")
        print(f"- action ID: {receipt['action_id']}")
        print(f"- receipt: {args.receipt}")
        return 0
    if args.command == "retrieve":
        artifact = retrieve_pilot(args)
        print("Fire Opal seed-11 result retrieved without resubmission")
        print(f"- action ID: {artifact['action_id']}")
        print(f"- result: {args.result}")
        return 0
    if args.command == "analyze":
        analysis = analyze_pilot(args)
        metric = analysis["model"]["test_metrics"]["balanced_accuracy"]
        print("Fire Opal seed-11 pilot analyzed offline")
        print(f"- test balanced accuracy: {metric:.6f}")
        print(f"- analysis: {args.analysis}")
        return 0
    raise PilotError(f"Unsupported command: {args.command}")


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except PilotError as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(2) from None
