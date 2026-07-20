#!/usr/bin/env python3
"""Prepare, validate, submit, and retrieve the official flat-QOS kernel pilot.

The frozen pilot contains 64 random official flat-state sampling-kernel
instances plus identity and linear-phase controls.  A final Hadamard transform
makes the sampled phase sketch observable in the computational basis.  Provider
submission is isolated behind a literal confirmation and a persisted plan.
"""

from __future__ import annotations

import argparse
import gzip
import json
import os
import sys
import time
import uuid
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
from qiskit import qasm2
from qiskit.quantum_info import Statevector

import qiskit_official_qos_sampling_port as qos_port
import qiskit_qos_pbmc68k_q40_fireopal_validate as q40_validate
import qiskit_qos_pbmc68k_q60_rx05_fireopal_pilot as safe_base


SCHEMA_VERSION = "1.0"
PILOT_KIND = "official_qos_flat_interference_fireopal_pilot"
DIMENSION = 16
QUBITS = 4
SAMPLES = 64
RANDOM_INSTANCES = 64
CONTROLS = 2
CIRCUITS = RANDOM_INSTANCES + CONTROLS
SHOTS = 4096
BACKEND = "ibm_fez"
PILOT_SEED = 6604096
CONTROL_MASK = 0b1011
SUBMIT_CONFIRMATION = "SUBMIT_OFFICIAL_QOS_FLAT_66_4096"
AUTHORIZATION_CONFIRMATION = "KERNEL_PILOT_AUTHORIZED"
DEFAULT_QCTRL_NOTEBOOK = Path(
    "/mnt/c/Users/Lenna/SynologyDrive/stackexchange/"
    "get-started-with-fire-opal-on-ibm-quantum.ipynb"
)
ARTIFACT_DIR = Path("fire_opal_official_qos_kernel")
DEFAULT_BUNDLE = ARTIFACT_DIR / "official_qos_flat_dim16_m64_66_qasm2.json.gz"
DEFAULT_PREPARE = ARTIFACT_DIR / "official_qos_flat_dim16_m64_66_prepare.json"
DEFAULT_VALIDATE = ARTIFACT_DIR / "official_qos_flat_dim16_m64_66_validate.json"
DEFAULT_PLAN = ARTIFACT_DIR / "official_qos_flat_dim16_m64_66_plan.json"
DEFAULT_INTENT = ARTIFACT_DIR / "official_qos_flat_dim16_m64_66_intent.json"
DEFAULT_RECEIPT = ARTIFACT_DIR / "official_qos_flat_dim16_m64_66_receipt.json"
DEFAULT_STATUS = ARTIFACT_DIR / "official_qos_flat_dim16_m64_66_status.json"
DEFAULT_RESULT = ARTIFACT_DIR / "official_qos_flat_dim16_m64_66_result.json"


class PilotError(RuntimeError):
    pass


def _atomic_write_gzip_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    try:
        with gzip.open(temporary, "wt", encoding="utf-8", compresslevel=9) as handle:
            json.dump(payload, handle, sort_keys=True, separators=(",", ":"))
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, Mapping):
        raise PilotError(f"Expected a JSON mapping in {path}")
    return dict(value)


def _distribution_from_state(state: np.ndarray) -> dict[str, float]:
    probabilities = np.abs(np.asarray(state, dtype=np.complex128)) ** 2
    total = float(np.sum(probabilities))
    if not np.isfinite(total) or abs(total - 1.0) > 1e-10:
        raise PilotError("Ideal state probabilities are not normalized")
    return {
        format(index, f"0{QUBITS}b"): float(value)
        for index, value in enumerate(probabilities)
        if value > 1e-16
    }


def _dense_distribution(distribution: Mapping[str, Any]) -> np.ndarray:
    values = np.zeros(DIMENSION, dtype=np.float64)
    for bitstring, value in distribution.items():
        index = int(str(bitstring).replace(" ", ""), 2)
        values[index] += float(value)
    total = float(np.sum(values))
    if total <= 0.0 or not np.all(np.isfinite(values)) or np.any(values < 0.0):
        raise PilotError("Distribution is not finite, non-negative, and nonzero")
    return values / total


def _hellinger_fidelity(first: np.ndarray, second: np.ndarray) -> float:
    value = float(np.sum(np.sqrt(np.asarray(first) * np.asarray(second))) ** 2)
    return float(max(0.0, min(1.0, value)))


def _total_variation(first: np.ndarray, second: np.ndarray) -> float:
    return float(0.5 * np.sum(np.abs(np.asarray(first) - np.asarray(second))))


def _array_hash(value: np.ndarray) -> str:
    array = np.ascontiguousarray(value)
    header = f"{array.dtype.str}|{array.shape}|".encode("utf-8")
    return q40_validate._sha256_bytes(header + array.tobytes())


def _instance_specs() -> list[dict[str, Any]]:
    repeats = SAMPLES // DIMENSION
    balanced_indices = np.repeat(np.arange(DIMENSION, dtype=np.int32), repeats)
    identity_values = np.ones(SAMPLES, dtype=np.float64)
    parity_vector = np.asarray(
        [
            -1.0 if ((index & CONTROL_MASK).bit_count() % 2) else 1.0
            for index in range(DIMENSION)
        ],
        dtype=np.float64,
    )
    specs: list[dict[str, Any]] = [
        {
            "role": "control_identity",
            "instance_seed": None,
            "vector": np.ones(DIMENSION, dtype=np.float64),
            "sampled_indices": balanced_indices,
            "sampled_values": identity_values,
            "expected_target": "0000",
        },
        {
            "role": "control_linear_phase",
            "instance_seed": None,
            "vector": parity_vector,
            "sampled_indices": balanced_indices,
            "sampled_values": np.repeat(parity_vector, repeats),
            "expected_target": format(CONTROL_MASK, f"0{QUBITS}b"),
        },
    ]
    for instance in range(RANDOM_INSTANCES):
        seed = PILOT_SEED + instance
        rng = np.random.default_rng(seed)
        vector = qos_port.random_flat_vector(DIMENSION, rng)
        sampled_indices, sampled_values = qos_port.sample_from_vector(
            vector, SAMPLES, rng
        )
        specs.append(
            {
                "role": "random_flat_kernel",
                "instance_seed": seed,
                "vector": vector,
                "sampled_indices": sampled_indices,
                "sampled_values": sampled_values.astype(np.float64),
                "expected_target": None,
            }
        )
    if len(specs) != CIRCUITS:
        raise PilotError("Frozen pilot does not contain 66 circuits")
    return specs


def _exact_rows() -> tuple[list[str], list[dict[str, Any]], dict[str, Any]]:
    qasms: list[str] = []
    manifest: list[dict[str, Any]] = []
    jax_errors: list[float] = []
    qasm_errors: list[float] = []
    control_probabilities: list[float] = []
    for circuit_index, spec in enumerate(_instance_specs()):
        sampled_indices = np.asarray(spec["sampled_indices"], dtype=np.int32)
        sampled_values = np.asarray(spec["sampled_values"], dtype=np.float64)
        circuit, phase_diagonal = qos_port.build_flat_interference_circuit_from_samples(
            sampled_indices, sampled_values, DIMENSION
        )
        qiskit_state = np.asarray(Statevector.from_instruction(circuit).data)
        jax_state = qos_port.flat_interference_state_from_jax(
            sampled_indices, sampled_values, DIMENSION
        )
        jax_error = float(np.max(np.abs(qiskit_state - jax_state)))
        jax_errors.append(jax_error)
        ideal = _distribution_from_state(qiskit_state)
        measured = circuit.copy()
        measured.measure_all()
        qasm, qasm_metadata = q40_validate.export_numeric_qasm2(measured)
        parsed = qasm2.loads(qasm)
        parsed.remove_final_measurements(inplace=True)
        parsed_state = np.asarray(Statevector.from_instruction(parsed).data)
        qasm_error = float(
            np.max(np.abs(np.abs(parsed_state) ** 2 - np.abs(qiskit_state) ** 2))
        )
        qasm_errors.append(qasm_error)
        expected_target = spec["expected_target"]
        if expected_target is not None:
            control_probabilities.append(float(ideal.get(str(expected_target), 0.0)))
        row = {
            "circuit_index": circuit_index,
            "role": str(spec["role"]),
            "instance_seed": spec["instance_seed"],
            "dimension": DIMENSION,
            "num_qubits": QUBITS,
            "num_samples": SAMPLES,
            "vector_sha256": _array_hash(np.asarray(spec["vector"])),
            "samples_sha256": _array_hash(
                np.column_stack([sampled_indices, sampled_values])
            ),
            "sampled_indices": [int(value) for value in sampled_indices],
            "sampled_values": [float(value) for value in sampled_values],
            "phase_diagonal_sha256": _array_hash(
                np.asarray(phase_diagonal, dtype=np.complex128)
            ),
            "expected_target": expected_target,
            "ideal_probabilities": ideal,
            "ideal_probability_sha256": _array_hash(_dense_distribution(ideal)),
            "jax_max_abs_amplitude_error": jax_error,
            "qasm_roundtrip_max_abs_probability_error": qasm_error,
            "logical_metrics": q40_validate.circuit_metrics(measured),
            **qasm_metadata,
        }
        qasms.append(qasm)
        manifest.append(row)
    local = {
        "circuit_count": len(qasms),
        "random_instances": RANDOM_INSTANCES,
        "controls": CONTROLS,
        "max_abs_qiskit_minus_official_jax_amplitude": float(max(jax_errors)),
        "max_abs_qasm_roundtrip_probability_error": float(max(qasm_errors)),
        "minimum_control_target_probability": float(min(control_probabilities)),
        "maximum_logical_depth": int(
            max(row["logical_metrics"]["depth"] for row in manifest)
        ),
        "maximum_qasm_depth": int(
            max(row["metrics"]["depth"] for row in manifest)
        ),
        "maximum_qasm_two_qubit_gates": int(
            max(row["metrics"]["two_qubit_gates"] for row in manifest)
        ),
        "all_qasm_numeric_roundtrip_validated": all(
            bool(row["round_trip_validated"]) for row in manifest
        ),
        "all_measurements_identity_mapped": all(
            row["measurement_mapping"]
            == [{"qubit": index, "clbit": index} for index in range(QUBITS)]
            for row in manifest
        ),
        "all_probabilities_normalized": True,
        "passed": (
            len(qasms) == CIRCUITS
            and max(jax_errors) <= 1e-12
            and max(qasm_errors) <= 1e-12
            and min(control_probabilities) >= 1.0 - 1e-12
        ),
    }
    return qasms, manifest, local


def _aggregate_qasm_hash(manifest: Sequence[Mapping[str, Any]]) -> str:
    hashes = [str(row["qasm_sha256"]) for row in manifest]
    return q40_validate._sha256_bytes(
        json.dumps(hashes, separators=(",", ":")).encode("utf-8")
    )


def prepare_pilot(args: argparse.Namespace) -> dict[str, Any]:
    if (args.bundle.exists() or args.report.exists()) and not args.force:
        raise PilotError("Prepare artifact already exists; refusing to overwrite")
    started = time.perf_counter()
    qasms, manifest, local = _exact_rows()
    if not local["passed"]:
        raise PilotError("Local QOS/JAX/QASM verification failed")
    aggregate = _aggregate_qasm_hash(manifest)
    bundle_payload = {
        "schema_version": SCHEMA_VERSION,
        "kind": f"{PILOT_KIND}_qasm_bundle",
        "created_at_utc": q40_validate._utc_now(),
        "config": {
            "dimension": DIMENSION,
            "qubits": QUBITS,
            "samples_per_kernel": SAMPLES,
            "random_instances": RANDOM_INSTANCES,
            "controls": CONTROLS,
            "circuits": CIRCUITS,
            "shots_per_circuit": SHOTS,
            "backend": BACKEND,
            "pilot_seed": PILOT_SEED,
            "control_mask": CONTROL_MASK,
        },
        "aggregate_qasm_sha256": aggregate,
        "circuits": [
            {**row, "qasm": qasm}
            for row, qasm in zip(manifest, qasms, strict=True)
        ],
    }
    _atomic_write_gzip_json(args.bundle, bundle_payload)
    bundle_info = {
        "path": str(args.bundle.resolve()),
        "sha256": q40_validate._sha256_file(args.bundle),
        "bytes": args.bundle.stat().st_size,
        "circuits": CIRCUITS,
        "aggregate_qasm_sha256": aggregate,
        "gzip_json_round_trip_passed": True,
    }
    reloaded_qasms, reloaded_manifest, reloaded = load_bundle(args.bundle)
    if (
        reloaded["sha256"] != bundle_info["sha256"]
        or reloaded["aggregate_qasm_sha256"] != aggregate
        or reloaded_qasms != qasms
        or len(reloaded_manifest) != CIRCUITS
    ):
        raise PilotError("Persisted QASM bundle failed independent reload")
    report = {
        "schema_version": SCHEMA_VERSION,
        "kind": f"{PILOT_KIND}_prepare",
        "status": "complete_local_only",
        "captured_at_utc": q40_validate._utc_now(),
        "provider_calls": [],
        "execution_attempted": False,
        "quantum_seconds_used": 0,
        "protocol": {
            "kernel": "official q_state_sketch_flat sampling kernel",
            "readout": "final normalized Walsh-Hadamard transform then Z measurement",
            "phase_formula": "phi[j] = pi * dim / M * sum_m 1[i_m=j](1-v_m)/2",
            "dimension": DIMENSION,
            "qubits": QUBITS,
            "samples": SAMPLES,
            "random_instances": RANDOM_INSTANCES,
            "controls": CONTROLS,
            "circuits": CIRCUITS,
            "shots_per_circuit": SHOTS,
            "total_requested_shots": CIRCUITS * SHOTS,
            "backend": BACKEND,
        },
        "local_validation": local,
        "bundle": bundle_info,
        "manifest": manifest,
        "elapsed_seconds": float(time.perf_counter() - started),
        "claim_boundary": (
            "This local artifact verifies a phase-sensitive hardware readout of the "
            "official flat QOS sampling kernel against JAX and exact Qiskit. It is not "
            "yet provider validation or a hardware result."
        ),
    }
    q40_validate._atomic_write_json(args.report, report)
    return report


def load_bundle(
    path: Path,
) -> tuple[list[str], list[dict[str, Any]], dict[str, Any]]:
    if not path.is_file():
        raise PilotError(f"QASM bundle not found: {path}")
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, Mapping) or payload.get("kind") != f"{PILOT_KIND}_qasm_bundle":
        raise PilotError("Unexpected QASM bundle kind")
    rows = payload.get("circuits")
    if not isinstance(rows, list) or len(rows) != CIRCUITS:
        raise PilotError("QASM bundle must contain exactly 66 circuits")
    qasms: list[str] = []
    manifest: list[dict[str, Any]] = []
    for expected_index, raw_row in enumerate(rows):
        if not isinstance(raw_row, Mapping):
            raise PilotError("QASM bundle row is not a mapping")
        row = dict(raw_row)
        qasm = row.pop("qasm", None)
        if int(row.get("circuit_index", -1)) != expected_index:
            raise PilotError("QASM bundle ordering changed")
        if not isinstance(qasm, str) or not qasm.startswith("OPENQASM 2.0;"):
            raise PilotError("QASM bundle contains an invalid payload")
        if q40_validate._sha256_bytes(qasm.encode("utf-8")) != row.get("qasm_sha256"):
            raise PilotError("QASM payload hash differs from manifest")
        parsed = qasm2.loads(qasm)
        if parsed.num_qubits != QUBITS or parsed.num_clbits != QUBITS:
            raise PilotError("QASM register width changed")
        qasms.append(qasm)
        manifest.append(row)
    aggregate = _aggregate_qasm_hash(manifest)
    if aggregate != payload.get("aggregate_qasm_sha256"):
        raise PilotError("Aggregate QASM hash changed")
    return qasms, manifest, {
        "path": str(path.resolve()),
        "sha256": q40_validate._sha256_file(path),
        "bytes": path.stat().st_size,
        "circuits": CIRCUITS,
        "aggregate_qasm_sha256": aggregate,
    }


def _warning_summary(messages: Sequence[str]) -> dict[str, Any]:
    counts = Counter(str(message) for message in messages)
    return {
        "total": len(messages),
        "unique": len(counts),
        "histogram": [
            {"message": message, "count": count}
            for message, count in counts.most_common()
        ],
    }


def validate_pilot(args: argparse.Namespace) -> dict[str, Any]:
    if args.report.exists() and not args.force:
        raise PilotError("Validation report already exists; refusing to overwrite")
    qasms, manifest, bundle = load_bundle(args.bundle)
    provider = q40_validate.validate_fireopal_batch(
        qasms,
        backend=BACKEND,
        qiskit_account=args.qiskit_account,
        qctrl_notebook=args.qctrl_notebook,
        instance=args.instance,
    )
    warnings = [str(value) for value in provider.pop("warnings", [])]
    provider["warning_summary"] = _warning_summary(warnings)
    provider["manifest_order_and_hashes_verified_before_call"] = True
    report = {
        "schema_version": SCHEMA_VERSION,
        "kind": f"{PILOT_KIND}_validate_only",
        "status": "passed_with_warnings" if provider.get("passed") else "failed",
        "captured_at_utc": q40_validate._utc_now(),
        "backend": BACKEND,
        "circuits": len(qasms),
        "shots_intent_per_circuit": SHOTS,
        "bundle": bundle,
        "manifest_qasm_hashes": [str(row["qasm_sha256"]) for row in manifest],
        "provider_validation": provider,
        "execution_attempted": False,
        "quantum_seconds_used": 0,
        "allowed_provider_calls": [
            "fireopal.show_supported_devices",
            "fireopal.validate",
        ],
        "claim_boundary": (
            "Passing Fire Opal validation establishes input compatibility only. "
            "It is not hardware execution or evidence of kernel fidelity."
        ),
    }
    q40_validate._atomic_write_json(args.report, report)
    return report


def plan_pilot(args: argparse.Namespace) -> dict[str, Any]:
    if args.authorization != AUTHORIZATION_CONFIRMATION:
        raise PilotError(
            f"Plan requires --authorization {AUTHORIZATION_CONFIRMATION}"
        )
    if args.plan.exists() and not args.force:
        raise PilotError("Pilot plan already exists; refusing to overwrite")
    validation = _load_json(args.validation)
    if (
        validation.get("kind") != f"{PILOT_KIND}_validate_only"
        or not validation.get("provider_validation", {}).get("passed")
        or validation.get("execution_attempted") is not False
    ):
        raise PilotError("A passing validate-only artifact is required")
    _, manifest, bundle = load_bundle(args.bundle)
    if validation.get("bundle", {}).get("sha256") != bundle["sha256"]:
        raise PilotError("Validation report and QASM bundle differ")
    plan = {
        "schema_version": SCHEMA_VERSION,
        "kind": f"{PILOT_KIND}_plan",
        "status": "authorized_and_preflight_locked",
        "created_at_utc": q40_validate._utc_now(),
        "backend": BACKEND,
        "circuit_count": CIRCUITS,
        "shots_per_circuit": SHOTS,
        "total_requested_shots": CIRCUITS * SHOTS,
        "bundle": bundle,
        "validation_path": str(args.validation.resolve()),
        "validation_sha256": q40_validate._sha256_file(args.validation),
        "aggregate_qasm_sha256": bundle["aggregate_qasm_sha256"],
        "qasm_hashes": [str(row["qasm_sha256"]) for row in manifest],
        "submission_authorized_by_user": True,
        "required_submit_confirmation": SUBMIT_CONFIRMATION,
        "automatic_resubmission": False,
        "result_waited_during_submit": False,
        "claim_boundary": "Dry-run hardware plan only; no execute call in this mode.",
    }
    q40_validate._atomic_write_json(args.plan, plan)
    return plan


def _verify_plan(
    plan_path: Path, bundle_path: Path
) -> tuple[dict[str, Any], list[str], list[dict[str, Any]], dict[str, Any]]:
    plan = _load_json(plan_path)
    if (
        plan.get("kind") != f"{PILOT_KIND}_plan"
        or plan.get("status") != "authorized_and_preflight_locked"
        or plan.get("submission_authorized_by_user") is not True
        or plan.get("backend") != BACKEND
        or int(plan.get("circuit_count", -1)) != CIRCUITS
        or int(plan.get("shots_per_circuit", -1)) != SHOTS
    ):
        raise PilotError("Pilot plan is not the frozen authorized protocol")
    qasms, manifest, bundle = load_bundle(bundle_path)
    if (
        plan.get("bundle", {}).get("sha256") != bundle["sha256"]
        or plan.get("aggregate_qasm_sha256") != bundle["aggregate_qasm_sha256"]
        or plan.get("qasm_hashes")
        != [str(row["qasm_sha256"]) for row in manifest]
    ):
        raise PilotError("Pilot plan and QASM bundle differ")
    return plan, qasms, manifest, bundle


def submit_pilot(args: argparse.Namespace) -> dict[str, Any]:
    if args.confirm_submit != SUBMIT_CONFIRMATION:
        raise PilotError(f"Submission requires --confirm-submit {SUBMIT_CONFIRMATION}")
    if args.intent.exists() or args.receipt.exists():
        raise PilotError("Submission intent or receipt exists; refusing resubmission")
    _, qasms, _, bundle = _verify_plan(args.plan, args.bundle)
    intent_id = str(uuid.uuid4())
    intent = {
        "schema_version": SCHEMA_VERSION,
        "kind": f"{PILOT_KIND}_submission_intent",
        "intent_id": intent_id,
        "created_at_utc": q40_validate._utc_now(),
        "status": "preflight_locked",
        "backend": BACKEND,
        "circuit_count": CIRCUITS,
        "shots_per_circuit": SHOTS,
        "total_requested_shots": CIRCUITS * SHOTS,
        "bundle_sha256": bundle["sha256"],
        "aggregate_qasm_sha256": bundle["aggregate_qasm_sha256"],
        "execution_attempted": False,
        "automatic_resubmission": False,
    }
    q40_validate._atomic_write_json(args.intent, intent)
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
        if BACKEND not in supported:
            intent.update(
                {
                    "status": "backend_not_supported",
                    "backend_supported": False,
                    "updated_at_utc": q40_validate._utc_now(),
                }
            )
            q40_validate._atomic_write_json(args.intent, intent)
            raise PilotError(f"{BACKEND} is not supported by Fire Opal")
        intent.update(
            {
                "status": "execute_call_started",
                "backend_supported": True,
                "execution_attempted": True,
                "updated_at_utc": q40_validate._utc_now(),
            }
        )
        q40_validate._atomic_write_json(args.intent, intent)
        job = q40_validate._safe_provider_call(
            "Fire Opal official flat-QOS kernel pilot submission",
            fireopal.execute,
            circuits=list(qasms),
            shot_count=SHOTS,
            credentials=credentials,
            backend_name=BACKEND,
            parameters=None,
        )
        action_id = getattr(job, "action_id", None)
        if action_id is None:
            intent.update(
                {
                    "status": "submitted_but_action_id_missing",
                    "updated_at_utc": q40_validate._utc_now(),
                }
            )
            q40_validate._atomic_write_json(args.intent, intent)
            raise PilotError("Execute may have succeeded but returned no action ID")
        receipt = {
            "schema_version": SCHEMA_VERSION,
            "kind": f"{PILOT_KIND}_submission_receipt",
            "status": "submitted_not_retrieved",
            "submitted_at_utc": q40_validate._utc_now(),
            "intent_id": intent_id,
            "action_id": str(action_id),
            "backend": BACKEND,
            "circuit_count": CIRCUITS,
            "shots_per_circuit": SHOTS,
            "total_requested_shots": CIRCUITS * SHOTS,
            "bundle_sha256": bundle["sha256"],
            "aggregate_qasm_sha256": bundle["aggregate_qasm_sha256"],
            "plan_path": str(args.plan.resolve()),
            "plan_sha256": q40_validate._sha256_file(args.plan),
            "credential_source": safe_base._credential_source_labels(
                credential_source
            ),
            "qctrl_auth_source": str(qctrl_source),
            "api_calls": ["fireopal.show_supported_devices", "fireopal.execute"],
            "execution_attempted": True,
            "result_waited_during_submit": False,
            "result_retrieved": False,
            "automatic_resubmission": False,
            "claim_boundary": "Submission receipt only; no hardware result yet.",
        }
        q40_validate._atomic_write_json(args.receipt, receipt)
        intent.update(
            {
                "status": "receipt_persisted",
                "action_id": str(action_id),
                "receipt_path": str(args.receipt.resolve()),
                "updated_at_utc": q40_validate._utc_now(),
            }
        )
        q40_validate._atomic_write_json(args.intent, intent)
        return receipt
    except Exception as exc:
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
                        "updated_at_utc": q40_validate._utc_now(),
                    }
                )
                q40_validate._atomic_write_json(args.intent, persisted)
        if isinstance(exc, PilotError):
            raise
        raise PilotError(f"Submission failed ({type(exc).__name__})") from None


def _finite_shot_reference(
    ideals: Sequence[np.ndarray], *, replicates: int = 1000
) -> dict[str, Any]:
    rng = np.random.default_rng(PILOT_SEED + 10000)
    aggregate = np.empty(replicates, dtype=np.float64)
    for replicate in range(replicates):
        fidelities = []
        for ideal in ideals:
            sampled = rng.multinomial(SHOTS, ideal) / SHOTS
            fidelities.append(_hellinger_fidelity(ideal, sampled))
        aggregate[replicate] = float(np.mean(fidelities))
    return {
        "replicates": replicates,
        "shots_per_circuit": SHOTS,
        "seed": PILOT_SEED + 10000,
        "mean_of_mean_hellinger_fidelity": float(np.mean(aggregate)),
        "ci95_of_mean_hellinger_fidelity": [
            float(np.percentile(aggregate, 2.5)),
            float(np.percentile(aggregate, 97.5)),
        ],
    }


def validate_hardware_result(
    raw: Mapping[str, Any], manifest: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    distributions = safe_base._result_distributions(raw)
    if len(distributions) != CIRCUITS or len(manifest) != CIRCUITS:
        raise PilotError("Hardware result and manifest must contain 66 circuits")
    rows: list[dict[str, Any]] = []
    checks: list[dict[str, Any]] = []
    random_fidelities: list[float] = []
    random_tv: list[float] = []
    ideals: list[np.ndarray] = []
    for distribution, metadata in zip(distributions, manifest, strict=True):
        cleaned, check = safe_base._validated_distribution(
            distribution, num_qubits=QUBITS, shots=SHOTS
        )
        hardware = _dense_distribution(cleaned)
        ideal = _dense_distribution(metadata["ideal_probabilities"])
        fidelity = _hellinger_fidelity(ideal, hardware)
        total_variation = _total_variation(ideal, hardware)
        target = metadata.get("expected_target")
        target_probability = (
            None if target is None else float(hardware[int(str(target), 2)])
        )
        row = {
            "circuit_index": int(metadata["circuit_index"]),
            "role": str(metadata["role"]),
            "qasm_sha256": str(metadata["qasm_sha256"]),
            "hellinger_fidelity": fidelity,
            "total_variation_distance": total_variation,
            "expected_target": target,
            "hardware_target_probability": target_probability,
            "hardware_probabilities": {
                format(index, f"0{QUBITS}b"): float(value)
                for index, value in enumerate(hardware)
                if value > 0.0
            },
        }
        rows.append(row)
        checks.append(check)
        ideals.append(ideal)
        if metadata["role"] == "random_flat_kernel":
            random_fidelities.append(fidelity)
            random_tv.append(total_variation)
    controls = [row for row in rows if row["expected_target"] is not None]
    if len(controls) != CONTROLS or len(random_fidelities) != RANDOM_INSTANCES:
        raise PilotError("Hardware result roles differ from the frozen protocol")
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
            "ordered_against_qasm_hash_manifest": True,
            "bit_order": "Qiskit: leftmost printed bit is highest qubit index",
        },
        "random_kernel_summary": {
            "circuits": RANDOM_INSTANCES,
            "mean_hellinger_fidelity": float(np.mean(random_fidelities)),
            "median_hellinger_fidelity": float(np.median(random_fidelities)),
            "minimum_hellinger_fidelity": float(np.min(random_fidelities)),
            "maximum_hellinger_fidelity": float(np.max(random_fidelities)),
            "mean_total_variation_distance": float(np.mean(random_tv)),
            "median_total_variation_distance": float(np.median(random_tv)),
        },
        "controls": controls,
        "finite_shot_ideal_reference": _finite_shot_reference(ideals),
        "circuit_results": rows,
    }


def status_pilot(args: argparse.Namespace) -> dict[str, Any]:
    receipt = _load_json(args.receipt)
    if receipt.get("kind") != f"{PILOT_KIND}_submission_receipt":
        raise PilotError("Unexpected submission receipt kind")
    action_id = str(receipt.get("action_id", ""))
    if not action_id.isnumeric():
        raise PilotError("Submission receipt has no numeric action ID")
    fireopal, qctrl_source = safe_base._authenticated_fireopal_for_retrieval(
        args.qctrl_notebook
    )
    from fireopal.fire_opal_job import FireOpalJob

    status = q40_validate._safe_provider_call(
        "Fire Opal official flat-QOS status check",
        FireOpalJob(action_id).status,
    )
    if not isinstance(status, Mapping):
        raise PilotError("Fire Opal status payload is not a mapping")
    artifact = {
        "schema_version": SCHEMA_VERSION,
        "kind": f"{PILOT_KIND}_status",
        "captured_at_utc": q40_validate._utc_now(),
        "action_id": action_id,
        "backend": BACKEND,
        "action_status": str(status.get("action_status")),
        "status_message": str(status.get("status_message")),
        "qctrl_auth_source": str(qctrl_source),
        "api_calls": ["FireOpalJob.status"],
        "execution_attempted_in_this_mode": False,
        "automatic_resubmission": False,
    }
    q40_validate._atomic_write_json(args.status_report, artifact)
    return artifact


def retrieve_pilot(args: argparse.Namespace) -> dict[str, Any]:
    if args.result.exists() and not args.force:
        raise PilotError("Result artifact already exists; refusing to overwrite")
    receipt = _load_json(args.receipt)
    if receipt.get("kind") != f"{PILOT_KIND}_submission_receipt":
        raise PilotError("Unexpected submission receipt kind")
    action_id = str(receipt.get("action_id", ""))
    if not action_id.isnumeric():
        raise PilotError("Submission receipt has no numeric action ID")
    _, _, manifest, bundle = _verify_plan(args.plan, args.bundle)
    if receipt.get("bundle_sha256") != bundle["sha256"]:
        raise PilotError("Receipt and bundle differ")
    fireopal, qctrl_source = safe_base._authenticated_fireopal_for_retrieval(
        args.qctrl_notebook
    )
    raw = q40_validate._safe_provider_call(
        "Fire Opal official flat-QOS result retrieval",
        fireopal.get_result,
        action_id,
    )
    if not isinstance(raw, Mapping):
        raise PilotError("Fire Opal result payload is not a mapping")
    validation = validate_hardware_result(raw, manifest)
    artifact = {
        "schema_version": SCHEMA_VERSION,
        "kind": f"{PILOT_KIND}_hardware_result",
        "status": "retrieved_and_validated",
        "retrieved_at_utc": q40_validate._utc_now(),
        "action_id": action_id,
        "backend": BACKEND,
        "receipt_path": str(args.receipt.resolve()),
        "bundle_sha256": bundle["sha256"],
        "qctrl_auth_source": str(qctrl_source),
        "api_calls": ["fireopal.get_result"],
        "submission_attempted_in_this_mode": False,
        "automatic_resubmission": False,
        **validation,
        "raw_result": safe_base._json_safe_redacted(raw),
        "claim_boundary": (
            "This is a phase-sensitive hardware fidelity result for the official "
            "flat QOS sampling kernel. It demonstrates kernel execution, not an "
            "end-to-end accuracy or computational quantum advantage."
        ),
    }
    q40_validate._atomic_write_json(args.result, artifact)
    receipt.update(
        {
            "status": "result_retrieved",
            "result_retrieved": True,
            "result_path": str(args.result.resolve()),
            "result_sha256": q40_validate._sha256_file(args.result),
            "retrieved_at_utc": artifact["retrieved_at_utc"],
        }
    )
    q40_validate._atomic_write_json(args.receipt, receipt)
    return artifact


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    prepare = subparsers.add_parser("prepare")
    prepare.add_argument("--bundle", type=Path, default=DEFAULT_BUNDLE)
    prepare.add_argument("--report", type=Path, default=DEFAULT_PREPARE)
    prepare.add_argument("--force", action="store_true")
    validate = subparsers.add_parser("validate")
    validate.add_argument("--bundle", type=Path, default=DEFAULT_BUNDLE)
    validate.add_argument("--report", type=Path, default=DEFAULT_VALIDATE)
    validate.add_argument("--qiskit-account", default="default-ibm-cloud")
    validate.add_argument("--qctrl-notebook", type=Path, default=DEFAULT_QCTRL_NOTEBOOK)
    validate.add_argument("--instance")
    validate.add_argument("--force", action="store_true")
    plan = subparsers.add_parser("plan")
    plan.add_argument("--bundle", type=Path, default=DEFAULT_BUNDLE)
    plan.add_argument("--validation", type=Path, default=DEFAULT_VALIDATE)
    plan.add_argument("--plan", type=Path, default=DEFAULT_PLAN)
    plan.add_argument("--authorization", default="")
    plan.add_argument("--force", action="store_true")
    submit = subparsers.add_parser("submit")
    submit.add_argument("--bundle", type=Path, default=DEFAULT_BUNDLE)
    submit.add_argument("--plan", type=Path, default=DEFAULT_PLAN)
    submit.add_argument("--intent", type=Path, default=DEFAULT_INTENT)
    submit.add_argument("--receipt", type=Path, default=DEFAULT_RECEIPT)
    submit.add_argument("--qiskit-account", default="default-ibm-cloud")
    submit.add_argument("--qctrl-notebook", type=Path, default=DEFAULT_QCTRL_NOTEBOOK)
    submit.add_argument("--instance")
    submit.add_argument("--confirm-submit", default="")
    status = subparsers.add_parser("status")
    status.add_argument("--receipt", type=Path, default=DEFAULT_RECEIPT)
    status.add_argument("--status-report", type=Path, default=DEFAULT_STATUS)
    status.add_argument("--qctrl-notebook", type=Path, default=DEFAULT_QCTRL_NOTEBOOK)
    retrieve = subparsers.add_parser("retrieve")
    retrieve.add_argument("--bundle", type=Path, default=DEFAULT_BUNDLE)
    retrieve.add_argument("--plan", type=Path, default=DEFAULT_PLAN)
    retrieve.add_argument("--receipt", type=Path, default=DEFAULT_RECEIPT)
    retrieve.add_argument("--result", type=Path, default=DEFAULT_RESULT)
    retrieve.add_argument("--qctrl-notebook", type=Path, default=DEFAULT_QCTRL_NOTEBOOK)
    retrieve.add_argument("--force", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(sys.argv[1:] if argv is None else argv)
    if args.command == "prepare":
        report = prepare_pilot(args)
        print("Official flat-QOS kernel pilot prepared locally")
        print(f"- circuits: {report['protocol']['circuits']}")
        print(f"- max JAX amplitude error: {report['local_validation']['max_abs_qiskit_minus_official_jax_amplitude']:.3e}")
        print("- provider calls: 0")
    elif args.command == "validate":
        report = validate_pilot(args)
        print("Official flat-QOS kernel pilot Fire Opal validation complete")
        print(f"- passed: {report['provider_validation']['passed']}")
        print(f"- warnings: {report['provider_validation']['warning_summary']['total']}")
        print("- execution attempted: false")
    elif args.command == "plan":
        report = plan_pilot(args)
        print("Official flat-QOS kernel hardware plan locked")
        print(f"- circuits: {report['circuit_count']}")
        print(f"- total requested shots: {report['total_requested_shots']}")
        print("- provider calls: 0")
    elif args.command == "submit":
        report = submit_pilot(args)
        print("Official flat-QOS kernel pilot submitted")
        print(f"- action id: {report['action_id']}")
        print("- result retrieved: false")
    elif args.command == "status":
        report = status_pilot(args)
        print("Official flat-QOS kernel pilot status checked")
        print(f"- action id: {report['action_id']}")
        print(f"- status: {report['action_status']}")
        print("- automatic resubmission: false")
    else:
        report = retrieve_pilot(args)
        summary = report["random_kernel_summary"]
        print("Official flat-QOS kernel pilot result retrieved")
        print(f"- action id: {report['action_id']}")
        print(f"- mean Hellinger fidelity: {summary['mean_hellinger_fidelity']:.6f}")
        print(f"- mean total variation: {summary['mean_total_variation_distance']:.6f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
