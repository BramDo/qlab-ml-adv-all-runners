#!/usr/bin/env python3
"""Build and optionally validate the frozen q60 module route with Fire Opal.

The runner exports numeric, virtual-qubit OpenQASM 2 in manifest order.  Its
only optional provider operations are supported-device discovery and
``fireopal.validate``.  There is intentionally no execution or retrieval path.
"""

from __future__ import annotations

import argparse
import gzip
import json
import os
import time
import warnings
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

import qiskit_qos_pbmc68k_q40_fireopal_validate as safe_validate
import qiskit_qos_pbmc68k_q60_module_pipeline as pipeline
import qiskit_qos_pbmc68k_q60_shallow_fireopal_validate as shallow


SCHEMA_VERSION = "1.0"
KIND = "pbmc68k_q60_coexpression_modules_b4_fireopal_validate_only"
BACKEND = "ibm_fez"
SHOTS = 128
MEASUREMENT_BASES = ("X", "Y", "Z")
MAX_PAYLOAD_DEPTH = 24
FIRE_OPAL_MAX_BATCH = 300
ARTIFACT_DIR = Path("fire_opal_pbmc68k_q60_modules_b4")


RunnerError = safe_validate.RunnerError


def phase_scope(phase: str) -> dict[str, Any]:
    if phase == "sentinel":
        return {
            "split_id": "seed11_sentinel",
            "train_samples": 32,
            "test_samples": 32,
            "base_circuits": 64,
            "measured_circuits": 192,
            "batch_sizes": [192],
            "quantum_seconds_estimate": {"low": 30, "central": 35, "high": 40},
            "phase_cap": 50,
        }
    if phase == "large":
        return {
            "split_id": "final_blind",
            "train_samples": 256,
            "test_samples": 256,
            "base_circuits": 512,
            "measured_circuits": 1536,
            "batch_sizes": [300, 300, 300, 300, 300, 36],
            "quantum_seconds_estimate": {"low": 240, "central": 280, "high": 320},
            "phase_cap": 400,
        }
    raise RunnerError("Phase must be sentinel or large")


def default_bundle(phase: str) -> Path:
    return ARTIFACT_DIR / f"pbmc68k_q60_modules_b4_seed11_{phase}_qasm2.json.gz"


def default_report(phase: str) -> Path:
    return ARTIFACT_DIR / f"pbmc68k_q60_modules_b4_seed11_{phase}_validate.json"


def batch_ranges(circuit_count: int, sizes: Sequence[int]) -> list[dict[str, int]]:
    if any(int(size) < 1 or int(size) > FIRE_OPAL_MAX_BATCH for size in sizes):
        raise RunnerError("Fire Opal batch sizes must be in 1..300")
    if sum(int(size) for size in sizes) != int(circuit_count):
        raise RunnerError("Fire Opal batch sizes do not cover every circuit")
    rows: list[dict[str, int]] = []
    start = 0
    for index, size in enumerate(sizes):
        stop = start + int(size)
        rows.append(
            {
                "batch_index": int(index),
                "start_circuit_index": int(start),
                "stop_circuit_index_exclusive": int(stop),
                "circuit_count": int(size),
            }
        )
        start = stop
    return rows


def _large_gate(
    phase: str, specification_path: Path, local_screen_path: Path
) -> dict[str, Any] | None:
    if phase != "large":
        return None
    screen = pipeline._load_json(local_screen_path, label="Five-split local screen")
    aggregate = screen.get("aggregate_gate", {})
    recorded = screen.get("specification", {})
    if (
        screen.get("kind") != pipeline.LOCAL_SCREEN_KIND
        or screen.get("completed") is not True
        or recorded.get("sha256") != pipeline._sha256_file(specification_path)
        or aggregate.get("large_hardware_phase_allowed") is not True
    ):
        raise RunnerError("Large phase is not allowed by the frozen local-screen gate")
    return {
        "path": str(local_screen_path.resolve()),
        "sha256": pipeline._sha256_file(local_screen_path),
        "status": screen.get("status"),
        "performance_gate_passed": aggregate.get("performance_gate_passed"),
        "claim_tier": aggregate.get("large_hardware_claim_tier"),
    }


def build_phase_circuits(
    specification: Mapping[str, Any],
    *,
    phase: str,
    cache_dir: Path,
    seed_transpiler: int,
) -> tuple[list[str], list[dict[str, Any]], dict[str, Any]]:
    scope = phase_scope(phase)
    data = pipeline.rebuild_split(
        specification, split_id=str(scope["split_id"]), cache_dir=cache_dir
    )
    qasms: list[str] = []
    manifest: list[dict[str, Any]] = []
    base_index = 0
    for split, blocks, labels, indices in (
        ("train", data["train_blocks"], data["y_train"], data["train_indices"]),
        ("test", data["test_blocks"], data["y_test"], data["test_indices"]),
    ):
        for sample_position, (block_row, label, source_index) in enumerate(
            zip(blocks, labels, indices, strict=True)
        ):
            circuit = pipeline.build_unmeasured_circuits(block_row[None, ...])[0]
            logical = safe_validate.circuit_metrics(circuit)
            for basis in MEASUREMENT_BASES:
                measured = shallow.measurement_circuit_for_basis(circuit, basis)
                qasm, qasm_metadata = safe_validate.export_numeric_qasm2(
                    measured, seed_transpiler=int(seed_transpiler)
                )
                qasms.append(qasm)
                manifest.append(
                    {
                        "circuit_index": int(len(qasms) - 1),
                        "base_circuit_index": int(base_index),
                        "phase": phase,
                        "seed": pipeline.SEED,
                        "role": "query",
                        "split": split,
                        "sample_position": int(sample_position),
                        "source_row_index": int(source_index),
                        "label_for_local_matched_analysis_only": int(label),
                        "labels_in_provider_payload": False,
                        "measurement_basis": basis,
                        "logical_metrics_before_measurement": logical,
                        **qasm_metadata,
                    }
                )
            base_index += 1
    if len(qasms) != int(scope["measured_circuits"]):
        raise RunnerError("Measured circuit count differs from the frozen phase")
    batches = batch_ranges(len(qasms), scope["batch_sizes"])
    for batch in batches:
        start = batch["start_circuit_index"]
        stop = batch["stop_circuit_index_exclusive"]
        hashes = [str(row["qasm_sha256"]) for row in manifest[start:stop]]
        batch["aggregate_qasm_sha256"] = safe_validate._sha256_bytes(
            json.dumps(hashes, separators=(",", ":")).encode("utf-8")
        )
    hashes = [str(row["qasm_sha256"]) for row in manifest]
    return qasms, manifest, {
        "scope": scope,
        "split": {
            "split_id": data["split_id"],
            "train_indices": data["train_indices"].tolist(),
            "test_indices": data["test_indices"].tolist(),
            "train_blocks_sha256": data["metadata"]["train_blocks_sha256"],
            "test_blocks_sha256": data["metadata"]["test_blocks_sha256"],
            "scaler": data["metadata"]["scaler"],
        },
        "batches": batches,
        "aggregate_qasm_sha256": safe_validate._sha256_bytes(
            json.dumps(hashes, separators=(",", ":")).encode("utf-8")
        ),
    }


def local_validation(
    qasms: Sequence[str], manifest: Sequence[Mapping[str, Any]], metadata: Mapping[str, Any]
) -> dict[str, Any]:
    scope = metadata["scope"]
    logical_depths = sorted(
        {int(row["logical_metrics_before_measurement"]["depth"]) for row in manifest}
    )
    logical_two_qubit = sorted(
        {
            int(row["logical_metrics_before_measurement"]["two_qubit_gates"])
            for row in manifest
        }
    )
    payload_depths = [int(row["metrics"]["depth"]) for row in manifest]
    expected_mapping = [
        {"qubit": index, "clbit": index} for index in range(pipeline.QUBITS)
    ]
    checks = {
        "circuit_count": len(qasms),
        "expected_circuit_count": int(scope["measured_circuits"]),
        "batches": metadata["batches"],
        "all_batches_within_limit": all(
            int(row["circuit_count"]) <= FIRE_OPAL_MAX_BATCH
            for row in metadata["batches"]
        ),
        "logical_depths": logical_depths,
        "expected_logical_depth": pipeline.EXPECTED_LOGICAL_DEPTH,
        "logical_two_qubit_gate_counts": logical_two_qubit,
        "expected_logical_two_qubit_gates": pipeline.EXPECTED_LOGICAL_TWO_QUBIT_GATES,
        "payload_depths": sorted(set(payload_depths)),
        "max_payload_depth": max(payload_depths),
        "payload_depth_gate": MAX_PAYLOAD_DEPTH,
        "all_target_qubits": all(
            int(row["metrics"]["num_qubits"]) == pipeline.QUBITS for row in manifest
        ),
        "all_one_classical_register": all(
            int(row["classical_register_count"]) == 1 for row in manifest
        ),
        "all_virtual_qubits_only": all(bool(row["virtual_qubits_only"]) for row in manifest),
        "all_parameters_numeric": all(bool(row["all_parameters_numeric"]) for row in manifest),
        "all_round_trips_validated": all(bool(row["round_trip_validated"]) for row in manifest),
        "all_measurements_identity_mapped": all(
            list(row["measurement_mapping"]) == expected_mapping for row in manifest
        ),
        "all_expected_measurement_bases": {
            str(row["measurement_basis"]) for row in manifest
        }
        == set(MEASUREMENT_BASES),
        "aggregate_qasm_sha256": metadata["aggregate_qasm_sha256"],
        "total_qasm_bytes": int(sum(int(row["qasm_bytes"]) for row in manifest)),
    }
    checks["passed"] = bool(
        checks["circuit_count"] == checks["expected_circuit_count"]
        and checks["all_batches_within_limit"]
        and checks["logical_depths"] == [pipeline.EXPECTED_LOGICAL_DEPTH]
        and checks["logical_two_qubit_gate_counts"]
        == [pipeline.EXPECTED_LOGICAL_TWO_QUBIT_GATES]
        and checks["max_payload_depth"] <= MAX_PAYLOAD_DEPTH
        and checks["all_target_qubits"]
        and checks["all_one_classical_register"]
        and checks["all_virtual_qubits_only"]
        and checks["all_parameters_numeric"]
        and checks["all_round_trips_validated"]
        and checks["all_measurements_identity_mapped"]
        and checks["all_expected_measurement_bases"]
    )
    return checks


def write_bundle(
    path: Path,
    qasms: Sequence[str],
    manifest: Sequence[Mapping[str, Any]],
    metadata: Mapping[str, Any],
    specification_path: Path,
    local_gate: Mapping[str, Any] | None,
) -> dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    payload = {
        "schema_version": SCHEMA_VERSION,
        "kind": "pbmc68k_q60_coexpression_modules_b4_numeric_qasm2_bundle",
        "phase": metadata["scope"]["split_id"],
        "config": {
            "architecture": pipeline.ARCHITECTURE,
            "qubits": pipeline.QUBITS,
            "block_count": pipeline.BLOCK_COUNT,
            "scale_law": pipeline.SCALE_LAW,
            "seed": pipeline.SEED,
            "backend": BACKEND,
            "shots": SHOTS,
            "measurement_bases": list(MEASUREMENT_BASES),
        },
        "specification": {
            "path": str(specification_path.resolve()),
            "sha256": pipeline._sha256_file(specification_path),
        },
        "large_phase_gate": local_gate,
        "scope": metadata["scope"],
        "split": metadata["split"],
        "batches": metadata["batches"],
        "aggregate_qasm_sha256": metadata["aggregate_qasm_sha256"],
        "circuits": [
            {**dict(row), "qasm": qasm}
            for row, qasm in zip(manifest, qasms, strict=True)
        ],
    }
    try:
        with gzip.open(temporary, "wt", encoding="utf-8", compresslevel=9) as handle:
            json.dump(payload, handle, sort_keys=True, separators=(",", ":"))
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        reloaded = json.load(handle)
    rows = reloaded.get("circuits", [])
    if len(rows) != len(qasms):
        raise RunnerError("Gzip JSON round trip changed the circuit count")
    hashes = [
        safe_validate._sha256_bytes(str(row["qasm"]).encode("utf-8")) for row in rows
    ]
    if hashes != [str(row["qasm_sha256"]) for row in manifest]:
        raise RunnerError("Gzip JSON round trip changed circuit content")
    return {
        "path": str(path.resolve()),
        "bytes": path.stat().st_size,
        "sha256": pipeline._sha256_file(path),
        "circuits": len(rows),
        "aggregate_qasm_sha256": metadata["aggregate_qasm_sha256"],
        "gzip_json_round_trip_passed": True,
    }


def validate_batches_with_provider(
    qasms: Sequence[str],
    batches: Sequence[Mapping[str, Any]],
    *,
    backend: str,
    qiskit_account: str | None,
    qctrl_notebook: Path | None,
    instance: str | None,
) -> dict[str, Any]:
    fireopal, credentials, credential_source, qctrl_source = (
        safe_validate._fire_opal_credentials_from_source(
            qiskit_account, qctrl_notebook, instance
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
    if backend not in supported:
        return {
            "requested": True,
            "passed": False,
            "execution_attempted": False,
            "quantum_seconds_used": 0,
            "backend": backend,
            "backend_supported": False,
            "errors": ["Backend is not in Fire Opal supported devices"],
            "warnings": [],
            "credential_source": credential_source,
            "qctrl_auth_source": qctrl_source,
            "api_calls": ["fireopal.show_supported_devices"],
            "batches": [],
        }
    batch_results: list[dict[str, Any]] = []
    for batch in batches:
        start = int(batch["start_circuit_index"])
        stop = int(batch["stop_circuit_index_exclusive"])
        with warnings.catch_warnings(record=True) as surfaced:
            warnings.simplefilter("always", RuntimeWarning)
            job = safe_validate._safe_provider_call(
                "Fire Opal q60 module compatibility validation",
                fireopal.validate,
                circuits=list(qasms[start:stop]),
                credentials=credentials,
                backend_name=backend,
            )
            action_id = getattr(job, "action_id", None)
            result = safe_validate._workflow_result(job)
        errors = [
            str(value) for value in result.get("results", []) if value not in (None, "")
        ]
        returned_warnings = [
            str(value) for value in result.get("warnings", []) if value not in (None, "")
        ]
        surfaced_warnings = [str(item.message) for item in surfaced]
        batch_results.append(
            {
                **dict(batch),
                "passed": not errors,
                "validation_action_id": str(action_id) if action_id is not None else None,
                "errors": errors,
                "warnings": returned_warnings + surfaced_warnings,
            }
        )
        if errors:
            break
    all_errors = [error for row in batch_results for error in row["errors"]]
    all_warnings = [warning for row in batch_results for warning in row["warnings"]]
    return {
        "requested": True,
        "passed": bool(len(batch_results) == len(batches) and not all_errors),
        "execution_attempted": False,
        "quantum_seconds_used": 0,
        "backend": backend,
        "backend_supported": True,
        "circuits_validated": int(sum(row["circuit_count"] for row in batch_results)),
        "credential_source": credential_source,
        "qctrl_auth_source": qctrl_source,
        "api_calls": [
            "fireopal.show_supported_devices",
            *["fireopal.validate" for _ in batch_results],
        ],
        "errors": all_errors,
        "warnings": all_warnings,
        "batches": batch_results,
    }


def prepare_and_validate(args: argparse.Namespace) -> dict[str, Any]:
    phase = str(args.phase)
    bundle_path = args.bundle or default_bundle(phase)
    output_path = args.output or default_report(phase)
    for path in (bundle_path, output_path):
        if path.exists() and not args.force:
            raise RunnerError(f"Refusing to overwrite existing artifact: {path}")
    specification = pipeline.load_prepared_specification(args.specification)
    gate = _large_gate(phase, args.specification, args.local_screen)
    started = time.perf_counter()
    qasms, manifest, metadata = build_phase_circuits(
        specification,
        phase=phase,
        cache_dir=args.cache_dir,
        seed_transpiler=int(args.seed_transpiler),
    )
    local = local_validation(qasms, manifest, metadata)
    bundle = write_bundle(
        bundle_path, qasms, manifest, metadata, args.specification, gate
    )
    provider: dict[str, Any] = {
        "requested": False,
        "passed": None,
        "execution_attempted": False,
        "quantum_seconds_used": 0,
        "api_calls": [],
        "batches": [],
    }
    if args.validate and local["passed"]:
        try:
            provider = validate_batches_with_provider(
                qasms,
                metadata["batches"],
                backend=str(args.backend),
                qiskit_account=args.qiskit_account,
                qctrl_notebook=args.qctrl_notebook,
                instance=args.instance,
            )
        except RunnerError as exc:
            provider = {
                "requested": True,
                "passed": False,
                "execution_attempted": False,
                "quantum_seconds_used": 0,
                "api_calls": [],
                "errors": [str(exc)],
                "warnings": [],
                "batches": [],
            }
    elif args.validate:
        provider = {
            "requested": True,
            "passed": False,
            "execution_attempted": False,
            "quantum_seconds_used": 0,
            "api_calls": [],
            "errors": ["Provider validation skipped because local gates failed"],
            "warnings": [],
            "batches": [],
        }
    passed = bool(local["passed"] and (not args.validate or provider.get("passed") is True))
    report = {
        "schema_version": SCHEMA_VERSION,
        "kind": KIND,
        "status": "pass" if passed else "fail",
        "captured_at_utc": pipeline._utc_now(),
        "environment": safe_validate.runtime_environment(),
        "execution_attempted": False,
        "quantum_seconds_used": 0,
        "allowed_provider_calls": ["fireopal.show_supported_devices", "fireopal.validate"],
        "phase": phase,
        "config": {
            "architecture": pipeline.ARCHITECTURE,
            "seed": pipeline.SEED,
            "qubits": pipeline.QUBITS,
            "block_count": pipeline.BLOCK_COUNT,
            "scale_law": pipeline.SCALE_LAW,
            "backend": str(args.backend),
            "shots_per_circuit_for_future_hardware": SHOTS,
            **metadata["scope"],
        },
        "specification": {
            "path": str(args.specification.resolve()),
            "sha256": pipeline._sha256_file(args.specification),
        },
        "large_phase_gate": gate,
        "split": metadata["split"],
        "local_validation": local,
        "qasm_bundle": bundle,
        "provider_validation": provider,
        "elapsed_seconds": time.perf_counter() - started,
        "claim_boundary": (
            "A passing Fire Opal validation establishes payload compatibility only. "
            "It spends zero quantum seconds and is not a hardware or advantage result."
        ),
    }
    pipeline._atomic_write_json(output_path, report)
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("sentinel", "large"), default="sentinel")
    parser.add_argument("--cache-dir", type=Path, default=Path("data_cache/pbmc68k"))
    parser.add_argument("--specification", type=Path, default=pipeline.DEFAULT_SPECIFICATION)
    parser.add_argument("--local-screen", type=Path, default=pipeline.DEFAULT_LOCAL_SCREEN)
    parser.add_argument("--backend", default=BACKEND)
    parser.add_argument("--seed-transpiler", type=int, default=safe_validate.DEFAULT_SEED_TRANSPILER)
    parser.add_argument("--validate", action="store_true")
    parser.add_argument("--qiskit-account", default="default-ibm-cloud")
    parser.add_argument("--qctrl-notebook", type=Path)
    parser.add_argument("--instance")
    parser.add_argument("--bundle", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--force", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if str(args.backend) != BACKEND:
        raise RunnerError("The frozen route does not permit a backend switch")
    report = prepare_and_validate(args)
    print("PBMC68k q60 module B=4 Fire Opal validate-only route")
    print(f"- phase: {report['phase']}")
    print(f"- circuits: {report['local_validation']['circuit_count']}")
    print(f"- batches: {[row['circuit_count'] for row in report['local_validation']['batches']]}")
    print(f"- local validation passed: {report['local_validation']['passed']}")
    print(f"- provider validation requested: {args.validate}")
    print("- hardware execution attempted: False")
    print("- quantum seconds used: 0")
    print(f"- bundle: {args.bundle or default_bundle(args.phase)}")
    print(f"- report: {args.output or default_report(args.phase)}")
    return 0 if report["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
