#!/usr/bin/env python3
"""Validate the frozen q60 grid-d12 multiscale-pair batch with Fire Opal.

The paper-minimum goal is a reproducible 60-qubit hardware demonstration, not
a precondition that the local classifier already proves quantum advantage.
The multiscale-pair panel is frozen from repeated training-only CV; its small
positive CV delta is recorded as an exploratory advantage signal, while the
non-confirming fixed-test and shot-noise results remain explicit.

This runner has no hardware execution path. It exports and round-trips 195
numeric virtual-qubit QASM payloads and can call only supported-device
discovery plus ``fireopal.validate``.
"""

from __future__ import annotations

import argparse
import gzip
import json
import os
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

import qiskit_qos_pbmc68k_q60_multiscale_readout_screen as multiscale
import qiskit_qos_pbmc68k_q60_scrambled_mixer_screen as architecture
import qiskit_qos_pbmc68k_q60_shallow_fireopal_validate as q60


SCHEMA_VERSION = "1.0"
ARCHITECTURE = "grid_mixer_d12"
PANEL = "multiscale_pairs"
DEFAULT_BACKEND = "ibm_fez"
DEFAULT_SHOTS = 128
DEFAULT_MAX_PAYLOAD_DEPTH = 48
DEFAULT_SCREEN_REPORT = multiscale.DEFAULT_OUTPUT
PINNED_SCREEN_SHA256 = (
    "6ce3c100aaa755e7a6fd80017a668c0316d17ea63a50e14a772b43a96ff86f7c"
)
DEFAULT_BUNDLE = Path(
    "fire_opal_pbmc68k_q60_shallow/"
    "pbmc68k_q60_seed11_grid_d12_multiscale_pairs_fireopal_qasm2.json.gz"
)
DEFAULT_OUTPUT = Path(
    "fire_opal_pbmc68k_q60_shallow/"
    "pbmc68k_q60_seed11_grid_d12_multiscale_pairs_fireopal_validate.json"
)

RunnerError = q60.RunnerError


def _load_frozen_screen(
    path: Path,
) -> tuple[dict[str, Any], list[dict[int, str]], list[int]]:
    if not path.is_file():
        raise RunnerError(f"Multiscale screen is missing: {path}")
    actual_hash = q60.q40_validate._sha256_file(path)
    if actual_hash != PINNED_SCREEN_SHA256:
        raise RunnerError("Multiscale screen hash differs from the paper pin")
    report = json.loads(path.read_text(encoding="utf-8"))
    if (
        report.get("kind")
        != "pbmc68k_q60_grid_d12_multiscale_readout_local_screen"
        or report.get("status") != "complete_local_only"
        or report.get("execution_attempted") is not False
        or report.get("selection", {}).get("winner") != PANEL
        or report.get("config", {}).get("architecture") != ARCHITECTURE
    ):
        raise RunnerError("Multiscale screen does not freeze the expected winner")
    mappings, panels = multiscale.build_multiscale_panels()
    selected = [
        int(value)
        for value in report["winner_final_evaluation"]["selected_master_indices"]
    ]
    if (
        len(selected) != int(report["config"]["selected_features"])
        or len(selected) != len(set(selected))
        or not set(selected).issubset(panels[PANEL])
    ):
        raise RunnerError("Frozen selected observables are inconsistent")
    for index in selected:
        q60.measurement_basis_for_mapping(mappings[index])
    return report, mappings, selected


def _base_rows(data: architecture.SeedData) -> list[dict[str, Any]]:
    vectors = architecture._parameter_vectors(data)
    rows: list[dict[str, Any]] = [
        {
            "vector": vectors[0],
            "role": "weighted_training_sketch",
            "split": "train",
            "sample_position": None,
            "source_row_index": None,
            "label": None,
        }
    ]
    cursor = 1
    for split, encoded, labels, indices in (
        ("train", data.encoded_train, data.y_train, data.train_indices),
        ("test", data.encoded_test, data.y_test, data.test_indices),
    ):
        for position, (sample, label, source_index) in enumerate(
            zip(encoded, labels, indices, strict=True)
        ):
            if not np.array_equal(vectors[cursor], sample):
                raise RunnerError("Architecture parameter ordering changed")
            rows.append(
                {
                    "vector": sample,
                    "role": "query",
                    "split": split,
                    "sample_position": int(position),
                    "source_row_index": int(source_index),
                    "label": float(label),
                }
            )
            cursor += 1
    if cursor != len(vectors):
        raise RunnerError("Architecture parameter manifest is incomplete")
    return rows


def build_seed_circuits(
    data: architecture.SeedData,
    mappings: Sequence[Mapping[int, str]],
    selected: Sequence[int],
    *,
    seed_transpiler: int,
) -> tuple[list[str], list[dict[str, Any]], dict[str, Any]]:
    qasms: list[str] = []
    manifest: list[dict[str, Any]] = []
    rows = _base_rows(data)
    for base_index, row in enumerate(rows):
        circuit = architecture.architecture_circuit(row["vector"], ARCHITECTURE)
        logical_metrics = q60.q40_validate.circuit_metrics(circuit)
        for basis in q60.MEASUREMENT_BASES:
            measured = q60.measurement_circuit_for_basis(circuit, basis)
            qasm, qasm_metadata = q60.q40_validate.export_numeric_qasm2(
                measured, seed_transpiler=int(seed_transpiler)
            )
            qasms.append(qasm)
            manifest.append(
                {
                    "circuit_index": len(qasms) - 1,
                    "seed": 11,
                    "base_circuit_index": int(base_index),
                    "measurement_basis": basis,
                    "logical_metrics_before_measurement": logical_metrics,
                    "role": row["role"],
                    "split": row["split"],
                    "sample_position": row["sample_position"],
                    "source_row_index": row["source_row_index"],
                    "label": row["label"],
                    **qasm_metadata,
                }
            )
    selected_rows = [
        {
            "master_index": int(index),
            "measurement_basis": q60.measurement_basis_for_mapping(mappings[index]),
            "pauli_mapping": q60._normalized_mapping(mappings[index]),
        }
        for index in selected
    ]
    return qasms, manifest, {
        "seed": 11,
        "logical_base_circuit_count": len(rows),
        "measured_circuit_count": len(qasms),
        "measurement_bases": list(q60.MEASUREMENT_BASES),
        "selected_observable_count": len(selected_rows),
        "selected_observables": selected_rows,
        "selection_protocol": {
            "panel": PANEL,
            "panel_selected_with_repeated_training_cv_only": True,
            "observables_selected_with_training_only": True,
            "uses_test_metrics_for_hardware_configuration": False,
        },
        "train_indices": [int(value) for value in data.train_indices],
        "test_indices": [int(value) for value in data.test_indices],
        "encoded_train_sha256": q60.q40_validate._array_sha256(data.encoded_train),
        "encoded_test_sha256": q60.q40_validate._array_sha256(data.encoded_test),
    }


def _advantage_signal(screen: Mapping[str, Any]) -> dict[str, Any]:
    winner = next(row for row in screen["panels"] if row["panel"] == PANEL)
    quantum_cv = float(
        winner["training_cross_validation"]["cv_mean_balanced_accuracy"]
    )
    classical_cv = float(
        screen["classical_same_split_reference"]["cv_mean_balanced_accuracy"]
    )
    quantum_test = float(
        screen["winner_final_evaluation"]["fixed_test"]["balanced_accuracy"]
    )
    classical_test = float(
        screen["classical_same_split_reference"]["fixed_test"]["balanced_accuracy"]
    )
    return {
        "classification": "exploratory_not_confirmed",
        "positive_training_cv_delta": quantum_cv - classical_cv,
        "quantum_training_cv": quantum_cv,
        "classical_training_cv": classical_cv,
        "fixed_test_delta": quantum_test - classical_test,
        "quantum_fixed_test": quantum_test,
        "classical_fixed_test": classical_test,
        "hardware_is_scientifically_worthwhile_despite_unconfirmed_advantage": True,
        "claim": (
            "The positive repeated-CV delta and multiscale-pair selection motivate "
            "a hardware feasibility experiment; they do not establish quantum advantage."
        ),
    }


def prepare_batch(args: argparse.Namespace) -> tuple[list[str], dict[str, Any]]:
    screen, mappings, selected = _load_frozen_screen(Path(args.screen_report))
    data = architecture.load_seed_data(args)
    qasms, manifest, seed_metadata = build_seed_circuits(
        data,
        mappings,
        selected,
        seed_transpiler=int(args.seed_transpiler),
    )
    if not qasms or len(qasms) > q60.FIRE_OPAL_MAX_BATCH:
        raise RunnerError("Fire Opal batch is outside the supported size")
    logical_depths = [
        int(row["logical_metrics_before_measurement"]["depth"]) for row in manifest
    ]
    payload_depths = [int(row["metrics"]["depth"]) for row in manifest]
    qasm_hashes = [str(row["qasm_sha256"]) for row in manifest]
    aggregate_sha256 = q60.q40_validate._sha256_bytes(
        json.dumps(qasm_hashes, separators=(",", ":")).encode("utf-8")
    )
    local_validation = {
        "circuit_count": len(qasms),
        "batch_limit": q60.FIRE_OPAL_MAX_BATCH,
        "within_batch_limit": len(qasms) <= q60.FIRE_OPAL_MAX_BATCH,
        "aggregate_sha256": aggregate_sha256,
        "total_qasm_bytes": sum(int(row["qasm_bytes"]) for row in manifest),
        "max_logical_depth_before_measurement": max(logical_depths),
        "max_payload_depth": max(payload_depths),
        "payload_depth_gate": int(args.max_payload_depth),
        "payload_depth_gate_passed": max(payload_depths)
        <= int(args.max_payload_depth),
        "all_target_qubits": all(
            int(row["metrics"]["num_qubits"]) == 60 for row in manifest
        ),
        "all_one_classical_register": all(
            int(row["classical_register_count"]) == 1 for row in manifest
        ),
        "all_virtual_qubits_only": all(
            bool(row["virtual_qubits_only"]) for row in manifest
        ),
        "all_parameters_numeric": all(
            bool(row["all_parameters_numeric"]) for row in manifest
        ),
        "all_round_trips_validated": all(
            bool(row["round_trip_validated"]) for row in manifest
        ),
        "all_expected_measurement_bases": (
            {str(row["measurement_basis"]) for row in manifest}
            == set(q60.MEASUREMENT_BASES)
        ),
        "frozen_selected_observables_recoverable_from_global_bases": all(
            row["measurement_basis"] in q60.MEASUREMENT_BASES
            for row in seed_metadata["selected_observables"]
        ),
        "source_hashes_reproduced": True,
        "accuracy_gate_required_for_paper_minimum": False,
    }
    local_validation["passed"] = bool(
        local_validation["within_batch_limit"]
        and local_validation["payload_depth_gate_passed"]
        and local_validation["all_target_qubits"]
        and local_validation["all_one_classical_register"]
        and local_validation["all_virtual_qubits_only"]
        and local_validation["all_parameters_numeric"]
        and local_validation["all_round_trips_validated"]
        and local_validation["all_expected_measurement_bases"]
        and local_validation[
            "frozen_selected_observables_recoverable_from_global_bases"
        ]
    )
    return qasms, {
        "config": {
            "dataset": "PBMC68k",
            "architecture": ARCHITECTURE,
            "panel": PANEL,
            "qubits": 60,
            "seed": 11,
            "train_samples": len(data.encoded_train),
            "test_samples": len(data.encoded_test),
            "logical_base_circuits": len(_base_rows(data)),
            "measurement_bases": list(q60.MEASUREMENT_BASES),
            "measured_circuits": len(qasms),
            "shots_per_circuit_for_future_hardware": int(args.shots),
            "backend": str(args.backend),
        },
        "source": data.metadata,
        "source_artifacts": {
            "screen_report": str(Path(args.screen_report).resolve()),
            "screen_report_sha256": q60.q40_validate._sha256_file(
                Path(args.screen_report)
            ),
            "source_report": str(Path(args.source_report).resolve()),
            "source_report_sha256": q60.q40_validate._sha256_file(
                Path(args.source_report)
            ),
            "reupload_report": str(Path(args.reupload_report).resolve()),
            "reupload_report_sha256": q60.q40_validate._sha256_file(
                Path(args.reupload_report)
            ),
            "dataset_cache": q60.q40_validate._dataset_artifacts(args.cache_dir),
        },
        "seed_batch": seed_metadata,
        "manifest": manifest,
        "local_validation": local_validation,
        "advantage_signal": _advantage_signal(screen),
    }


def write_qasm_bundle(
    path: Path,
    qasms: Sequence[str],
    prepared: Mapping[str, Any],
) -> dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    payload = {
        "schema_version": SCHEMA_VERSION,
        "kind": "pbmc68k_q60_grid_d12_multiscale_pairs_numeric_qasm2_batch",
        "config": prepared["config"],
        "source": prepared["source"],
        "seed_batch": prepared["seed_batch"],
        "circuits": [
            {**row, "qasm": qasm}
            for row, qasm in zip(prepared["manifest"], qasms, strict=True)
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
    circuits = reloaded.get("circuits", [])
    if len(circuits) != len(qasms):
        raise RunnerError("QASM bundle round trip changed circuit count")
    hashes = [
        q60.q40_validate._sha256_bytes(str(row["qasm"]).encode("utf-8"))
        for row in circuits
    ]
    if hashes != [str(row["qasm_sha256"]) for row in prepared["manifest"]]:
        raise RunnerError("QASM bundle round trip changed circuit content")
    return {
        "path": str(path.resolve()),
        "bytes": path.stat().st_size,
        "sha256": q60.q40_validate._sha256_file(path),
        "circuits": len(qasms),
        "gzip_json_round_trip_passed": True,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", type=Path, default=Path("data_cache/pbmc68k"))
    parser.add_argument(
        "--source-report", type=Path, default=architecture.DEFAULT_SOURCE_REPORT
    )
    parser.add_argument(
        "--reupload-report", type=Path, default=architecture.DEFAULT_REUPLOAD_REPORT
    )
    parser.add_argument("--screen-report", type=Path, default=DEFAULT_SCREEN_REPORT)
    parser.add_argument("--shots", type=int, default=DEFAULT_SHOTS)
    parser.add_argument(
        "--seed-transpiler", type=int, default=q60.DEFAULT_SEED_TRANSPILER
    )
    parser.add_argument(
        "--max-payload-depth", type=int, default=DEFAULT_MAX_PAYLOAD_DEPTH
    )
    parser.add_argument("--backend", default=DEFAULT_BACKEND)
    parser.add_argument("--validate", action="store_true")
    parser.add_argument("--qiskit-account", default="default-ibm-cloud")
    parser.add_argument("--qctrl-notebook", type=Path)
    parser.add_argument("--instance")
    parser.add_argument("--bundle", type=Path, default=DEFAULT_BUNDLE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--force", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.shots < 1 or args.max_payload_depth < 1:
        raise RunnerError("Shots and depth gate must be positive")
    for path in (args.bundle, args.output):
        if path.exists() and not args.force:
            raise RunnerError(f"Refusing to overwrite existing artifact: {path}")
    started = time.perf_counter()
    qasms, prepared = prepare_batch(args)
    bundle = write_qasm_bundle(args.bundle, qasms, prepared)
    provider: dict[str, Any] = {
        "requested": bool(args.validate),
        "passed": None,
        "execution_attempted": False,
        "quantum_seconds_used": 0,
        "api_calls": [],
    }
    if args.validate and prepared["local_validation"]["passed"]:
        try:
            provider = {
                "requested": True,
                **q60.q40_validate.validate_fireopal_batch(
                    qasms,
                    backend=str(args.backend),
                    qiskit_account=args.qiskit_account,
                    qctrl_notebook=args.qctrl_notebook,
                    instance=args.instance,
                ),
            }
        except RunnerError as exc:
            provider = {
                "requested": True,
                "passed": False,
                "execution_attempted": False,
                "quantum_seconds_used": 0,
                "api_calls": [],
                "errors": [str(exc)],
                "warnings": [],
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
        }
    passed = bool(prepared["local_validation"]["passed"])
    if args.validate:
        passed = passed and provider.get("passed") is True
    report = {
        "schema_version": SCHEMA_VERSION,
        "kind": "pbmc68k_q60_grid_d12_multiscale_pairs_fireopal_validate_only",
        "status": "pass" if passed else "fail",
        "captured_at_utc": q60.q40_validate._utc_now(),
        "environment": q60.q40_validate.runtime_environment(),
        "execution_attempted": False,
        "quantum_seconds_used": 0,
        "allowed_provider_calls": [
            "fireopal.show_supported_devices",
            "fireopal.validate",
        ],
        "config": prepared["config"],
        "source": prepared["source"],
        "source_artifacts": prepared["source_artifacts"],
        "seed_batch": prepared["seed_batch"],
        "local_validation": prepared["local_validation"],
        "advantage_signal": prepared["advantage_signal"],
        "qasm_bundle": bundle,
        "provider_validation": provider,
        "elapsed_seconds": time.perf_counter() - started,
        "paper_minimum": {
            "hardware_feasibility_is_the_primary_goal": True,
            "accuracy_superiority_is_not_a_submission_prerequisite": True,
            "honest_classical_and_noiseless_comparisons_required": True,
        },
        "claim_boundary": (
            "A passing validation establishes 60-qubit grid-d12 input compatibility "
            "only. The positive training-CV delta is exploratory; hardware execution "
            "and a matched analysis are required for the paper, and quantum advantage "
            "is not yet established."
        ),
    }
    q60.q40_validate._atomic_write_json(args.output, report)
    print("PBMC68k q60 grid-d12 Fire Opal validation route")
    print(f"- circuits: {len(qasms)}")
    print(f"- max payload depth: {prepared['local_validation']['max_payload_depth']}")
    print(f"- local validation passed: {prepared['local_validation']['passed']}")
    print(f"- exploratory CV delta: {prepared['advantage_signal']['positive_training_cv_delta']:+.6f}")
    print(f"- provider validation requested: {args.validate}")
    if args.validate:
        print(f"- provider validation passed: {provider.get('passed')}")
        print(f"- validation action: {provider.get('validation_action_id')}")
        print(f"- warnings: {len(provider.get('warnings', []))}")
    print("- hardware execution attempted: False")
    print(f"- bundle: {args.bundle}")
    print(f"- report: {args.output}")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
