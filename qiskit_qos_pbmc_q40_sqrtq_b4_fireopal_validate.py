#!/usr/bin/env python3
"""Validate the frozen q40 sqrt(q), B=4 PBMC68k pilot with Fire Opal.

The virtual batch contains the existing seed-11 split only: 32 training and
32 test samples, each measured in the global X, Y, and Z bases (192 circuits).
The architecture is frozen from a label-free local resource preflight.  This
runner has no hardware execution or result-retrieval path: the only optional
provider calls are supported-device discovery and ``fireopal.validate``.
"""

from __future__ import annotations

import argparse
import gzip
import json
import math
import os
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

import qiskit_qos_pbmc68k_q60_shallow_fireopal_validate as shallow
import qiskit_qos_pbmc68k_utils as pbmc
import qiskit_qos_pbmc_coherent_stream_hardness_screen as coherent
import qiskit_qos_pbmc_width_scaled_entangler_screen as width_scaled


SCHEMA_VERSION = "1.0"
KIND = "pbmc68k_q40_sqrtq_b4_fireopal_validate_only"
ARCHITECTURE = "coherent_stream_b4_width_scaled_entangler"
SCALE_LAW = "sqrt_q"
QUBITS = 40
BLOCK_COUNT = 4
SEED = 11
TRAIN_SAMPLES = 32
TEST_SAMPLES = 32
MEASUREMENT_BASES = ("X", "Y", "Z")
EXPECTED_CIRCUITS = (TRAIN_SAMPLES + TEST_SAMPLES) * len(MEASUREMENT_BASES)
EXPECTED_LOGICAL_DEPTH = 20
EXPECTED_LOGICAL_TWO_QUBIT_GATES = 87
DEFAULT_BACKEND = "ibm_fez"
DEFAULT_SHOTS = 128
DEFAULT_MAX_PAYLOAD_DEPTH = 24
DEFAULT_SEED_TRANSPILER = shallow.DEFAULT_SEED_TRANSPILER
DEFAULT_SCREEN_REPORT = width_scaled.DEFAULT_OUTPUT
PINNED_SCREEN_SHA256 = (
    "70755a4431bc0144fcaef1157b4d41b00eec69537f157318acf2b4ac75aa7448"
)
DEFAULT_SOURCE_REPORT = Path(
    "fire_opal_pbmc68k_q60_shallow/"
    "pbmc68k_q60_seed11_train_only_tuning.json"
)
PINNED_SOURCE_REPORT_SHA256 = (
    "dbb8c6ed6ab5d90bfe99ea53e8966e9a5694d491345d922b7d754528b537385c"
)
DEFAULT_BUNDLE = Path(
    "fire_opal_pbmc68k_q40_sqrtq_b4/"
    "pbmc68k_q40_sqrtq_b4_seed11_fireopal_qasm2.json.gz"
)
DEFAULT_OUTPUT = Path(
    "fire_opal_pbmc68k_q40_sqrtq_b4/"
    "pbmc68k_q40_sqrtq_b4_seed11_fireopal_validate.json"
)

RunnerError = shallow.RunnerError
q40_validate = shallow.q40_validate


def _load_json(path: Path, *, label: str) -> dict[str, Any]:
    if not path.is_file():
        raise RunnerError(f"{label} is missing: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RunnerError(f"{label} could not be read ({type(exc).__name__})") from None
    if not isinstance(value, dict):
        raise RunnerError(f"{label} is not a JSON object")
    return value


def load_frozen_specifications(
    screen_path: Path, source_path: Path
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Load and verify the exact local screen and seed-11 split artifacts."""

    if q40_validate._sha256_file(screen_path) != PINNED_SCREEN_SHA256:
        raise RunnerError("Width-scaled screen hash differs from the frozen pilot")
    screen = _load_json(screen_path, label="Width-scaled screen")
    configuration = screen.get("configuration", {})
    confirmation = screen.get("confirmation_gate", {})
    preflight = screen.get("preflight_funnel", {})
    if (
        screen.get("kind") != "pbmc_b4_width_scaled_entangler_mps_screen"
        or screen.get("completed") is not True
        or screen.get("execution_attempted") is not False
        or int(screen.get("provider_calls_made", -1)) != 0
        or int(configuration.get("block_count", -1)) != BLOCK_COUNT
        or list(configuration.get("widths", [])) != [20, 30, 40]
        or confirmation.get("structural_candidate_for_next_gate") != SCALE_LAW
        or SCALE_LAW not in preflight.get("survivors", [])
    ):
        raise RunnerError("Width-scaled screen does not freeze the expected candidate")

    candidate = next(
        (
            row
            for row in preflight.get("candidates", [])
            if row.get("scale_law") == SCALE_LAW
        ),
        None,
    )
    if not isinstance(candidate, Mapping) or candidate.get(
        "qualified_for_confirmation"
    ) is not True:
        raise RunnerError("sqrt(q) did not pass the frozen label-free preflight")

    if q40_validate._sha256_file(source_path) != PINNED_SOURCE_REPORT_SHA256:
        raise RunnerError("Seed-11 source report hash differs from the frozen pilot")
    source = _load_json(source_path, label="Seed-11 source report")
    source_config = source.get("config", {})
    split = source.get("split", {})
    if (
        int(source_config.get("seed", -1)) != SEED
        or int(source_config.get("train_samples", -1)) != TRAIN_SAMPLES
        or int(source_config.get("test_samples", -1)) != TEST_SAMPLES
        or len(split.get("train_indices", [])) != TRAIN_SAMPLES
        or len(split.get("test_indices", [])) != TEST_SAMPLES
    ):
        raise RunnerError("Source report does not contain the frozen 32/32 seed-11 split")
    return screen, source


def load_seed_blocks(
    args: argparse.Namespace,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    """Rebuild B=4 blocks for the frozen seed-11 rows from the cached dataset."""

    screen, source_report = load_frozen_specifications(
        Path(args.screen_report), Path(args.source_report)
    )
    screen_config = screen["configuration"]
    x, labels, source_meta = pbmc.load_pbmc68k(cache_dir=str(args.cache_dir))
    x_pair, y_pair, pair_meta = pbmc.select_binary_pair(
        x,
        labels,
        positive_label=str(screen_config["positive_label"]),
        negative_label=str(screen_config["negative_label"]),
    )
    if (
        coherent.flat_gate._sha256_sparse(x_pair)
        != str(screen["dataset"]["sparse_matrix_sha256"])
        or coherent.flat_gate._sha256_array(y_pair)
        != str(screen["dataset"]["labels_sha256"])
    ):
        raise RunnerError("Cached PBMC68k binary pair does not reproduce the screen")

    train_indices = np.asarray(
        source_report["split"]["train_indices"], dtype=np.int64
    )
    test_indices = np.asarray(source_report["split"]["test_indices"], dtype=np.int64)
    if (
        len(np.unique(train_indices)) != TRAIN_SAMPLES
        or len(np.unique(test_indices)) != TEST_SAMPLES
        or np.intersect1d(train_indices, test_indices).size
        or np.min(np.concatenate((train_indices, test_indices))) < 0
        or np.max(np.concatenate((train_indices, test_indices))) >= x_pair.shape[0]
    ):
        raise RunnerError("Frozen seed-11 row indices are invalid")

    encoding = {
        "num_qubits": QUBITS,
        "block_count": BLOCK_COUNT,
        "hash_seed": int(screen_config["pbmc_hash_seed"]),
        "value_mode": "log-product",
        "max_active_genes": int(screen_config["pbmc_max_active_genes"]),
    }
    train_blocks, train_stats = coherent.build_coherent_blocks(
        x_pair[train_indices], **encoding
    )
    test_blocks, test_stats = coherent.build_coherent_blocks(
        x_pair[test_indices], **encoding
    )
    return (
        train_blocks,
        test_blocks,
        np.asarray(y_pair[train_indices], dtype=np.float64),
        np.asarray(y_pair[test_indices], dtype=np.float64),
        train_indices,
        test_indices,
        {
            "source": {**source_meta, **pair_meta},
            "encoding": encoding,
            "train_encoding_stats": train_stats,
            "test_encoding_stats": test_stats,
            "train_blocks_sha256": coherent._array_sha256(train_blocks),
            "test_blocks_sha256": coherent._array_sha256(test_blocks),
        },
    )


def build_seed_circuits(
    train_blocks: np.ndarray,
    test_blocks: np.ndarray,
    y_train: np.ndarray,
    y_test: np.ndarray,
    train_indices: Sequence[int],
    test_indices: Sequence[int],
    *,
    seed_transpiler: int = DEFAULT_SEED_TRANSPILER,
) -> tuple[list[str], list[dict[str, Any]], dict[str, Any]]:
    """Build exactly three measured virtual circuits for every frozen sample."""

    train_blocks = np.asarray(train_blocks, dtype=np.float64)
    test_blocks = np.asarray(test_blocks, dtype=np.float64)
    y_train = np.asarray(y_train, dtype=np.float64)
    y_test = np.asarray(y_test, dtype=np.float64)
    expected_shape = (BLOCK_COUNT, QUBITS)
    if train_blocks.shape[1:] != expected_shape or test_blocks.shape[1:] != expected_shape:
        raise RunnerError("Encoded blocks must have shape N x 4 x 40")
    if (
        len(train_blocks) != len(y_train)
        or len(test_blocks) != len(y_test)
        or len(train_blocks) != len(train_indices)
        or len(test_blocks) != len(test_indices)
    ):
        raise RunnerError("Block, label, and index lengths differ")

    multiplier = width_scaled.pair_multiplier(SCALE_LAW, QUBITS)
    qasms: list[str] = []
    manifest: list[dict[str, Any]] = []
    base_index = 0
    for split, blocks, labels, source_indices in (
        ("train", train_blocks, y_train, train_indices),
        ("test", test_blocks, y_test, test_indices),
    ):
        for sample_position, (sample, label, source_index) in enumerate(
            zip(blocks, labels, source_indices, strict=True)
        ):
            circuit = coherent.coherent_stream_circuit(
                sample, pair_multiplier=multiplier
            )
            logical_metrics = q40_validate.circuit_metrics(circuit)
            for basis in MEASUREMENT_BASES:
                measured = shallow.measurement_circuit_for_basis(circuit, basis)
                qasm, qasm_metadata = q40_validate.export_numeric_qasm2(
                    measured, seed_transpiler=int(seed_transpiler)
                )
                qasms.append(qasm)
                manifest.append(
                    {
                        "circuit_index": len(qasms) - 1,
                        "base_circuit_index": int(base_index),
                        "seed": SEED,
                        "role": "query",
                        "split": split,
                        "sample_position": int(sample_position),
                        "source_row_index": int(source_index),
                        "label_for_local_matched_analysis_only": float(label),
                        "measurement_basis": basis,
                        "logical_metrics_before_measurement": logical_metrics,
                        **qasm_metadata,
                    }
                )
            base_index += 1

    return qasms, manifest, {
        "seed": SEED,
        "train_samples": len(train_blocks),
        "test_samples": len(test_blocks),
        "logical_base_circuit_count": int(base_index),
        "measured_circuit_count": len(qasms),
        "measurement_bases": list(MEASUREMENT_BASES),
        "train_indices": [int(value) for value in train_indices],
        "test_indices": [int(value) for value in test_indices],
        "train_blocks_sha256": coherent._array_sha256(train_blocks),
        "test_blocks_sha256": coherent._array_sha256(test_blocks),
        "pair_multiplier": multiplier,
        "selection_protocol": {
            "architecture_selected_by_label_free_resource_preflight": True,
            "test_inputs_used_for_architecture_selection": False,
            "test_labels_used_for_architecture_selection": False,
            "labels_in_provider_payload": False,
        },
    }


def prepare_batch(args: argparse.Namespace) -> tuple[list[str], dict[str, Any]]:
    (
        train_blocks,
        test_blocks,
        y_train,
        y_test,
        train_indices,
        test_indices,
        source_metadata,
    ) = load_seed_blocks(args)
    qasms, manifest, seed_metadata = build_seed_circuits(
        train_blocks,
        test_blocks,
        y_train,
        y_test,
        train_indices,
        test_indices,
        seed_transpiler=int(args.seed_transpiler),
    )
    combined_blocks = np.concatenate((train_blocks, test_blocks), axis=0)
    angle_statistics = width_scaled.rzz_angle_statistics(
        combined_blocks, scale_law=SCALE_LAW
    )
    logical_depths = [
        int(row["logical_metrics_before_measurement"]["depth"]) for row in manifest
    ]
    logical_two_qubit = [
        int(row["logical_metrics_before_measurement"]["two_qubit_gates"])
        for row in manifest
    ]
    payload_depths = [int(row["metrics"]["depth"]) for row in manifest]
    qasm_hashes = [str(row["qasm_sha256"]) for row in manifest]
    aggregate_sha256 = q40_validate._sha256_bytes(
        json.dumps(qasm_hashes, separators=(",", ":")).encode("utf-8")
    )
    local_validation = {
        "circuit_count": len(qasms),
        "expected_circuit_count": EXPECTED_CIRCUITS,
        "batch_limit": shallow.FIRE_OPAL_MAX_BATCH,
        "within_batch_limit": 0 < len(qasms) <= shallow.FIRE_OPAL_MAX_BATCH,
        "aggregate_qasm_sha256": aggregate_sha256,
        "total_qasm_bytes": sum(int(row["qasm_bytes"]) for row in manifest),
        "logical_depths": sorted(set(logical_depths)),
        "expected_logical_depth": EXPECTED_LOGICAL_DEPTH,
        "logical_two_qubit_gate_counts": sorted(set(logical_two_qubit)),
        "expected_logical_two_qubit_gates": EXPECTED_LOGICAL_TWO_QUBIT_GATES,
        "payload_depths": sorted(set(payload_depths)),
        "max_payload_depth": max(payload_depths),
        "payload_depth_gate": int(args.max_payload_depth),
        "payload_depth_gate_passed": max(payload_depths) <= int(args.max_payload_depth),
        "all_target_qubits": all(
            int(row["metrics"]["num_qubits"]) == QUBITS for row in manifest
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
            == set(MEASUREMENT_BASES)
        ),
        "all_rzz_angles_abs_le_pi": angle_statistics["fraction_abs_above_pi"] == 0.0,
        "source_hashes_reproduced": True,
    }
    local_validation["passed"] = bool(
        local_validation["circuit_count"] == EXPECTED_CIRCUITS
        and local_validation["within_batch_limit"]
        and local_validation["logical_depths"] == [EXPECTED_LOGICAL_DEPTH]
        and local_validation["logical_two_qubit_gate_counts"]
        == [EXPECTED_LOGICAL_TWO_QUBIT_GATES]
        and local_validation["payload_depth_gate_passed"]
        and local_validation["all_target_qubits"]
        and local_validation["all_one_classical_register"]
        and local_validation["all_virtual_qubits_only"]
        and local_validation["all_parameters_numeric"]
        and local_validation["all_round_trips_validated"]
        and local_validation["all_expected_measurement_bases"]
        and local_validation["all_rzz_angles_abs_le_pi"]
    )
    return qasms, {
        "config": {
            "dataset": "PBMC68k",
            "architecture": ARCHITECTURE,
            "scale_law": SCALE_LAW,
            "pair_multiplier": math.sqrt(QUBITS),
            "qubits": QUBITS,
            "block_count": BLOCK_COUNT,
            "seed": SEED,
            "train_samples": TRAIN_SAMPLES,
            "test_samples": TEST_SAMPLES,
            "logical_base_circuits": TRAIN_SAMPLES + TEST_SAMPLES,
            "measurement_bases": list(MEASUREMENT_BASES),
            "measured_circuits": len(qasms),
            "shots_per_circuit_for_future_hardware": int(args.shots),
            "backend": str(args.backend),
        },
        "source": source_metadata,
        "source_artifacts": {
            "width_scaled_screen": str(Path(args.screen_report).resolve()),
            "width_scaled_screen_sha256": q40_validate._sha256_file(
                Path(args.screen_report)
            ),
            "seed11_source_report": str(Path(args.source_report).resolve()),
            "seed11_source_report_sha256": q40_validate._sha256_file(
                Path(args.source_report)
            ),
            "dataset_cache": q40_validate._dataset_artifacts(Path(args.cache_dir)),
        },
        "seed_batch": seed_metadata,
        "angle_statistics": angle_statistics,
        "manifest": manifest,
        "local_validation": local_validation,
    }


def write_qasm_bundle(
    path: Path, qasms: Sequence[str], prepared: Mapping[str, Any]
) -> dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    payload = {
        "schema_version": SCHEMA_VERSION,
        "kind": "pbmc68k_q40_sqrtq_b4_seed11_numeric_qasm2_batch",
        "config": prepared["config"],
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
        q40_validate._sha256_bytes(str(row["qasm"]).encode("utf-8"))
        for row in circuits
    ]
    if hashes != [str(row["qasm_sha256"]) for row in prepared["manifest"]]:
        raise RunnerError("QASM bundle round trip changed circuit content")
    return {
        "path": str(path.resolve()),
        "bytes": path.stat().st_size,
        "sha256": q40_validate._sha256_file(path),
        "circuits": len(qasms),
        "gzip_json_round_trip_passed": True,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", type=Path, default=Path("data_cache/pbmc68k"))
    parser.add_argument("--screen-report", type=Path, default=DEFAULT_SCREEN_REPORT)
    parser.add_argument("--source-report", type=Path, default=DEFAULT_SOURCE_REPORT)
    parser.add_argument("--shots", type=int, default=DEFAULT_SHOTS)
    parser.add_argument("--max-payload-depth", type=int, default=DEFAULT_MAX_PAYLOAD_DEPTH)
    parser.add_argument("--seed-transpiler", type=int, default=DEFAULT_SEED_TRANSPILER)
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
                **q40_validate.validate_fireopal_batch(
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
        "kind": KIND,
        "status": "pass" if passed else "fail",
        "captured_at_utc": q40_validate._utc_now(),
        "environment": q40_validate.runtime_environment(),
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
        "angle_statistics": prepared["angle_statistics"],
        "local_validation": prepared["local_validation"],
        "qasm_bundle": bundle,
        "provider_validation": provider,
        "elapsed_seconds": time.perf_counter() - started,
        "claim_boundary": (
            "A passing Fire Opal validation establishes input compatibility only. "
            "The sqrt(q) architecture survived a bounded label-free MPS resource "
            "preflight, but that is not proof of classical hardness, predictive "
            "superiority, hardware performance, or quantum advantage."
        ),
    }
    q40_validate._atomic_write_json(args.output, report)
    print("PBMC68k q40 sqrt(q) B=4 Fire Opal validate-only route")
    print(f"- circuits: {len(qasms)}")
    print(f"- logical depth: {prepared['local_validation']['logical_depths']}")
    print(f"- max payload depth: {prepared['local_validation']['max_payload_depth']}")
    print(f"- local validation passed: {prepared['local_validation']['passed']}")
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
