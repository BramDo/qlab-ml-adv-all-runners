#!/usr/bin/env python3
"""Validate the frozen q60 RX(0.5x) re-uploading batch with Fire Opal.

The runner reuses the exact seed-11 PBMC68k indices and hashes from the first
train-only tuning artifact.  It freezes single=0.75, phase=0.25, pair=0.95,
adds RX(0.5*x) after the two-layer nearest-neighbour RZZ path, and selects the
balanced 8/8/8 observable representation using training data only.

This file deliberately has no hardware execution path.  It exports and
round-trips numeric virtual-qubit QASM, enforces local payload gates, and calls
at most Fire Opal supported-device discovery plus validate.
"""

from __future__ import annotations

import argparse
import json
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
from qiskit import QuantumCircuit

import qiskit_qos_pbmc68k_pairwise_screen as pairwise_screen
import qiskit_qos_pbmc68k_q60_balanced_representation as balanced
import qiskit_qos_pbmc68k_q60_reuploading_tune as reupload
import qiskit_qos_pbmc68k_q60_shallow_fireopal_validate as q60
import qiskit_qos_pbmc68k_utils as pbmc
import qiskit_qos_toy_model as toy


SCHEMA_VERSION = "1.0"
DEFAULT_BACKEND = "ibm_fez"
DEFAULT_ARCHITECTURE = "rx_0p5"
DEFAULT_READOUT_SHOTS = 128
DEFAULT_MAX_PAYLOAD_DEPTH = 24
DEFAULT_SOURCE_TUNING = Path(
    "fire_opal_pbmc68k_q60_shallow/pbmc68k_q60_seed11_train_only_tuning.json"
)
DEFAULT_REUPLOAD_REPORT = Path(
    "fire_opal_pbmc68k_q60_shallow/pbmc68k_q60_balanced_reuploading_tuning.json"
)
DEFAULT_BUNDLE = Path(
    "fire_opal_pbmc68k_q60_shallow/"
    "pbmc68k_q60_seed11_rx05_fireopal_qasm2.json.gz"
)
DEFAULT_OUTPUT = Path(
    "fire_opal_pbmc68k_q60_shallow/"
    "pbmc68k_q60_seed11_rx05_fireopal_validate.json"
)

RunnerError = q60.RunnerError


def rx05_circuit(
    linear_values: np.ndarray,
    pair_values: np.ndarray,
    *,
    single_scale: float,
    phase_scale: float,
    pair_scale: float,
    post_scale: float = 0.5,
) -> QuantumCircuit:
    """Build H-RY-RZ-RZZ-RX with the frozen nearest-neighbour path."""

    linear_values = np.asarray(linear_values, dtype=np.float64)
    circuit = q60.shallow_circuit(
        linear_values,
        pair_values,
        single_scale=float(single_scale),
        phase_scale=float(phase_scale),
        pair_scale=float(pair_scale),
    )
    for qubit, value in enumerate(linear_values):
        circuit.rx(float(post_scale) * float(value), qubit)
    return circuit


def _configuration_from_reports(
    source_report: Mapping[str, Any], reupload_report: Mapping[str, Any]
) -> dict[str, Any]:
    source_config = source_report["config"]
    protocol = reupload_report["pre_registered_protocol"]
    candidates = reupload_report["training_only_reuploading_cv"]["aggregate"][
        "candidates"
    ]
    if not any(row["architecture"] == DEFAULT_ARCHITECTURE for row in candidates):
        raise RunnerError("RX(0.5x) is absent from the re-uploading report")
    return {
        "dataset": "PBMC68k",
        "positive_label": str(source_config["positive_label"]),
        "negative_label": str(source_config["negative_label"]),
        "qubits": int(source_config["qubits"]),
        "seed": int(source_config["seed"]),
        "value_mode": str(source_config["value_mode"]),
        "max_active_genes": int(source_config["max_active_genes"]),
        "single_scale": float(protocol["single_scale_frozen"]),
        "phase_scale": float(protocol["phase_scale_frozen"]),
        "pair_scale": float(protocol["pair_scale_frozen"]),
        "post_axis": "rx",
        "post_scale": 0.5,
        "architecture": DEFAULT_ARCHITECTURE,
    }


def build_seed_circuits(
    *,
    encoded_train: np.ndarray,
    encoded_test: np.ndarray,
    y_train: np.ndarray,
    y_test: np.ndarray,
    train_indices: Sequence[int],
    test_indices: Sequence[int],
    configuration: Mapping[str, Any],
    shot_intent: int,
    z_quota: int,
    transverse_quota: int,
    multiqubit_quota: int,
    sensitivity_threshold: float,
    seed_transpiler: int = q60.DEFAULT_SEED_TRANSPILER,
) -> tuple[list[str], list[dict[str, Any]], dict[str, Any]]:
    """Build one frozen seed batch with training-only balanced observables."""

    encoded_train = np.asarray(encoded_train, dtype=np.float64)
    encoded_test = np.asarray(encoded_test, dtype=np.float64)
    y_train = np.asarray(y_train, dtype=np.float64)
    y_test = np.asarray(y_test, dtype=np.float64)
    if encoded_train.ndim != 2 or encoded_test.ndim != 2:
        raise RunnerError("Encoded PBMC matrices must be two-dimensional")
    if encoded_train.shape[1] != encoded_test.shape[1]:
        raise RunnerError("Train/test encoded dimensions differ")
    if len(encoded_train) != len(y_train) or len(encoded_test) != len(y_test):
        raise RunnerError("Encoded matrices and labels differ in length")

    sketch_linear, sketch_pair = q60.sketch_parameters(encoded_train, y_train)
    base_rows: list[dict[str, Any]] = [
        {
            "circuit": rx05_circuit(
                sketch_linear,
                sketch_pair,
                single_scale=float(configuration["single_scale"]),
                phase_scale=float(configuration["phase_scale"]),
                pair_scale=float(configuration["pair_scale"]),
            ),
            "parameters": (sketch_linear, sketch_pair),
            "role": "weighted_training_sketch",
            "split": "train",
            "sample_position": None,
            "source_row_index": None,
            "label": None,
        }
    ]
    for split, encoded, labels, indices in (
        ("train", encoded_train, y_train, train_indices),
        ("test", encoded_test, y_test, test_indices),
    ):
        for position, (sample, label, source_index) in enumerate(
            zip(encoded, labels, indices, strict=True)
        ):
            linear, pair = q60.query_parameters(sample)
            base_rows.append(
                {
                    "circuit": rx05_circuit(
                        linear,
                        pair,
                        single_scale=float(configuration["single_scale"]),
                        phase_scale=float(configuration["phase_scale"]),
                        pair_scale=float(configuration["pair_scale"]),
                    ),
                    "parameters": (linear, pair),
                    "role": "query",
                    "split": split,
                    "sample_position": int(position),
                    "source_row_index": int(source_index),
                    "label": float(label),
                }
            )

    mappings = toy.pauli_feature_mappings(encoded_train.shape[1], family="local")
    parameter_rows = [row["parameters"] for row in base_rows]
    entangled_features = reupload.reupload_feature_matrix(
        parameter_rows,
        mappings,
        configuration=configuration,
        architecture=DEFAULT_ARCHITECTURE,
        pair_scale=float(configuration["pair_scale"]),
    )
    zero_features = reupload.reupload_feature_matrix(
        parameter_rows,
        mappings,
        configuration=configuration,
        architecture=DEFAULT_ARCHITECTURE,
        pair_scale=0.0,
    )
    train_start = 1
    train_stop = train_start + len(encoded_train)
    test_stop = train_stop + len(encoded_test)
    selected, selection_scores, selection_audit = (
        balanced.select_balanced_train_only_features(
            entangled_features[train_start:train_stop],
            zero_features[train_start:train_stop],
            y_train,
            mappings=mappings,
            z_quota=int(z_quota),
            transverse_quota=int(transverse_quota),
            multiqubit_quota=int(multiqubit_quota),
            shot_intent=int(shot_intent),
            sensitivity_threshold=float(sensitivity_threshold),
        )
    )
    model = entangled_features[0, selected]
    query_train = entangled_features[train_start:train_stop, selected]
    query_test = entangled_features[train_stop:test_stop, selected]
    head_train = q60._head_features(model, query_train)
    head_test = q60._head_features(model, query_test)
    train_scores, test_scores = q60._ridge_scores(head_train, head_test, y_train)

    qasms: list[str] = []
    manifest: list[dict[str, Any]] = []
    seed = int(configuration["seed"])
    for base_index, row in enumerate(base_rows):
        logical_metrics = q60.q40_validate.circuit_metrics(row["circuit"])
        for basis in q60.MEASUREMENT_BASES:
            measured = q60.measurement_circuit_for_basis(row["circuit"], basis)
            qasm, qasm_metadata = q60.q40_validate.export_numeric_qasm2(
                measured, seed_transpiler=int(seed_transpiler)
            )
            qasms.append(qasm)
            manifest.append(
                {
                    "circuit_index": len(qasms) - 1,
                    "seed": seed,
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
            "candidate_index": int(index),
            "selection_score": float(selection_scores[int(index)]),
            "measurement_basis": q60.measurement_basis_for_mapping(
                mappings[int(index)]
            ),
            "pauli_mapping": q60._normalized_mapping(mappings[int(index)]),
        }
        for index in selected
    ]
    return qasms, manifest, {
        "seed": seed,
        "logical_base_circuit_count": len(base_rows),
        "measurement_bases": list(q60.MEASUREMENT_BASES),
        "measured_circuit_count": len(qasms),
        "candidate_observable_count": len(mappings),
        "selected_observable_count": len(selected),
        "selected_observables": selected_rows,
        "selection_protocol": {
            "uses_training_labels_only": True,
            "uses_test_inputs": False,
            "uses_test_labels": False,
            "representation": "balanced_8_8_8",
            "future_shot_intent": int(shot_intent),
        },
        "selection_audit": selection_audit,
        "ideal_causal_cone_screen": {
            "method": "exact statevector on at-most-four-qubit causal cones",
            "train_balanced_accuracy": q60._balanced_accuracy(y_train, train_scores),
            "test_balanced_accuracy": q60._balanced_accuracy(y_test, test_scores),
            "claim_boundary": "Noiseless local diagnostic only; not a hardware prediction.",
        },
        "encoded_train_sha256": q60.q40_validate._array_sha256(encoded_train),
        "encoded_test_sha256": q60.q40_validate._array_sha256(encoded_test),
    }


def prepare_batch(args: argparse.Namespace) -> tuple[list[str], dict[str, Any]]:
    source_path = Path(args.source_tuning_report)
    reupload_path = Path(args.reupload_report)
    source_report = json.loads(source_path.read_text(encoding="utf-8"))
    reupload_report = json.loads(reupload_path.read_text(encoding="utf-8"))
    configuration = _configuration_from_reports(source_report, reupload_report)

    x, labels, source_meta = pbmc.load_pbmc68k(cache_dir=str(args.cache_dir))
    x_pair, y_pair, pair_meta = pbmc.select_binary_pair(
        x,
        labels,
        positive_label=str(configuration["positive_label"]),
        negative_label=str(configuration["negative_label"]),
    )
    train_indices = np.asarray(
        source_report["split"]["train_indices"], dtype=np.int64
    )
    test_indices = np.asarray(
        source_report["split"]["test_indices"], dtype=np.int64
    )
    encoded_train, train_stats = pairwise_screen.build_pairwise_hashed_matrix(
        x_pair[train_indices],
        feature_dim=int(configuration["qubits"]),
        hash_seed=int(configuration["seed"]),
        value_mode=str(configuration["value_mode"]),
        max_active_genes=int(configuration["max_active_genes"]),
    )
    encoded_test, test_stats = pairwise_screen.build_pairwise_hashed_matrix(
        x_pair[test_indices],
        feature_dim=int(configuration["qubits"]),
        hash_seed=int(configuration["seed"]),
        value_mode=str(configuration["value_mode"]),
        max_active_genes=int(configuration["max_active_genes"]),
    )
    expected_train_hash = str(source_report["split"]["encoded_train_sha256"])
    expected_test_hash = str(source_report["split"]["encoded_test_sha256"])
    if q60.q40_validate._array_sha256(encoded_train) != expected_train_hash:
        raise RunnerError("Seed-11 encoded training hash does not reproduce")
    if q60.q40_validate._array_sha256(encoded_test) != expected_test_hash:
        raise RunnerError("Seed-11 encoded test hash does not reproduce")

    qasms, manifest, seed_meta = build_seed_circuits(
        encoded_train=encoded_train,
        encoded_test=encoded_test,
        y_train=y_pair[train_indices].astype(np.float64),
        y_test=y_pair[test_indices].astype(np.float64),
        train_indices=train_indices,
        test_indices=test_indices,
        configuration=configuration,
        shot_intent=int(args.readout_shots),
        z_quota=int(args.z_quota),
        transverse_quota=int(args.transverse_quota),
        multiqubit_quota=int(args.multiqubit_quota),
        sensitivity_threshold=float(args.sensitivity_threshold),
        seed_transpiler=int(args.seed_transpiler),
    )
    if not qasms or len(qasms) > q60.FIRE_OPAL_MAX_BATCH:
        raise RunnerError("Fire Opal batch is outside the supported size")

    payload_depths = [int(row["metrics"]["depth"]) for row in manifest]
    logical_depths = [
        int(row["logical_metrics_before_measurement"]["depth"])
        for row in manifest
    ]
    qasm_hashes = [row["qasm_sha256"] for row in manifest]
    aggregate_sha256 = q60.q40_validate._sha256_bytes(
        json.dumps(qasm_hashes, separators=(",", ":")).encode("utf-8")
    )
    expected_counts = {
        "single_z": int(args.z_quota),
        "pair_sensitive_local_xy": int(args.transverse_quota),
        "pair_sensitive_multiqubit": int(args.multiqubit_quota),
    }
    selected_counts = seed_meta["selection_audit"]["selected_counts"]
    structural_gate = all(
        int(selected_counts.get(name, -1)) == count
        for name, count in expected_counts.items()
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
            row["metrics"]["num_qubits"] == int(configuration["qubits"])
            for row in manifest
        ),
        "all_one_classical_register": all(
            row["classical_register_count"] == 1 for row in manifest
        ),
        "all_virtual_qubits_only": all(
            row["virtual_qubits_only"] for row in manifest
        ),
        "all_parameters_numeric": all(
            row["all_parameters_numeric"] for row in manifest
        ),
        "all_round_trips_validated": all(
            row["round_trip_validated"] for row in manifest
        ),
        "all_expected_measurement_bases": (
            {row["measurement_basis"] for row in manifest}
            == set(q60.MEASUREMENT_BASES)
        ),
        "balanced_representation_structural_gate": structural_gate,
        "balanced_representation_expected_counts": expected_counts,
        "source_hashes_reproduced": True,
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
        and structural_gate
    )
    config = {
        **configuration,
        "train_samples": int(len(train_indices)),
        "test_samples": int(len(test_indices)),
        "observable_quotas": {
            "single_z": int(args.z_quota),
            "pair_sensitive_local_xy": int(args.transverse_quota),
            "pair_sensitive_multiqubit": int(args.multiqubit_quota),
        },
        "future_readout_shots": int(args.readout_shots),
        "backend": str(args.backend),
        "entangler_schedule": "even_then_odd_nearest_neighbour_rzz",
        "logical_depth_intent": 6,
    }
    seed_meta["train_indices"] = [int(value) for value in train_indices]
    seed_meta["test_indices"] = [int(value) for value in test_indices]
    seed_meta["train_encoding_stats"] = train_stats
    seed_meta["test_encoding_stats"] = test_stats
    return qasms, {
        "config": config,
        "source": {**source_meta, **pair_meta},
        "source_artifacts": {
            "source_tuning_report": str(source_path.resolve()),
            "source_tuning_report_sha256": q60.q40_validate._sha256_file(source_path),
            "reupload_report": str(reupload_path.resolve()),
            "reupload_report_sha256": q60.q40_validate._sha256_file(reupload_path),
            "dataset_cache": q60.q40_validate._dataset_artifacts(args.cache_dir),
        },
        "seeds": [seed_meta],
        "manifest": manifest,
        "local_validation": local_validation,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", type=Path, default=Path("data_cache/pbmc68k"))
    parser.add_argument(
        "--source-tuning-report", type=Path, default=DEFAULT_SOURCE_TUNING
    )
    parser.add_argument("--reupload-report", type=Path, default=DEFAULT_REUPLOAD_REPORT)
    parser.add_argument("--z-quota", type=int, default=balanced.DEFAULT_Z_QUOTA)
    parser.add_argument(
        "--transverse-quota", type=int, default=balanced.DEFAULT_TRANSVERSE_QUOTA
    )
    parser.add_argument(
        "--multiqubit-quota", type=int, default=balanced.DEFAULT_MULTIQUBIT_QUOTA
    )
    parser.add_argument(
        "--sensitivity-threshold",
        type=float,
        default=balanced.DEFAULT_SENSITIVITY_THRESHOLD,
    )
    parser.add_argument("--readout-shots", type=int, default=DEFAULT_READOUT_SHOTS)
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
    if (
        args.readout_shots < 1
        or args.max_payload_depth < 1
        or args.z_quota < 1
        or args.transverse_quota < 1
        or args.multiqubit_quota < 1
    ):
        raise RunnerError("Shots, quotas, and depth gate must be positive")
    for path in (args.bundle, args.output):
        if path.exists() and not args.force:
            raise RunnerError(f"Refusing to overwrite existing artifact: {path}")

    started = time.perf_counter()
    qasms, prepared = prepare_batch(args)
    bundle = q60.write_qasm_bundle(args.bundle, qasms, prepared)
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
        "kind": "pbmc68k_q60_rx05_fireopal_validate_only",
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
        "seeds": prepared["seeds"],
        "local_validation": prepared["local_validation"],
        "qasm_bundle": bundle,
        "provider_validation": provider,
        "elapsed_seconds": time.perf_counter() - started,
        "claim_boundary": (
            "A passing provider validation establishes RX(0.5x) input "
            "compatibility only. It is not hardware execution, mitigation "
            "improvement, or evidence of quantum advantage."
        ),
    }
    q60.q40_validate._atomic_write_json(args.output, report)
    print("PBMC68k q60 RX(0.5x) Fire Opal validation route")
    print(f"- circuits: {len(qasms)}")
    print(
        "- max payload depth: "
        f"{prepared['local_validation']['max_payload_depth']}"
    )
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
